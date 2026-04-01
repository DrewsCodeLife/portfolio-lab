import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

import pickle
import math
import sys
import os

# Load parent path and temporarily add it to sys path so we can use markowitz.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import the file from the parent directory
import markowitz as mkw

np.random.seed(42)


# |------------------------| Constants & globals |------------------------|
STARTING_WEALTH = 10000

RUN_OPTIMIZATION = False
RUN_SIMULATION = False

PLOT_SELECTOR = [False, False, True, False]

data  = pd.read_csv("data/cleaned/cleaned.csv")

dates = data["Date"]
data = data.loc[:, data.columns != "Date"]

tckrs = data.loc[:, data.columns != "DTB3"]
ret = tckrs.pct_change()

dtb3_daily = (data["DTB3"] / 100) / 252

ret = ret.join(dtb3_daily.rename("DTB3"))

# First day has no 'daily return', so we drop it
data = data.iloc[1:].reset_index(drop=True)
ret = ret.iloc[1:].reset_index(drop=True)

T = len(ret)
N = ret.shape[1]

asset_names = ['BIL', 'BND', 'DTB3', 'EFA', 'RWR', 'SPY', 'VBMFX', 'VNQ', 'VTI', 'VXUS']
# |------------------------| Constants & globals |------------------------|


# |--------------------------| Helper Functions |--------------------------|
def fancy_estimate_mu_sigma(first, last, rets=ret, shrink=True):
    """
    Estimate mean values and covariance matrix of daily returns
    Inputs:
        - rets    : df of daily returns
        - shrink  : Whether or not to use Ledoit-Wolf shrinkage on covariance
        - lookback: Number of rows to use, if None uses all
                    - Essentially defining the training data set
    """
    if first is not None and last is not None:
        # Arr slicing is exclusive of second argument, we step 1 past to include it
        local_data = rets.iloc[first:last + 1]
    else:
        # Fallback to the whole dataset.
        #   - Leftover from old code, shouldn't happen here.
        local_data = rets.copy()

    mu = local_data.mean().values.reshape(-1, 1) # (5x1 vector, reshaped for matrix mult)

    if shrink:
        lw = LedoitWolf().fit(local_data.values)
        sigma = lw.covariance_
    else:
        sigma = np.cov(local_data.values, rowvar=False, ddof=1)

    # This is a covariance matrix, it'd better already be symmetric.
    # Nonetheless, we explicitly make it symmetric for safety.
    # - If it weren't symmetric, there could be multiple local optima,
    #     And I'm pretty sure the solver would do this anyway.
    sigma = (sigma + sigma.T) / 2
    return mu, sigma


def rolling_lookback(lookback=252):
    """
    Starting from the first date, train markowitz optimization models on a
        rolling lookback window.
    """
    min_var_list = []
    frontier_list = []

    current_day = 0
    while (current_day + lookback) <= (len(dates) - lookback):
        # Default to global ret & shrink=True
        mu, sigma = fancy_estimate_mu_sigma(current_day, current_day + lookback)

        cond_number = np.linalg.cond(sigma)
        print(f"Lookback period: {dates[current_day]} -> {dates[current_day + lookback]}, " \
               + f"Condition number of Sigma: {cond_number:.3e}")

        if cond_number >= math.pow(10, 5):
            print(f"WARNING: Condition number is greater than 100,000: {cond_number}")

        mu_vec = mu.ravel()  # Reshape mu

        # Find minimum variance portfolio
        w_mv, st = mkw.solve_min_variance_long_only(sigma, w_max=None)

        frontier = mkw.build_long_only_frontier(mu_vec, sigma, n_points=10, w_max=None)

        if len(frontier) < 1:
            print("\n\nERROR: Lookback period was unsolvable over 50 portfolios\n\n")

        # Min var is not guaranteed to have solved. If it failed, w_mv is nan.
        min_var_list.append(w_mv)
        frontier_list.append(frontier)

        current_day = current_day + 1
    return min_var_list, frontier_list


def simulate_portfolios(min_var_list, frontier_list, sim_length=252):
    period = ret.iloc[-sim_length:]

    print("Simulating Minimum Variance Portfolios")
    min_var_pfs_ot = []
    ctr = 0
    for arr in min_var_list:
        print(f"\r{((ctr / 3246) * 100):.2f}% mv", flush=True)
        pf = arr * STARTING_WEALTH

        pf_over_time = []
        for _, row in period.iterrows():
            # Assumes that data is aligned, matching mu in markowitz.py
            pf = pf * (1.0 + row[asset_names].values)
            pf_over_time.append(pf.sum())

        ctr = ctr + 1
        min_var_pfs_ot.append(pf_over_time)


    print("Simulating Frontier Portfolios")
    frontier_pfs_ot = []
    ctr = 0
    for list_of_pfs in frontier_list:
        one_group = []
        for opt in list_of_pfs:
            print(f"\r{((ctr / 32460) * 100):.2f}% ftr", flush=True)
            pf = opt['weights'] * STARTING_WEALTH

            pf_over_time = []
            for _, row in period.iterrows():
                # Assumes that data is aligned, matching mu in markowitz.py
                pf = pf * (1.0 + row[asset_names].values)
                pf_over_time.append(pf.sum())

            ctr = ctr + 1
            one_group.append(pf_over_time)
        frontier_pfs_ot.append(one_group)


    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    pf = weights * STARTING_WEALTH

    print("Simulating Naive Portfolio")
    naive_pf_ot = []
    for _, row in period.iterrows():
        pf = pf * (1.0 + row[asset_names].values)
        naive_pf_ot.append(pf.sum())

    return min_var_pfs_ot, frontier_pfs_ot, naive_pf_ot
# |--------------------------| Helper Functions |--------------------------|


# |--------------------------| Main functionality |--------------------------|
if RUN_OPTIMIZATION:
    min_var_list, frontier_list = rolling_lookback()

    with open("min_var_list.pkl", 'wb') as file:
        pickle.dump(min_var_list, file)
    with open("frontier_list.pkl", 'wb') as file:
        pickle.dump(frontier_list, file)
else:
    with open("min_var_list.pkl", 'rb') as file:
        min_var_list = pickle.load(file)
    with open("frontier_list.pkl", 'rb') as file:
        frontier_list = pickle.load(file)

if RUN_SIMULATION:
    mv_pfs, ftr_pfs, naive_pf = simulate_portfolios(min_var_list, frontier_list)

    with open("min_variance_portfolios_OT.pkl", 'wb') as file:
        pickle.dump(mv_pfs, file)
    with open("frontier_portfolios_OT.pkl", 'wb') as file:
        pickle.dump(ftr_pfs, file)
    with open("naive_portfolio_OT.pkl", 'wb') as file:
        pickle.dump(naive_pf, file)
    print("Saved portfolios over time")
else:
    with open("min_variance_portfolios_OT.pkl", 'rb') as file:
        mv_pfs = pickle.load(file)
    with open("frontier_portfolios_OT.pkl", 'rb') as file:
        ftr_pfs = pickle.load(file)
    with open("naive_portfolio_OT.pkl", 'rb') as file:
        naive_pf = pickle.load(file)

# window is a 252 trading period for each mv portfolio, -1 takes last day of each window
mv_final  = [window[-1] for window in mv_pfs]
ftr_final = [[path[-1] for path in window] for window in ftr_pfs]
naive_final = naive_pf[-1]

if PLOT_SELECTOR[0]:
    plt.hist(mv_final)
    plt.savefig("min_var_ret_dist.png")
    plt.show()

if PLOT_SELECTOR[1]:
    plt.hist(ftr_final)
    plt.savefig("ftr_pf_ret_dist.png")
    plt.show()

if PLOT_SELECTOR[2]:
    combined = []
    for outcome in mv_final:
        combined.append((outcome, "min_var"))

    for group in ftr_final:
        for outcome in group:
            print(outcome)
            combined.append((outcome, "frontier"))

    df = pd.DataFrame(combined, columns=["outcome", "group"])

    mv_vals  = df.loc[df["group"] == "min_var", "outcome"]
    ftr_vals = df.loc[df["group"] == "frontier", "outcome"]

    all_vals = df["outcome"]
    bins = np.histogram_bin_edges(all_vals, bins="fd")

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].hist(mv_vals, bins=bins, density=True)
    axes[0].set_title("Min Variance Outcomes")

    axes[1].hist(ftr_vals, bins=bins, density=True)
    axes[1].set_title("Frontier Outcomes")

    plt.savefig('min_var_and_ftr_outcomes.png')
    plt.show()

if PLOT_SELECTOR[3]:
    plt.plot(ftr_pfs[-1][-1])
    plt.plot(naive_pf, label='baseline')
    plt.legend()
    plt.savefig("most_recent_lookback_pfs.png")
    plt.show()