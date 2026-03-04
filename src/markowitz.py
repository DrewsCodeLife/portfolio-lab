import pandas as pd
import numpy as np
import cvxpy as cp
import math

from pandas.api.types import is_numeric_dtype
from sklearn.covariance import LedoitWolf
from scipy.linalg import solve

np.random.seed(42)

data    = pd.read_csv("data/cleaned/cleaned.csv")
returns = pd.read_csv("data/cleaned/daily_returns.csv")

dates = data["Date"]

data = data.loc[:, data.columns != "Date"]
returns = returns.loc[:, returns.columns != "Date"]

# First day has no 'daily return', so we drop it
data = data.iloc[1:].reset_index(drop=True)
returns = returns.iloc[1:].reset_index(drop=True)

T = len(returns)
N = returns.shape[1]

for c in returns.columns:
  if not is_numeric_dtype(returns[c]):
    print("ERROR: NON-NUMERIC COLUMN: ", c, "\nData type: ", returns[c].dtype)
    exit(-1)

def estimate_mu_sigma(rets, lookback=None, shrink=True):
  """
  Estimate mean values and covariance matrix of daily returns
  Inputs:
    - rets    : df of daily returns
    - shrink  : Whether or not to use Ledoit-Wolf shrinkage on covariance
    - lookback: Number of rows to use, if None uses all
                - Essentially defining the training data set
  """
  if lookback is not None:
    local_data = rets.iloc[-lookback:]
  else:
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

def compute_markowitz_constants(mu, sigma):
  """
  Computes various constants used by the Markowitz optimization process
  """
  ones = np.ones((sigma.shape[0], 1))     # It's a vector of ones
  x = solve(sigma, ones, assume_a='pos')  # 'pos' = symmetric positive definite (SPD)
  y = solve(sigma, mu, assume_a='pos')    # Covariance matrix is inherently SPD
  A = (ones.T @ x).item() # Matrix multiply ones.T and x, fetch the resultant value
  B = (ones.T @ y).item()
  C = (mu.T @ y).item()
  Delta = A * C - B * B
  return {
    "A": A,
    "B": B,
    "C": C,
    "Delta":
    Delta,
    "Sigma_inv_1": x,
    "Sigma_inv_mu": y,
    "ones": ones
    }


def solve_min_variance_long_only(Sigma, w_max=None, solver=None):
    """
    Minimum-variance portfolio with long-only constraints.
    - This is the 'safest' portfolio that should change the least (positive or negative)

    Solves:
        minimize      : w^T Sigma w
        Constrained by: sum(w) = 1
                        w >= 0
                        (optional) w <= w_max

    Parameters
    ----------
    Sigma : (N, N) ndarray
        Covariance matrix (daily).
    w_max : float or array-like of shape (N,), optional
        Upper bound(s) on weights. Prevent the model from overfixating on one asset class.
    solver : str
        OSQP is good since we expect a quadratic-like problem

    Returns
    -------
    w_value : (N,) ndarray
        Optimal weights, or None if infeasible/failed.
    status : str
        CVXPY solver status string.
    """
    N = Sigma.shape[0]

    w = cp.Variable(N)  # Portfolio weights

    #cp.quad_form(w, Sigma) assumes we want w^T Sigma w
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = []                    # Construct constraints
    constraints.append(cp.sum(w) == 1)  # fully invested
    constraints.append(w >= 0)          # long-only (no shorting)

    # Apply asset participation cap if provided
    if w_max is not None:
        # If w_max is a scalar (single value), CVXPY broadcasts it to all classes
        constraints.append(w <= w_max)

    problem = cp.Problem(objective, constraints)

    if solver is not None:
      problem.solve(solver=solver, verbose=False)
    else:
      problem.solve(verbose=False)

    # If solver succeeded, return numeric weights
    if w.value is None:
        return None, problem.status

    return np.asarray(w.value).ravel(), problem.status


def solve_frontier_point_long_only(mu, Sigma, R_target, w_max=None, solver=None):
    """
    Given a target return (daily) that the user wants, provide an 'optimized' portfolio

    Solves:
        minimize      : w^T Sigma w
        Constrained by: sum(w) = 1
                        w >= 0
                        mu^T w = R_target
                        (optional) w <= w_max

    Parameters
    ----------
    mu : (US_Equity, International, Bonds, REITs, Cash) ndarray
        Expected daily returns
    Sigma : (N,N) ndarray
        Covariance matrix
    R_target : float
        User requested daily return
    w_max : float or (N,) array, optional
        Weight caps.
    solver : str
        Solver to use. OSQP is just as good here as it was for min var

    Returns
    -------
    w_value : (N,) ndarray or None
        Optimal weights, or None if infeasible/failed.
    status : str
        CVXPY status.
    """
    mu = np.asarray(mu).reshape(-1)  # ensure shape (N,)
    N = Sigma.shape[0]

    w = cp.Variable(N)

    objective = cp.Minimize(cp.quad_form(w, Sigma))

    constraints = [
        cp.sum(w) == 1,      # fully invested
        w >= 0,              # long-only
        mu @ w == R_target   # hit target return exactly
    ]

    if w_max is not None:
        constraints.append(w <= w_max)

    problem = cp.Problem(objective, constraints)
    if solver is not None:
      problem.solve(solver=solver, verbose=False)
    else:
      problem.solve(verbose=False)

    if w.value is None:
        return None, problem.status

    return np.asarray(w.value).ravel(), problem.status


def compute_portfolio_stats(mu, Sigma, w, annualize=True, trading_days=252):
    """
    Utility helper: compute expected return and volatility for weights w.

    mu: expected daily returns
    Sigma: daily covariance
    w: weights

    Returns dict with daily and (optionally) annualized metrics.
    """
    mu = np.asarray(mu).reshape(-1)
    w = np.asarray(w).reshape(-1)

    exp_ret_daily = float(mu @ w)
    var_daily = float(w.T @ Sigma @ w)
    vol_daily = float(np.sqrt(max(var_daily, 0.0)))

    out = {
        "exp_return_daily": exp_ret_daily,
        "vol_daily": vol_daily,
    }

    if annualize:
        out["exp_return_ann"] = exp_ret_daily * trading_days
        out["vol_ann"] = vol_daily * np.sqrt(trading_days)

    return out


def build_long_only_frontier(mu, Sigma, n_points=50, w_max=None, solver=None):
    """
    Build a long-only efficient frontier by sweeping target returns.

    Important detail:
    - With long-only constraints, not every target return is feasible.
    - The feasible expected-return range is roughly [min(mu), max(mu)] under long-only,
      but in practice the covariance + sum-to-1 can still make some exact equality targets
      numerically finicky.
    - So we generate targets inside a "safe" range and skip infeasible solves.

    Returns
    -------
    frontier : list of dicts
        Each dict includes weights, target return, realized return/vol, status.
    """
    mu = np.asarray(mu).reshape(-1)

    mu_min = float(mu.min())
    mu_max = float(mu.max())

    # Generate safe range
    eps = 1e-12
    R_grid = np.linspace(mu_min + eps, mu_max - eps, n_points)

    frontier = []

    for R_target in R_grid:
        w, status = solve_frontier_point_long_only(mu, Sigma, R_target, w_max=w_max, solver=solver)

        if w is None:
            # Infeasible target return for the constraints, skip it.
            continue

        stats = compute_portfolio_stats(mu, Sigma, w, annualize=True)
        frontier.append({
            "target_R": float(R_target),
            "status": status,
            "weights": w,
            **stats
        })

    return frontier


# mu is ndarray (US_Equity, International, Bonds, REITs, Cash)
# lookback is user defined (longer results in more historically averaged portfolio)
# - Will need to constrain in UI relative to dataset
# - Long term is more data which is often considered better, but it also means an average
#     derived from older data, meaning it may not reflect future conditions as well
# - Short term simply might result in a bad model
lookback = 252
mu, Sigma = estimate_mu_sigma(returns, lookback=lookback, shrink=True)

consts = compute_markowitz_constants(mu, Sigma)

# If the condition number is too large, we've got a problem
cond_number = np.linalg.cond(Sigma)
print(f"Assets: {returns.shape[1]}, Lookback days: {lookback}, Condition number of Sigma: {cond_number:.3e}")
print("A, B, C, Delta:", consts['A'], consts['B'], consts['C'], consts['Delta'])

if cond_number >= math.pow(10, 5):
  print(f"WARNING: Condition number is greater than 100,000: {cond_number}")

mu_vec = mu.ravel()  # Reshape mu

# Find minimum variance portfolio
# - Probably useful in the UI as a contrast
w_mv, st = solve_min_variance_long_only(Sigma, w_max=None)  # optionally w_max=0.6
print("MV status:", st, "sum:", None if w_mv is None else w_mv.sum())
if w_mv is not None:
    print("MV stats:", compute_portfolio_stats(mu_vec, Sigma, w_mv))

# Now construct the actual efficient frontier, using 50 points to approximate the curve
frontier = build_long_only_frontier(mu_vec, Sigma, n_points=50, w_max=None)
print("Frontier points (feasible):", len(frontier))

if len(frontier) >= 1:
  print("First frontier point:", frontier[0]["exp_return_ann"], frontier[0]["vol_ann"], frontier[0]["weights"])
else:
   print(
        "Well darn, cvxpy determined the problem was literally unsolvable.\n" \
        + "For literally every target return. Nice one bud."
      )

frontier_df = pd.DataFrame(frontier)

TRADING_DAYS = 252  # How many trading days per year?
frontier_df["exp_return_ann"] = frontier_df["exp_return_daily"] * TRADING_DAYS
frontier_df["vol_ann"] = frontier_df["vol_daily"] * math.sqrt(TRADING_DAYS)

# Show the first few frontier points and the min variance portfolio
pd.set_option('display.precision', 6)
print("\nMinimum-variance weights (sum to):", w_mv.sum())
print(pd.Series(w_mv, index=returns.columns))

print(frontier_df.head())

frontier_df_head = frontier_df[["target_R", "exp_return_ann", "vol_ann"]].head(12).copy()
frontier_df_head.columns = ["target_R_daily", "exp_return_ann", "vol_ann"]
print(frontier_df_head)

frontier_df.to_csv("derived_frontiers.csv")
