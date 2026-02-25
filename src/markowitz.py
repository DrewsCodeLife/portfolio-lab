import pandas as pd
import numpy as np
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
data.iloc[1:].reset_index(drop=True)
returns.iloc[1:].reset_index(drop=True)

T = len(returns)
N = returns.shape[1]

for c in returns.columns:
  if not is_numeric_dtype(returns[c]):
    print("ERROR: NON-NUMERIC COLUMN: ", c, "\ndtype: ", returns[c].dtype)
    exit(-1)

def estimate_mu_sigma(rets, lookback=None, shrink=True):
  """
  Estimate mean values and covariance matrix of daily returns
  - rets    : DataFrame of daily returns
  - shrink  : Whether or not to use Ledoit-Wolf shrinkage on covariance
  - lookback: Number of rows to use, if None uses all
  """
  if lookback is not None:
    local_data = rets.iloc[-lookback:]
  else:
    local_data = rets.copy()

  mu = rets.mean().values.reshape(-1, 1) # (5x1 vector, reshaped for matrix mult)

  if shrink:
    lw = LedoitWolf().fit(local_data.values)
    sigma = lw.covariance_
  else:
    sigma = np.cov(local_data.values, rowvar=False, ddof=1)

  # Coerce the matrix into a symmetric state
  sigma = (sigma + sigma.T) / 2
  return mu, sigma

def compute_markowitz_constants(mu, sigma):
  """
  Computes various constant used by our Markowitz optimization process
  Return: The constants
  """
  ones = np.ones((sigma.shape[0], 1))
  x = solve(sigma, ones, assume_a='pos')
  y = solve(sigma, mu, assume_a='pos')
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

def target_return_weights(R, constants):
  """
  Find the weight vector for target return R
  Return: Weights as 1D array
  """
  A = constants["A"]
  B = constants["B"]
  C = constants["C"]
  Delta = constants["Delta"]
  x = constants["Sigma_inv_1"]
  y = constants["Sigma_inv_mu"]
  w = ((C - B * R) / Delta) * x + ((A * R - B) / Delta) * y
  return w.ravel() # Return flattened view of arr

def min_var_weight(constants):
  x = constants["Sigma_inv_1"]
  A = constants["A"]
  return (x / A).ravel()

def tang_weight(mu, constants, r_f=0.0):
  ones = constants["ones"]
  y = constants["Sigma_inv_mu"]
  x = constants["Sigma_inv_1"]
  vec = y - r_f * x
  denom = (ones.T @ vec).item()
  return (vec / denom).ravel()


lookback = 252  # User defined
mu, Sigma = estimate_mu_sigma(returns, lookback=lookback, shrink=True)

consts = compute_markowitz_constants(mu, Sigma)

cond_number = np.linalg.cond(Sigma)
print(f"Assets: {returns.shape[1]}, Lookback days: {lookback}, Condition number of Sigma: {cond_number:.3e}")
print("A, B, C, Delta:", consts['A'], consts['B'], consts['C'], consts['Delta'])

w_mv = min_var_weight(consts)
w_t = tang_weight(mu, consts, r_f=0.0)

# compute frontier by sweeping target returns
mu_vals = mu.ravel()
R_min = mu_vals.min() * 0.8   # a bit below min mean
R_max = mu_vals.max() * 1.2   # a bit above max mean
R_grid = np.linspace(R_min, R_max, 80)

frontier = []
for R in R_grid:
    w = target_return_weights(R, consts)
    exp_return = float(mu.ravel() @ w)    # daily expected return
    vol = math.sqrt(float(w.T @ Sigma @ w))  # daily volatility
    frontier.append((R, exp_return, vol, w.copy()))

frontier_df = pd.DataFrame({
    "target_R": [f[0] for f in frontier],
    "exp_return_daily": [f[1] for f in frontier],
    "vol_daily": [f[2] for f in frontier],
})

TRADING_DAYS = 252
frontier_df["exp_return_ann"] = frontier_df["exp_return_daily"] * TRADING_DAYS
frontier_df["vol_ann"] = frontier_df["vol_daily"] * math.sqrt(TRADING_DAYS)

# Show the first few frontier points and the special portfolios
pd.set_option('display.precision', 6)
print("\nMinimum-variance weights (sum to):", w_mv.sum())
print(pd.Series(w_mv, index=returns.columns))
print("\nTangency weights (sum to):", w_t.sum())
print(pd.Series(w_t, index=returns.columns))

print(frontier_df.head())

frontier_df_head = frontier_df[["target_R", "exp_return_ann", "vol_ann"]].head(12).copy()
frontier_df_head.columns = ["target_R_daily", "exp_return_ann", "vol_ann"]
print(frontier_df_head)

weights_list = [f[3] for f in frontier]
weights_df = pd.DataFrame(weights_list, columns=returns.columns)
print(weights_df)

print("\nDone")