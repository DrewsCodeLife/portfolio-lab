import pandas as pd
import numpy as np

df = pd.read_csv("data/generated/derived_frontiers.csv")
cleaned = pd.read_csv("data/cleaned/daily_returns.csv")

cleaned = cleaned.iloc[1:].reset_index(drop=True)
print(cleaned.head)

STARTING_WEALTH = 10000
classes = ["US_Equity", "International", "Bonds", "REITs", "Cash"]

# This line is from ChatGPT, weights are saved as space separated str
pf_series = df["weights"].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

portfolios = np.vstack(pf_series.to_numpy())  # shape: (n_portfolios, 5)

# Input values are proportions of invested wealth, so we can trivially multiply.
portfolios = portfolios * STARTING_WEALTH

pfs_over_time = []

for _, row in cleaned.iterrows():
  # Assumes that data is aligned, matching mu in markowitz.py
  portfolios = portfolios * (1.0 + row[classes].values)
  pfs_over_time.append(portfolios.sum(axis=1).copy())

final_values = portfolios.sum(axis=1)

print(final_values)

pfs_save = pd.DataFrame(pfs_over_time)
pfs_save.to_csv("simulated_portfolios.csv")

print(f"Best portfolio in simulation: idx {np.argmax(final_values)}, final value {max(final_values)}")
