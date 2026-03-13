import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# cols are portfolios, row is value at a timestep
res = pd.read_csv("data/generated/simulated_portfolios.csv")
pfs = pd.read_csv("data/generated/derived_frontiers.csv")

MAX_RISK = [1, 0, 0, 0, 0]

subset = res.iloc[:, [1, 10, 20, 30, 40, 49]]
pf_series = pfs["weights"].apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))

portfolio_weights = pf_series.iloc[[0, 10, 20, 30, 40]]

portfolio_weights = pd.concat([portfolio_weights, pd.Series([MAX_RISK])], ignore_index=True)

plt.figure()

plt.plot(subset.index, subset.iloc[:, 0], label="Min  Risk")
plt.plot(subset.index, subset.iloc[:, 1], label="low  Risk")
plt.plot(subset.index, subset.iloc[:, 2], label="Med  Risk")
plt.plot(subset.index, subset.iloc[:, 3], label="high Risk")
plt.plot(subset.index, subset.iloc[:, 4], label="max  Risk")

plt.legend()
plt.xlabel("Observation")
plt.ylabel("Value")
plt.title("Selected Features")
plt.show()