The world of financial assets is vast and complex. Naturally, we must define a scope for this project. *This file exists as a running definition for the scope, data sources, and methodology of this project.*

## Scope

| Asset Class | Ticker | Description |
|---|---|---|
| U.S. Equities | SPY | Exposure to large cap U.S. stocks through a common, well known benchmark. |
| U.S. Equities | VTI | Tracks the CRSP total market index, including medium, small, and micro cap stocks. |
| International Equities | EFA | Exposure to developed, non-U.S. markets, frequently used for portfolio analysis |
| International Equities | VXUS | Vanguard's total international stock ETF. Broader diversification than EFA and modern international equity exposure. |
| Fixed Income | BND | Seeks to track the broad U.S. bond market. Commonly used as a proxy for bond exposure. |
| Fixed Income | VBMFX | Older Vanguard mutual fund tracking U.S. bond market with long historical record. |
| Real Assets / REITs | VNQ | Tracks U.S. real estate equity investment trusts, capturing real-asset behavior while remaining tradeable. |
| Real Assets / REITs | RWR | Seeks to provide exposure to publically traded REIT securities in the U.S. |
| Cash / Cash equivalents | BIL | The bloomberg 1-3 month Treasury bond ETF. Closely approximates HYSAs and money market funds. |
| Cash / Cash equivalents | DTB3 | 3-month U.S. Treasury bill rate published by the Federal Reserve. Clean, government-sourced benchmark for cash-like returns. |

SPY, VTI, EFA, VXUS, BND, VNQ, RWR, and BIL were collected from: [Kaggle Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs?resource=download)

DTB3 was collected from [FRED](https://fred.stlouisfed.org/series/DTB3)

VBMFX was scraped from Yahoo Finance using the yfinance python package, the script for this fetch operation is available in `src/` as "fetch_vbmfx.py"
