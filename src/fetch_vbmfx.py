import yfinance as yf
import pandas as pd

# tckr = yf.Ticker("VBMFX")
data = yf.download("VBMFX", keepna=True, interval='1d')

data.to_csv("vbmfx.csv")