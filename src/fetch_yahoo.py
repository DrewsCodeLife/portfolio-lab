"""
Creation Date: 2/10/2026

WARNING: If you run this script, you'll almost certainly fail to download
          all of the data due to yahoo finance rate limitations. I suggest
          a VPN or downloading with only one "FETCH_" variable set to true, and
          performing the download over a day or so.
"""


import yfinance as yf
import pandas as pd
import time

"""
Comments are to track the 'value' of acquiring a data set and if it's acquired.
- Low numbers are higher priority
- Priority makes the naive approximation of value by how few rows a dataset has.

Might need to drop VXUS or replace it with a different equivalent that has a longer history. Or not.
"""
FETCH_BIL   = False  # Acquired
FETCH_BND   = False  # Acquired
FETCH_EFA   = False  # 2
FETCH_RWR   = False  # 2
FETCH_SPY   = False  # 2
FETCH_VBMFX = False  # last
FETCH_VNQ   = False  # 2
FETCH_VTI   = False  # 2
FETCH_VXUS  = True   # 1

if FETCH_BIL:
  print("Downloading BIL...")
  data = yf.download("BIL", keepna=True, interval='1d')
  data.to_csv("bil.csv")


if FETCH_BND:
  print("Downloading BND...")
  data = yf.download("BND", keepna=True, interval='1d')
  data.to_csv("bnd.csv")


if FETCH_EFA:
  print("Downloading EFA...")
  data = yf.download("EFA", keepna=True, interval='1d')
  data.to_csv("efa.csv")


if FETCH_RWR:
  print("Downloading RWR...")
  data = yf.download("RWR", keepna=True, interval='1d')
  data.to_csv("rwr.csv")


if FETCH_SPY:
  print("Downloading SPY...")
  data = yf.download("SPY", keepna=True, interval='1d')
  data.to_csv("spy.csv")

if FETCH_VBMFX:
  print("Downloading VBMFX...")
  data = yf.download("VBMFX", keepna=True, interval='1d')
  data.to_csv("vbmfx.csv")


if FETCH_VNQ:
  print("Downloading VNQ...")
  data = yf.download("VNQ", keepna=True, interval='1d')
  data.to_csv("vnq.csv")


if FETCH_VTI:
  print("Downloading VTI...")
  data = yf.download("VTI", keepna=True, interval='1d')
  data.to_csv("vti.csv")


if FETCH_VXUS:
  print("Downloading VXUS...")
  data = yf.download("VXUS", keepna=True, interval='1d')
  data.to_csv("vxus.csv")