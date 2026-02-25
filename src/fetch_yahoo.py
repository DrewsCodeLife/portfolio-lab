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

FETCH_BIL   = False
FETCH_BND   = True
FETCH_EFA   = False
FETCH_RWR   = False
FETCH_SPY   = False
FETCH_VBMFX = False
FETCH_VNQ   = False
FETCH_VTI   = False
FETCH_VXUS  = False

if FETCH_BIL:
  print("Downloading BIL...")
  data = yf.download("BIL", keepna=True, interval='1d')
  data.to_csv("bil.csv")

if FETCH_BND:
  print("Waiting...")
  time.sleep(60)
  print("Downloading BND...")
  data = yf.download("BND", keepna=True, interval='1d')
  data.to_csv("bnd.csv")


if FETCH_EFA:
  print("Waiting...")
  time.sleep(60)
  print("Downloading EFA...")
  data = yf.download("EFA", keepna=True, interval='1d')
  data.to_csv("efa.csv")


if FETCH_RWR:
  print("Waiting...")
  time.sleep(60)
  print("Downloading RWR...")
  data = yf.download("RWR", keepna=True, interval='1d')
  data.to_csv("rwr.csv")


if FETCH_SPY:
  print("Waiting...")
  time.sleep(60)
  print("Downloading SPY...")
  data = yf.download("SPY", keepna=True, interval='1d')
  data.to_csv("spy.csv")

if FETCH_VBMFX:
  print("Waiting...")
  time.sleep(60)
  print("Downloading VBMFX...")
  data = yf.download("VBMFX", keepna=True, interval='1d')
  data.to_csv("vbmfx.csv")


if FETCH_VNQ:
  print("Waiting...")
  time.sleep(60)
  print("Downloading VNQ...")
  data = yf.download("VNQ", keepna=True, interval='1d')
  data.to_csv("vnq.csv")


if FETCH_VTI:
  print("Waiting...")
  time.sleep(60)
  print("Downloading VTI...")
  data = yf.download("VTI", keepna=True, interval='1d')
  data.to_csv("vti.csv")


if FETCH_VXUS:
  print("Waiting...")
  time.sleep(60)
  print("Downloading VXUS...")
  data = yf.download("VXUS", keepna=True, interval='1d')
  data.to_csv("vxus.csv")