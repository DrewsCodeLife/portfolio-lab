from avkey import key

import pandas as pd
import requests
import csv
import os



DATA_FOLDER   = 'data/alpha vantage/'
BASE          = 'https://www.alphavantage.co/query?function='
TYPE          = 'TIME_SERIES_DAILY_ADJUSTED'
SYMBOL_PREFIX = '&symbol='
FULL          = '&outputsize=full'
DTYPE         = '&datatype=csv'

RUN_ALL = True

symbol_list = ['VXUS', 'EFA', 'RWR', 'SPY', 'VNQ', 'VTI', 'BIL', 'BND', 'VBMFX']
if RUN_ALL:
  for SYMBOL in symbol_list:
    PREFIX = BASE + TYPE + SYMBOL_PREFIX + SYMBOL
    SUFFIX = FULL + DTYPE + key

    r = requests.get(PREFIX + SUFFIX)
    data = r.text
    with open(os.path.join(DATA_FOLDER, SYMBOL + '.csv'), "w", newline="") as f:
      f.write(data)
# else:
#   r = requests.get(BASE + TYPE + )