import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import requests


'''
time_frame= 5 minutes
duration=6 days
'''

df = yf.Ticker('MSFT').history(period="6d", interval="5m")
df.to_csv('msft.csv')
print(df.tail())