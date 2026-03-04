import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.data_fetcher import GoldDataFetcher
import yfinance as yf

print("--- GoldDataFetcher (Web Scrapers/APIs) ---")
fetcher = GoldDataFetcher()
price = fetcher.fetch_current_price()
print(f"Current price from fetcher: {price}")

print("\n--- yfinance GC=F ---")
df = yf.download("GC=F", period="5d")
print("yfinance recent close prices:")
print(df['Close'].tail())
