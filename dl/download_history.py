import os
import argparse
import yfinance as yf
import pandas as pd

# load symbol mapping from project config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import COMMODITY_SYMBOLS


def download_historical_data(commodity_key="gold"):
    """
    Download full historical data for the given commodity from yfinance and save to data/.
    Gold is saved as gc_f_full_history.csv (backward-compatible); others as {key}_full_history.csv.
    """
    if commodity_key not in COMMODITY_SYMBOLS:
        print(f"Unknown commodity key: {commodity_key}. Available: {list(COMMODITY_SYMBOLS.keys())}")
        return

    info = COMMODITY_SYMBOLS[commodity_key]
    ticker = info["symbol"]
    name = info["name"]

    filename = "gc_f_full_history.csv" if commodity_key == "gold" else f"{commodity_key}_full_history.csv"

    print(f"Downloading full history for {name} ({ticker}) from yfinance...")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)

    try:
        df = yf.download(ticker, period="max", interval="1d")

        if df.empty:
            print("Download failed: empty response.")
            return

        # flatten multi-level column index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.dropna(subset=['Close'])
        df.to_csv(file_path)

        print(f"Download complete.")
        print(f"-> Commodity: {name} ({ticker})")
        print(f"-> Records:   {len(df)} daily bars")
        print(f"-> Span:      {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"-> Saved to:  {file_path}")

    except Exception as e:
        print(f"Error fetching long-period data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download commodity futures historical data")
    parser.add_argument("--commodity", default="gold", help="commodity key (e.g. gold, silver, copper)")
    args = parser.parse_args()
    download_historical_data(commodity_key=args.commodity)
