"""
Generic commodity data downloader.
Supports downloading historical data for commodity futures, stock indices, forex, and more.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd

from core.config import DATA_DIR, COMMODITY_SYMBOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommodityDataDownloader:
    """Generic commodity data downloader."""

    def __init__(self, data_dir: str = None):
        """
        Initialise the downloader.

        Args:
            data_dir: directory for storing data; defaults to DATA_DIR from config
        """
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols = COMMODITY_SYMBOLS

    def get_available_commodities(self) -> Dict[str, dict]:
        """Return a copy of all available commodities."""
        return self.symbols.copy()

    def list_commodities(self) -> None:
        """Print all available commodities grouped by category."""
        print("\n=== Available commodities ===\n")
        categories = {
            "Precious Metals": ["gold", "silver", "platinum", "palladium"],
            "Crude Oil & Energy": ["wti_oil", "brent_oil", "natural_gas"],
            "Agricultural": ["corn", "wheat", "soybean"],
            "Equity Indices": ["sp500", "nasdaq", "vix"],
            "Forex": ["dxy"],
            "Bond Yields": ["tnx"],
        }

        for category, items in categories.items():
            print(f"[{category}]")
            for key in items:
                if key in self.symbols:
                    info = self.symbols[key]
                    print(f"  {key:15} - {info['name']:15} ({info['symbol']:10}) [{info['unit']}]")
            print()

    def download_single(
        self,
        commodity_key: str,
        period: str = "max",
        interval: str = "1d",
        start: str = None,
        end: str = None,
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single commodity.

        Args:
            commodity_key: commodity key (e.g. 'gold', 'wti_oil')
            period:        time period ('1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max')
            interval:      bar interval ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo')
            start:         start date (YYYY-MM-DD)
            end:           end date   (YYYY-MM-DD)

        Returns:
            DataFrame or None
        """
        if commodity_key not in self.symbols:
            logger.error(f"Unknown commodity key: {commodity_key}")
            logger.info(f"Available keys: {list(self.symbols.keys())}")
            return None

        symbol_info = self.symbols[commodity_key]
        symbol = symbol_info["symbol"]
        name = symbol_info["name"]

        logger.info(f"Downloading {name} ({symbol})...")

        try:
            if start and end:
                df = yf.download(symbol, start=start, end=end, interval=interval)
            else:
                df = yf.download(symbol, period=period, interval=interval)

            if df.empty:
                logger.warning(f"{name} returned empty data")
                return None

            # flatten multi-level column index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # drop rows with missing close
            df = df.dropna(subset=["Close"])

            logger.info(f"Downloaded {len(df)} records: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return None

    def download_and_save(
        self,
        commodity_key: str,
        period: str = "max",
        interval: str = "1d",
        filename: str = None,
    ) -> Optional[str]:
        """
        Download data and save to a CSV file.

        Args:
            commodity_key: commodity key
            period:        time period
            interval:      bar interval
            filename:      custom filename; auto-generated if not provided

        Returns:
            saved file path, or None on failure
        """
        df = self.download_single(commodity_key, period, interval)
        if df is None:
            return None

        symbol_info = self.symbols[commodity_key]
        if filename is None:
            # derive filename from symbol (e.g. GC=F → gc_f.csv)
            safe_symbol = symbol_info["symbol"].replace("^", "").replace("=", "_").replace("-", "_").lower()
            filename = f"{safe_symbol}.csv"

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)

        logger.info(f"Data saved: {filepath}")

        return filepath

    def download_multiple(
        self,
        commodity_keys: List[str],
        period: str = "max",
        interval: str = "1d",
        save_to_file: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple commodities.

        Args:
            commodity_keys: list of commodity keys
            period:         time period
            interval:       bar interval
            save_to_file:   whether to save each download to a CSV

        Returns:
            dict {commodity_key: DataFrame}
        """
        results = {}

        for key in commodity_keys:
            if key not in self.symbols:
                logger.warning(f"Skipping unknown commodity: {key}")
                continue

            if save_to_file:
                filepath = self.download_and_save(key, period, interval)
                if filepath:
                    results[key] = pd.read_csv(filepath, index_col=0, parse_dates=True)
            else:
                df = self.download_single(key, period, interval)
                if df is not None:
                    results[key] = df

        return results

    def download_all(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, str]:
        """
        Download data for all commodities.

        Args:
            period:   time period
            interval: bar interval

        Returns:
            dict {commodity_key: filepath}
        """
        results = {}
        total = len(self.symbols)

        for idx, key in enumerate(self.symbols.keys(), 1):
            logger.info(f"\n[{idx}/{total}] Processing: {key}")
            filepath = self.download_and_save(key, period, interval)
            if filepath:
                results[key] = filepath

        logger.info(f"\n=== Download complete ===")
        logger.info(f"Success: {len(results)}/{total}")

        return results

    def download_energy(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Download energy commodities (crude oil, natural gas)."""
        energy_keys = ["wti_oil", "brent_oil", "natural_gas"]
        return self.download_multiple(energy_keys, period, interval)

    def download_metals(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Download precious metals (gold, silver, platinum, palladium)."""
        metal_keys = ["gold", "silver", "platinum", "palladium"]
        return self.download_multiple(metal_keys, period, interval)

    def download_indices(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Download equity index data."""
        index_keys = ["sp500", "nasdaq", "vix"]
        return self.download_multiple(index_keys, period, interval)

    def get_correlation(
        self,
        commodity_keys: List[str],
        period: str = "1y",
        method: str = "pearson",
    ) -> Optional[pd.DataFrame]:
        """
        Compute price correlation between multiple commodities.

        Args:
            commodity_keys: list of commodity keys
            period:         time period
            method:         correlation method ('pearson', 'kendall', 'spearman')

        Returns:
            correlation matrix DataFrame
        """
        # download data
        data = self.download_multiple(commodity_keys, period, save_to_file=False)
        if not data:
            return None

        # extract close prices
        close_prices = pd.DataFrame()
        for key, df in data.items():
            close_prices[key] = df["Close"]

        # compute returns
        returns = close_prices.pct_change().dropna()

        # compute correlation matrix
        correlation = returns.corr(method=method)

        return correlation

    def get_gold_oil_ratio(
        self,
        period: str = "max",
    ) -> Optional[pd.DataFrame]:
        """
        Compute gold-to-oil price ratio.

        Returns:
            DataFrame with date, gold price, oil price, and ratio
        """
        # download gold and oil data
        gold_df = self.download_single("gold", period)
        oil_df = self.download_single("wti_oil", period)

        if gold_df is None or oil_df is None:
            return None

        # merge
        merged = pd.DataFrame()
        merged["gold"] = gold_df["Close"]
        merged["oil"] = oil_df["Close"]
        merged = merged.dropna()

        # compute ratio
        merged["gold_oil_ratio"] = merged["gold"] / merged["oil"]

        return merged


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generic commodity data downloader")
    parser.add_argument("--list", action="store_true", help="list all available commodities")
    parser.add_argument("--download", type=str, nargs="+", help="download specified commodities (e.g. gold wti_oil)")
    parser.add_argument("--all", action="store_true", help="download all commodities")
    parser.add_argument("--metals", action="store_true", help="download precious metals")
    parser.add_argument("--energy", action="store_true", help="download energy commodities (crude oil, natural gas)")
    parser.add_argument("--indices", action="store_true", help="download equity indices")
    parser.add_argument("--correlation", type=str, nargs="+", help="compute correlation between commodities (e.g. gold wti_oil dxy)")
    parser.add_argument("--gold-oil-ratio", action="store_true", help="compute gold/oil price ratio")
    parser.add_argument("--period", type=str, default="max", help="time period (default: max)")

    args = parser.parse_args()
    downloader = CommodityDataDownloader()

    if args.list:
        downloader.list_commodities()
    elif args.download:
        downloader.download_multiple(args.download, period=args.period)
    elif args.all:
        downloader.download_all(period=args.period)
    elif args.metals:
        downloader.download_metals(period=args.period)
    elif args.energy:
        downloader.download_energy(period=args.period)
    elif args.indices:
        downloader.download_indices(period=args.period)
    elif args.correlation:
        corr = downloader.get_correlation(args.correlation, period=args.period)
        if corr is not None:
            print("\n=== Price Correlation Matrix ===\n")
            print(corr.round(3))
    elif args.gold_oil_ratio:
        ratio = downloader.get_gold_oil_ratio(period=args.period)
        if ratio is not None:
            print("\n=== Gold/Oil Ratio ===\n")
            print(f"Latest date: {ratio.index[-1].strftime('%Y-%m-%d')}")
            print(f"Gold:  ${ratio['gold'].iloc[-1]:.2f}")
            print(f"Oil:   ${ratio['oil'].iloc[-1]:.2f}")
            print(f"Ratio: {ratio['gold_oil_ratio'].iloc[-1]:.2f}")
    else:
        # default: show help
        downloader.list_commodities()
        print("\nUsage examples:")
        print("  python dl/commodity_downloader.py --list                    # list all commodities")
        print("  python dl/commodity_downloader.py --download gold wti_oil  # download gold and WTI crude")
        print("  python dl/commodity_downloader.py --energy                 # download all energy data")
        print("  python dl/commodity_downloader.py --all                    # download everything")
        print("  python dl/commodity_downloader.py --correlation gold wti_oil dxy  # compute correlation")
        print("  python dl/commodity_downloader.py --gold-oil-ratio         # gold/oil ratio")


if __name__ == "__main__":
    main()
