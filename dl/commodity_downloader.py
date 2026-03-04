"""
通用商品数据下载器
支持多种商品期货、股票指数、外汇等数据的下载
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
    """通用商品数据下载器"""

    def __init__(self, data_dir: str = None):
        """
        初始化下载器

        Args:
            data_dir: 数据存储目录，默认使用配置中的 DATA_DIR
        """
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols = COMMODITY_SYMBOLS

    def get_available_commodities(self) -> Dict[str, dict]:
        """获取所有可用的商品列表"""
        return self.symbols.copy()

    def list_commodities(self) -> None:
        """打印所有可用的商品"""
        print("\n=== 可下载的商品列表 ===\n")
        categories = {
            "贵金属": ["gold", "silver", "platinum", "palladium"],
            "原油能源": ["wti_oil", "brent_oil", "natural_gas"],
            "农产品": ["corn", "wheat", "soybean"],
            "股票指数": ["sp500", "nasdaq", "vix"],
            "外汇": ["dxy"],
            "债券收益率": ["tnx"],
        }

        for category, items in categories.items():
            print(f"【{category}】")
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
        下载单个商品的历史数据

        Args:
            commodity_key: 商品键名 (如 'gold', 'wti_oil')
            period: 时间周期 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: 时间间隔 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame 或 None
        """
        if commodity_key not in self.symbols:
            logger.error(f"未知的商品键名: {commodity_key}")
            logger.info(f"可用键名: {list(self.symbols.keys())}")
            return None

        symbol_info = self.symbols[commodity_key]
        symbol = symbol_info["symbol"]
        name = symbol_info["name"]

        logger.info(f"正在下载 {name} ({symbol}) 数据...")

        try:
            if start and end:
                df = yf.download(symbol, start=start, end=end, interval=interval)
            else:
                df = yf.download(symbol, period=period, interval=interval)

            if df.empty:
                logger.warning(f"{name} 数据为空")
                return None

            # 处理多重索引
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # 清理数据
            df = df.dropna(subset=["Close"])

            logger.info(f"下载成功: {len(df)} 条记录, 时间范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            logger.error(f"下载 {name} 数据失败: {e}")
            return None

    def download_and_save(
        self,
        commodity_key: str,
        period: str = "max",
        interval: str = "1d",
        filename: str = None,
    ) -> Optional[str]:
        """
        下载数据并保存到CSV文件

        Args:
            commodity_key: 商品键名
            period: 时间周期
            interval: 时间间隔
            filename: 自定义文件名，默认自动生成

        Returns:
            保存的文件路径，失败返回 None
        """
        df = self.download_single(commodity_key, period, interval)
        if df is None:
            return None

        symbol_info = self.symbols[commodity_key]
        if filename is None:
            # 使用符号作为文件名 (如 GC=F -> gc_f.csv)
            safe_symbol = symbol_info["symbol"].replace("^", "").replace("=", "_").replace("-", "_").lower()
            filename = f"{safe_symbol}.csv"

        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)

        logger.info(f"数据已保存: {filepath}")

        return filepath

    def download_multiple(
        self,
        commodity_keys: List[str],
        period: str = "max",
        interval: str = "1d",
        save_to_file: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下载多个商品数据

        Args:
            commodity_keys: 商品键名列表
            period: 时间周期
            interval: 时间间隔
            save_to_file: 是否保存到文件

        Returns:
            字典 {commodity_key: DataFrame}
        """
        results = {}

        for key in commodity_keys:
            if key not in self.symbols:
                logger.warning(f"跳过未知商品: {key}")
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
        下载所有商品数据

        Args:
            period: 时间周期
            interval: 时间间隔

        Returns:
            字典 {commodity_key: filepath}
        """
        results = {}
        total = len(self.symbols)

        for idx, key in enumerate(self.symbols.keys(), 1):
            logger.info(f"\n[{idx}/{total}] 正在处理: {key}")
            filepath = self.download_and_save(key, period, interval)
            if filepath:
                results[key] = filepath

        logger.info(f"\n=== 下载完成 ===")
        logger.info(f"成功: {len(results)}/{total}")

        return results

    def download_energy(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """下载能源类商品数据 (原油、天然气)"""
        energy_keys = ["wti_oil", "brent_oil", "natural_gas"]
        return self.download_multiple(energy_keys, period, interval)

    def download_metals(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """下载贵金属数据 (黄金、白银、铂金、钯金)"""
        metal_keys = ["gold", "silver", "platinum", "palladium"]
        return self.download_multiple(metal_keys, period, interval)

    def download_indices(
        self,
        period: str = "max",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """下载股票指数数据"""
        index_keys = ["sp500", "nasdaq", "vix"]
        return self.download_multiple(index_keys, period, interval)

    def get_correlation(
        self,
        commodity_keys: List[str],
        period: str = "1y",
        method: str = "pearson",
    ) -> Optional[pd.DataFrame]:
        """
        计算多个商品之间的价格相关性

        Args:
            commodity_keys: 商品键名列表
            period: 时间周期
            method: 相关性计算方法 ('pearson', 'kendall', 'spearman')

        Returns:
            相关性矩阵 DataFrame
        """
        # 下载数据
        data = self.download_multiple(commodity_keys, period, save_to_file=False)
        if not data:
            return None

        # 提取收盘价
        close_prices = pd.DataFrame()
        for key, df in data.items():
            close_prices[key] = df["Close"]

        # 计算收益率
        returns = close_prices.pct_change().dropna()

        # 计算相关性矩阵
        correlation = returns.corr(method=method)

        return correlation

    def get_gold_oil_ratio(
        self,
        period: str = "max",
    ) -> Optional[pd.DataFrame]:
        """
        计算黄金/原油比价

        Returns:
            DataFrame 包含日期、金价、油价、比价
        """
        # 下载黄金和原油数据
        gold_df = self.download_single("gold", period)
        oil_df = self.download_single("wti_oil", period)

        if gold_df is None or oil_df is None:
            return None

        # 合并数据
        merged = pd.DataFrame()
        merged["gold"] = gold_df["Close"]
        merged["oil"] = oil_df["Close"]
        merged = merged.dropna()

        # 计算比价
        merged["gold_oil_ratio"] = merged["gold"] / merged["oil"]

        return merged


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="通用商品数据下载器")
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的商品",
    )
    parser.add_argument(
        "--download",
        type=str,
        nargs="+",
        help="下载指定商品数据 (如: gold wti_oil brent_oil)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有商品数据",
    )
    parser.add_argument(
        "--metals",
        action="store_true",
        help="下载贵金属数据",
    )
    parser.add_argument(
        "--energy",
        action="store_true",
        help="下载能源数据 (原油、天然气)",
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="下载股票指数数据",
    )
    parser.add_argument(
        "--correlation",
        type=str,
        nargs="+",
        help="计算指定商品之间的相关性 (如: gold wti_oil dxy)",
    )
    parser.add_argument(
        "--gold-oil-ratio",
        action="store_true",
        help="计算黄金/原油比价",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="时间周期 (默认: max)",
    )

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
            print("\n=== 价格相关性矩阵 ===\n")
            print(corr.round(3))
    elif args.gold_oil_ratio:
        ratio = downloader.get_gold_oil_ratio(period=args.period)
        if ratio is not None:
            print("\n=== 黄金/原油比价 ===\n")
            print(f"最新数据: {ratio.index[-1].strftime('%Y-%m-%d')}")
            print(f"金价: ${ratio['gold'].iloc[-1]:.2f}")
            print(f"油价: ${ratio['oil'].iloc[-1]:.2f}")
            print(f"比价: {ratio['gold_oil_ratio'].iloc[-1]:.2f}")
    else:
        # 默认显示帮助
        downloader.list_commodities()
        print("\n使用示例:")
        print("  python dl/commodity_downloader.py --list                    # 列出所有商品")
        print("  python dl/commodity_downloader.py --download gold wti_oil  # 下载黄金和WTI原油")
        print("  python dl/commodity_downloader.py --energy                 # 下载所有能源数据")
        print("  python dl/commodity_downloader.py --all                    # 下载所有商品")
        print("  python dl/commodity_downloader.py --correlation gold wti_oil dxy  # 计算相关性")
        print("  python dl/commodity_downloader.py --gold-oil-ratio         # 黄金/原油比价")


if __name__ == "__main__":
    main()