import os
import argparse
import yfinance as yf
import pandas as pd

# 从项目配置中读取 symbol 映射
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import COMMODITY_SYMBOLS


def download_historical_data(commodity_key="gold"):
    """
    从 yfinance 下载指定品种的全量历史数据并保存到 data/ 目录。
    黄金默认保存为 gc_f_full_history.csv（向后兼容），其他品种保存为 {key}_full_history.csv。
    """
    if commodity_key not in COMMODITY_SYMBOLS:
        print(f"未知品种 key: {commodity_key}。可用品种: {list(COMMODITY_SYMBOLS.keys())}")
        return

    info = COMMODITY_SYMBOLS[commodity_key]
    ticker = info["symbol"]
    name = info["name"]

    if commodity_key == "gold":
        filename = "gc_f_full_history.csv"
    else:
        filename = f"{commodity_key}_full_history.csv"

    print(f"开始从 yfinance 建立 {name} ({ticker}) 长期历史数据库本地缓存...")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)

    try:
        df = yf.download(ticker, period="max", interval="1d")

        if df.empty:
            print("下载失败：返回数据为空。")
            return

        # 清洗多重索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.dropna(subset=['Close'])
        df.to_csv(file_path)

        print(f"数据全部落地成功！")
        print(f"-> 品种: {name} ({ticker})")
        print(f"-> 记录总数: {len(df)} 根真实日K线")
        print(f"-> 时间跨度: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"-> 存储路径: {file_path}")

    except Exception as e:
        print(f"拉取长周期数据时发生错误：{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载期货历史数据")
    parser.add_argument("--commodity", default="gold", help="品种 key（如 gold, silver, copper）")
    args = parser.parse_args()
    download_historical_data(commodity_key=args.commodity)
