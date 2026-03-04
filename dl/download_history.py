import os
import yfinance as yf
import pandas as pd

def download_historical_gold_data(filename="gc_f_full_history.csv"):
    """
    一次性从 yfinance 下载过去尽可能多的黄金历史数据(最高支持获取所有可用数据，一般追溯至2000年)
    并保存到项目的 data/ 目录下
    """
    print("开始从 yfinance 建立几十年的长期黄金数据库本地缓存...")
    
    # 获取项目绝对路径的 data 目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)
    
    # 请求所有全量数据
    try:
        # yfinance 默认支持 period="max" 获取最大年限数据
        df = yf.download("GC=F", period="max", interval="1d")
        
        if df.empty:
            print("下载失败：返回数据为空。")
            return
            
        # 清洗掉多重索引，展开平铺
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # 抛却无用 NaN 的交易日，保留真正的 OCHL 数据
        df = df.dropna(subset=['Close'])
        
        # 保存到本地 CSV 文件
        df.to_csv(file_path)
        
        print(f"数据全部落地成功！")
        print(f"-> 记录总数: {len(df)} 根真实日K线")
        print(f"-> 时间跨度: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"-> 存储路径: {file_path}")
        
    except Exception as e:
        print(f"拉取长周期数据时发生错误：{e}")

if __name__ == "__main__":
    download_historical_gold_data()
