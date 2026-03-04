import os
import sys
# 确保模块搜寻路径包含根目录，否则 Python 直接运行此文件时会找不到 rag 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import asyncio
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from openai import AsyncOpenAI
import platform
from rag.engine import GoldMultimodalRAG

async def fetch_weekly_news(client, week_start, week_end):
    """搜索当周的黄金重大基本面新闻"""
    query = f"搜索从 {week_start} 到 {week_end} 这一周内关于黄金市场的核心宏观新闻或重大经济数据（如非农、CPI、美联储决议、地缘政治）。请用精炼的一句话概述导致当周金价波动的核心驱动力（50字以内）。没查到大事就回答无可考记录。"
    try:
        completion = await client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[
                {"role": "system", "content": "你是黄金交易研究员，你擅长通过网络回溯历史行情驱动事件。"},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            extra_body={"enable_search": True}
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
         print(f" LLM新闻生成失败: {e}")
         return "缺乏宏观驱动或检索失败。"

async def process_week(sem, client, idx, total_weeks, date_str, start_str, history_slice, s, img_dir, rag, rag_lock):
    async with sem:
        img_filename = f"week_{date_str}.png"
        img_path = os.path.join(img_dir, img_filename)
        
        print(f"[{idx+1}/{total_weeks}] 处理节点: {date_str} (生成图形并搜索网摘)...")
        
        # 1. 异步画图 (避免阻塞主事件循环)
        def plot_img():
            try:
                mpf.plot(history_slice, type='candle', mav=(5, 20), style=s, 
                         title=f"Gold Snap: {date_str}",
                         savefig=img_path, volume=False)
                return True
            except Exception as e:
                print(f"[{date_str}] 画图失败: {e}")
                return False
                
        plotting_success = await asyncio.to_thread(plot_img)
        if not plotting_success:
            return
            
        # 2. 异步获取大模型总结
        news_summary = await fetch_weekly_news(client, start_str, date_str)
        text_content = f"时间截面: {date_str} 周末。\n近期三个月走势如附图。本周新闻与核心驱动情绪：{news_summary}"
        
        # 3. 异步获取 Embedding 向量
        def fetch_vector():
            try:
                # 关闭过多的日志以保持终端清晰
                import logging
                logging.getLogger("gold_rag").setLevel(logging.WARNING) 
                
                return rag.get_embedding(text=text_content, image_path=img_path)
            except Exception as e:
                print(f"[{date_str}] 获取Embedding失败: {e}")
                return None
                
        vector = await asyncio.to_thread(fetch_vector)
        if not vector:
            return
            
        # 4. 互斥锁确保局部写入 Faiss 不崩盘
        async with rag_lock:
            def safe_insert():
                try:
                    rag.add_knowledge(
                        knowledge_id=f"weekly_{date_str}",
                        text=text_content,
                        image_path=img_path,
                        metadata={"type": "weekly_snapshot", "date": date_str},
                        precomputed_vector=vector
                    )
                except Exception as e:
                    print(f"[{date_str}] 入库出错 {e}")
            await asyncio.to_thread(safe_insert)
            
        print(f"[{date_str}] ✅ 归档成功! 摘要: {news_summary[:40]}...")

async def main(years=5):
    print(f"=== 开始构筑黄金近 {years} 年周线级别的图形与情绪多模态知识库 (🚀异步并发版) ===")
    
    rag = GoldMultimodalRAG()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("遇到错误：缺少 DASHSCOPE_API_KEY环境变量。")
        return
        
    client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kline_weekly_images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gc_f_full_history.csv")
    print(f"正在从本地知识大库加载金价数据 (GC=F): {csv_file}")
    
    if not os.path.exists(csv_file):
        print("遇到错误：未找到本地 CSV 数据。请先运行 dl/download_history.py")
        return
        
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # 根据 years 截取最近的数据，如果是全量加载可以适当修改
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff_date]
    
    if df.empty:
        print("过滤后的数据为空。")
        return
        
    df_weekly = df.resample('W').last()
    
    # 设置合理并发量，兼顾 API 并发与主机的处理能力
    sem = asyncio.Semaphore(50)
    rag_lock = asyncio.Lock()
    
    total_weeks = len(df_weekly)
    print(f"总计发现 {total_weeks} 个周度节点。开始同时派发百炼模型搜寻任务...")
    
    mc = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)
    
    tasks = []
    
    for idx, (week_end, row) in enumerate(df_weekly.iterrows()):
        week_start = week_end - datetime.timedelta(days=6)
        date_str = week_end.strftime("%Y-%m-%d")
        start_str = week_start.strftime("%Y-%m-%d")
        
        history_slice = df[df.index <= week_end].tail(60)
        
        if len(history_slice) < 10:
            continue
            
        task = asyncio.create_task(
            process_week(sem, client, idx, total_weeks, date_str, start_str, history_slice, s, img_dir, rag, rag_lock)
        )
        tasks.append(task)
        
    await asyncio.gather(*tasks)
    print("\n=== 五年周度高精度走势图片与新闻记忆库异步构建完成！===")


def run():
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(years=5))

if __name__ == "__main__":
    run()
