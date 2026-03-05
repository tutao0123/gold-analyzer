import os
import sys
# ensure the project root is on the module search path
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
    """Search for major gold fundamental news for the given week."""
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
        print(f" LLM news generation failed: {e}")
        return "No macro driver found or retrieval failed."

async def process_week(sem, client, idx, total_weeks, date_str, start_str, history_slice, s, img_dir, rag, rag_lock):
    async with sem:
        img_filename = f"week_{date_str}.png"
        img_path = os.path.join(img_dir, img_filename)

        print(f"[{idx+1}/{total_weeks}] Processing: {date_str} (generating chart + fetching news)...")

        # 1. plot chart asynchronously (avoid blocking the event loop)
        def plot_img():
            try:
                mpf.plot(history_slice, type='candle', mav=(5, 20), style=s,
                         title=f"Gold Snap: {date_str}",
                         savefig=img_path, volume=False)
                return True
            except Exception as e:
                print(f"[{date_str}] Chart generation failed: {e}")
                return False

        plotting_success = await asyncio.to_thread(plot_img)
        if not plotting_success:
            return

        # 2. fetch LLM news summary asynchronously
        news_summary = await fetch_weekly_news(client, start_str, date_str)
        text_content = f"时间截面: {date_str} 周末。\n近期三个月走势如附图。本周新闻与核心驱动情绪：{news_summary}"

        # 3. fetch embedding vector asynchronously
        def fetch_vector():
            try:
                # suppress verbose logging to keep terminal clean
                import logging
                logging.getLogger("gold_rag").setLevel(logging.WARNING)

                return rag.get_embedding(text=text_content, image_path=img_path)
            except Exception as e:
                print(f"[{date_str}] Embedding failed: {e}")
                return None

        vector = await asyncio.to_thread(fetch_vector)
        if not vector:
            return

        # 4. use mutex to serialise Faiss writes
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
                    print(f"[{date_str}] Insert error: {e}")
            await asyncio.to_thread(safe_insert)

        print(f"[{date_str}] ✅ Archived. Summary: {news_summary[:40]}...")

async def main(years=5):
    print(f"=== Building gold weekly chart + sentiment multimodal knowledge base (last {years} years, async) ===")

    rag = GoldMultimodalRAG()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
        return

    client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kline_weekly_images")
    os.makedirs(img_dir, exist_ok=True)

    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gc_f_full_history.csv")
    print(f"Loading gold price data from local CSV (GC=F): {csv_file}")

    if not os.path.exists(csv_file):
        print("Error: local CSV not found. Please run dl/download_history.py first.")
        return

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    # filter to the requested number of years
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff_date]

    if df.empty:
        print("No data after date filter.")
        return

    df_weekly = df.resample('W').last()

    # concurrency limit: balance API rate and host capacity
    sem = asyncio.Semaphore(50)
    rag_lock = asyncio.Lock()

    total_weeks = len(df_weekly)
    print(f"Found {total_weeks} weekly snapshots. Dispatching embedding tasks...")

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
    print("\n=== Weekly chart + news memory base built successfully! ===")


def run():
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(years=5))

if __name__ == "__main__":
    run()
