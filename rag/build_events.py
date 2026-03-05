import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import mplfinance as mpf
import pandas as pd
from openai import OpenAI
from rag.engine import GoldMultimodalRAG

def fetch_event_description(query: str) -> str:
    """
    Use DashScope qwen-plus with web search to summarise a historical financial event's
    impact on gold prices.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    print(f"   [LLM web search] Fetching event news: {query}")
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是黄金交易历史学家。请结合联网搜索总结以下事件期间，受该事件（及其伴随的新闻情绪、非农通胀等宏观数据）影响，黄金当时出现了什么样的走势。语言请高度精炼（150字以内）。"},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            extra_body={"enable_search": True}
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"   [Warning] LLM description generation failed: {e}")
        return "Local fallback: failed to retrieve news summary."


def generate_kline_image(symbol="GC=F", start_date=None, end_date=None, output_path=None):
    """
    Download historical data from Yahoo Finance and render a candlestick PNG chart.
    """
    print(f"   [{start_date} to {end_date}] Downloading data...")
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        print("   [Error] No price data returned.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # chart style
    mc = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)

    # render candlestick chart
    mpf.plot(df, type='candle', mav=(5, 20), style=s,
             title=f"Gold (GC=F) Event",
             savefig=output_path, volume=False)

    return os.path.abspath(output_path)


def build_historical_knowledge():
    """
    Build a knowledge base of major historical gold price events
    (with LLM-powered news summaries).
    """
    print("=== Building gold historical chart multimodal knowledge base (web-search powered) ===")

    rag = GoldMultimodalRAG()
    if not rag.api_key:
        print("Missing DASHSCOPE_API_KEY; cannot run.")
        return

    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kline_images")
    os.makedirs(img_dir, exist_ok=True)

    historical_events = [
        {"id": "lehman_2008", "start": "2008-08-01", "end": "2008-11-30", "query": "2008年9月雷曼兄弟破产引爆次贷危机，当时避险情绪和美元流动性危机对黄金的抛售与反弹的作用。"},
        {"id": "euro_debt_2011", "start": "2011-07-01", "end": "2011-09-30", "query": "2011年8月欧债危机高潮期间，标普下调美国主权评级，随后黄金创下1920点历史新高的相关新闻。"},
        {"id": "fed_hike_2015", "start": "2015-11-01", "end": "2016-01-31", "query": "2015年12月美联储十年来首次加息落地，这一时间点由于'买预期卖事实'，当时黄金价格筑底成功结束漫长熊市。"},
        {"id": "brexit_2016", "start": "2016-06-01", "end": "2016-07-31", "query": "2016年6月底英国脱欧公投（黑天鹅事件）爆出冷门，黄金因为避险情绪爆发出现的暴力大阳线上涨行情。"},
        {"id": "svb_collapse_2023", "start": "2023-03-01", "end": "2023-04-30", "query": "2023年3月硅谷银行（SVB）破产危机引发市场对美国中小银行业的担忧，导致的黄金快速拉升脱离底部。"},
        {"id": "china_pboc_2024", "start": "2024-03-01", "end": "2024-05-31", "query": "2024年3-5月，在全球央行购金热潮下，即使通胀高企且美联储降息预期落空美元走强，黄金依然单边上行的脱钩行情。"},
        {"id": "trump_win_2024", "start": "2024-10-15", "end": "2024-12-15", "query": "2024年11月美国总统大选特朗普胜选，大选尘埃落定后短期避险降温，以及'特朗普交易'推高强美元和美债收益率造成黄金的大跌回调。"}
    ]

    for event in historical_events:
        print(f"\n>>>> Processing event: {event['id']}")
        img_path = os.path.join(img_dir, f"{event['id']}.png")

        # 1. generate chart image
        actual_path = generate_kline_image(
            symbol="GC=F",
            start_date=event["start"],
            end_date=event["end"],
            output_path=img_path
        )

        if not actual_path:
            continue

        # 2. call LLM with web search to expand event summary
        description = fetch_event_description(event["query"])

        # 3. insert chart + text into multimodal vector store
        print(f"   [Inserting] Combining chart image and text description into RAG store...")
        rag.add_knowledge(
            knowledge_id=event['id'],
            text=description,
            image_path=actual_path,
            metadata={"type": "historical_case", "start": event['start'], "end": event['end']}
        )

    print("\n=== Historical event knowledge base built successfully! ===")


if __name__ == "__main__":
    build_historical_knowledge()
