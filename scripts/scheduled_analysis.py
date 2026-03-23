#!/usr/bin/env python
"""
黄金期货定时分析脚本
支持多种分析模式：盘前盘点、午间复盘、盘前预测、盘后总结
"""

import os
import sys
from datetime import datetime

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.base_agent import Msg
from rag.engine import GoldMultimodalRAG
from core.config import COMMODITY_SYMBOLS


# ================== 分析模式配置 ==================

ANALYSIS_MODES = {
    "pre_market": {
        "name": "盘前盘点",
        "question": "请分析隔夜全球市场动态（美股、美债、美元、原油），结合亚洲早盘黄金走势，给出今日交易的关键观察点和潜在机会。重点关注：1) 隔夜重大消息 2) 技术面关键位置 3) 今日经济数据日历",
        "output_prefix": "pre_market"
    },
    "midday": {
        "name": "午间复盘",
        "question": "请复盘上午黄金交易情况，分析：1) 上午价格行为特征 2) 成交量变化 3) 美市开盘前策略调整建议 4) 下午需要关注的关键位置",
        "output_prefix": "midday"
    },
    "pre_us": {
        "name": "美市盘前预测",
        "question": "美市即将开盘，请给出黄金走势预测和交易计划：1) 当前技术形态分析 2) 美市可能的驱动因素 3) 最佳入场点和止损位 4) 目标价位和退出策略",
        "output_prefix": "pre_us"
    },
    "post_market": {
        "name": "盘后总结",
        "question": "请总结今日黄金全天交易：1) 价格行为回顾 2) 多空力量对比 3) 持仓建议（如有）4) 明日交易展望和关键观察点",
        "output_prefix": "post_market"
    }
}


def run_analysis(mode: str, commodity_key: str = "gold"):
    """
    执行定时分析任务
    
    Args:
        mode: 分析模式 (pre_market, midday, pre_us, post_market)
        commodity_key: 商品代码 (默认 gold)
    """
    if mode not in ANALYSIS_MODES:
        print(f"错误：未知分析模式 '{mode}'")
        print(f"可用模式：{', '.join(ANALYSIS_MODES.keys())}")
        return
    
    mode_config = ANALYSIS_MODES[mode]
    commodity = COMMODITY_SYMBOLS.get(commodity_key)
    
    if not commodity:
        print(f"错误：未知商品 '{commodity_key}'")
        return
    
    print(f"\n{'=' * 60}")
    print(f"  {mode_config['name']} - {commodity['name']}")
    print(f"  时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
    print(f"{'=' * 60}\n")
    
    # 获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：DASHSCOPE_API_KEY 环境变量未设置")
        return
    
    # 初始化 RAG
    print("=== 加载 RAG 知识库 ===")
    rag = GoldMultimodalRAG()
    
    # 导入 Agent
    from agents import (
        RAGAnalystAgent, MacroAnalystAgent, QuantEngineerAgent,
        DLPredictorAgent, PortfolioManagerAgent,
        CrossMarketAgent, RiskControlAgent, SentimentAgent
    )
    
    cname = commodity["name"]
    _kw = dict(model_name="qwen-plus", api_key=api_key, commodity=commodity)
    
    # 创建 Agent 实例
    analyst_agent = RAGAnalystAgent(
        name="技术面研究员",
        sys_prompt=f"你是专门分析{cname}行情的智能体。使用提供给你的【历史 K 线记忆库匹配】，得出技术分析结论。",
        rag_db=rag,
        **_kw
    )
    macro_agent = MacroAnalystAgent(
        name="宏观基本面分析师",
        sys_prompt=f"你是{cname}交易团队的宏观分析师。基于提供给你的实时宏观数据，并利用联网能力补充最新情报。",
        enable_search=True,
        **_kw
    )
    quant_agent = QuantEngineerAgent(
        name="量化工程师",
        sys_prompt=f"你是{cname}交易团队的底层数据工程师。只负责客观陈述数据，绝不带主观情绪。",
        **_kw
    )
    # dl_agent 已禁用
    cross_market_agent = CrossMarketAgent(
        name="跨市场联动分析师",
        sys_prompt=f"你是跨市场联动专家。分析美元指数、美债收益率、VIX、原油等与{cname}的联动关系。",
        **_kw
    )
    sentiment_agent = SentimentAgent(
        name="情绪分析师",
        sys_prompt=f"你是市场情绪面分析专家。通过联网搜索分析{cname}的讨论热度和情绪倾向。",
        enable_search=True,
        **_kw
    )
    pm_agent = PortfolioManagerAgent(
        name="首席策略官",
        sys_prompt=f"你是{cname}基金的主理人。综合各专家意见，做出最终交易决策。",
        **_kw
    )
    risk_agent = RiskControlAgent(
        name="风控合规官",
        sys_prompt=f"你是独立的风控合规审计官。对首席策略官的决策进行风险审查。",
        **_kw
    )
    
    # 用户问题
    question = mode_config["question"]
    print(f"分析问题：{question}\n")
    
    user_issue = Msg(name="Manager", content=question, role="user")
    
    # 第一轮：专家并行报告
    print("\n=== 第一轮：各领域专家独立报告 ===\n")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    agents_round1 = [
        analyst_agent,
        macro_agent,
        quant_agent,
        cross_market_agent,
        sentiment_agent,
    ]
    
    round1_replies = [None] * len(agents_round1)
    
    with ThreadPoolExecutor(max_workers=len(agents_round1)) as executor:
        future_to_idx = {
            executor.submit(agent, user_issue): i
            for i, agent in enumerate(agents_round1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            agent = agents_round1[idx]
            try:
                reply = future.result()
                round1_replies[idx] = reply
                print(f"\n[{agent.name} 报告]:\n{reply.content}\n")
            except Exception as e:
                fallback = Msg(name=agent.name, role="assistant",
                               content=f"【{agent.name}分析失败】{e}")
                round1_replies[idx] = fallback
                print(f"\n[{agent.name}] 报告失败：{e}\n")
    
    # 第二轮：首席策略官综合研判
    print("\n=== 第二轮：首席策略官综合研判 ===\n")
    pm_reply = pm_agent(round1_replies)
    print(f"\n[首席策略官 最终决议]:\n{pm_reply.content}\n")
    
    # 第三轮：风控合规官独立审查
    print("\n=== 第三轮：风控合规官独立审查 ===\n")
    risk_reply = risk_agent(pm_reply)
    print(f"\n[风控合规官 审查结论]:\n{risk_reply.content}\n")
    
    # 保存报告
    save_report(commodity, mode_config, question, round1_replies, pm_reply, risk_reply)
    
    print(f"\n{'=' * 60}")
    print(f"  分析完成！")
    print(f"{'=' * 60}\n")


def save_report(commodity: dict, mode_config: dict, question: str, 
                round1_replies: list, pm_reply, risk_reply):
    """保存分析报告"""
    from core.config import REPORT_DIR
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        REPORT_DIR, 
        f"{mode_config['output_prefix']}_{timestamp}.txt"
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{commodity['name']} - {mode_config['name']}\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"分析问题：{question}\n\n")
        f.write(f"{'─' * 60}\n")
        f.write("各领域专家报告\n")
        f.write(f"{'─' * 60}\n\n")
        for reply in round1_replies:
            f.write(f"【{reply.name}】\n{reply.content}\n\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"【首席策略官 最终决议】\n{pm_reply.content}\n\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"【风控合规官 审查结论】\n{risk_reply.content}\n")
    
    print(f"\n报告已保存至：{filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="黄金期货定时分析脚本")
    parser.add_argument(
        "mode",
        choices=["pre_market", "midday", "pre_us", "post_market"],
        help="分析模式"
    )
    parser.add_argument(
        "--commodity",
        default="gold",
        help="商品代码 (默认：gold)"
    )
    
    args = parser.parse_args()
    run_analysis(args.mode, args.commodity)
