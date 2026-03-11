import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# add project root to module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import Msg
from rag.engine import GoldMultimodalRAG
from core.config import REPORT_DIR, COMMODITY_SYMBOLS, ANALYZABLE_COMMODITIES


def _select_commodity() -> dict:
    """Display the commodity selection menu and return the chosen commodity dict (with 'key' field)."""
    print("\n可分析品种：")
    options = []
    for key in ANALYZABLE_COMMODITIES:
        if key in COMMODITY_SYMBOLS:
            options.append(key)

    for i, key in enumerate(options, 1):
        info = COMMODITY_SYMBOLS[key]
        print(f"  {i:2d}. {info['name']} ({info['symbol']})")

    try:
        raw = input(f"\n请选择品种（默认1，直接回车选黄金）: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""

    idx = 0
    if raw:
        try:
            idx = int(raw) - 1
            if idx < 0 or idx >= len(options):
                print(f"输入超出范围，使用默认品种（黄金）")
                idx = 0
        except ValueError:
            print("无效输入，使用默认品种（黄金）")
            idx = 0

    selected_key = options[idx]
    commodity = dict(COMMODITY_SYMBOLS[selected_key])
    commodity["key"] = selected_key
    print(f"\n已选择: {commodity['name']} ({commodity['symbol']})\n")
    return commodity


def _save_report(commodity: dict, user_question: str, round1_replies: list, pm_reply, risk_reply) -> str:
    """Save the analysis results to the reports/ directory and return the file path."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORT_DIR, f"analysis_{timestamp}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{commodity['name']}分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"分析问题: {user_question}\n\n")
        f.write(f"{'─' * 60}\n")
        f.write("各领域专家报告\n")
        f.write(f"{'─' * 60}\n\n")
        for reply in round1_replies:
            f.write(f"【{reply.name}】\n{reply.content}\n\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"【首席策略官 最终决议】\n{pm_reply.content}\n\n")
        f.write(f"{'─' * 60}\n")
        f.write(f"【风控合规官 审查结论】\n{risk_reply.content}\n")

    return filepath


def main():
    print("=== Initialising multi-agent futures analysis framework ===")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
        return

    # commodity selection
    commodity = _select_commodity()
    cname = commodity["name"]

    # Model configuration: critical decision-making agents use stronger model
    _model_standard = "qwen-plus"
    _model_advanced = "qwen3-max"
    _kw_standard = dict(model_name=_model_standard, api_key=api_key, commodity=commodity)
    _kw_advanced = dict(model_name=_model_advanced, api_key=api_key, commodity=commodity)

    print("=== Loading local multimodal RAG store ===")
    rag = GoldMultimodalRAG()

    # ---------------- import domain-specific agents ----------------
    from agents import (
        RAGAnalystAgent, MacroAnalystAgent, QuantEngineerAgent,
        DLPredictorAgent, PortfolioManagerAgent,
        CrossMarketAgent, RiskControlAgent, SentimentAgent
    )

    analyst_agent = RAGAnalystAgent(
        name="技术面研究员",
        sys_prompt=f"""你是{cname}交易团队的技术面研究员，专注于价格形态与趋势分析。

【分析框架】
1. K线形态：头肩顶/底、双顶/底、三角形、旗形等经典形态识别
2. 均线系统：MA5/MA10/MA20 多空排列与金叉死叉信号
3. 支撑阻力：关键价位、前高前低、斐波那契回撤位
4. 量价配合：放量突破、缩量回调、量价背离

【工作原则】
- 基于RAG历史K线记忆库匹配相似形态
- 区分历史价格与当前价格，不可直接搬用历史价位
- 给出明确的支撑位、阻力位、止损位、目标位
- 标注形态确认条件与失效条件

【输出要求】
100字内，结论简洁明确，避免模棱两可。""",
        rag_db=rag,
        **_kw_standard
    )
    macro_agent = MacroAnalystAgent(
        name="宏观基本面分析师",
        sys_prompt=f"你是{cname}交易团队的宏观分析师。基于提供给你的实时宏观数据，并利用联网能力补充最新情报，判断宏观环境对{cname}的影响方向。",
        enable_search=True,
        **_kw_standard
    )
    quant_agent = QuantEngineerAgent(
        name="量化工程师",
        sys_prompt=f"你是{cname}交易团队的底层数据工程师。只负责客观陈述数据，绝不带主观情绪。",
        **_kw_standard
    )
    dl_agent = DLPredictorAgent(
        name="算法预测师",
        sys_prompt=f"你是基于神经网络的时间序列预测AI。负责将底层数理模型（LSTM + Transformer 双模型）对{cname}的预测目标点位和回测绩效反馈给策略官。",
        **_kw_standard
    )
    cross_market_agent = CrossMarketAgent(
        name="跨市场联动分析师",
        sys_prompt=f"你是跨市场联动专家。你负责追踪美元指数(DXY)、美国10年期国债收益率、VIX恐慌指数、原油等与{cname}高度相关的品种，分析它们的联动关系对价格的影响。",
        **_kw_standard
    )
    sentiment_agent = SentimentAgent(
        name="情绪分析师",
        sys_prompt=f"你是市场情绪面分析专家。你通过联网搜索社交媒体和财经论坛上关于{cname}的讨论热度和情绪倾向，为团队提供散户情绪温度计和反向指标参考。",
        enable_search=True,
        **_kw_standard
    )
    pm_agent = PortfolioManagerAgent(
        name="首席策略官",
        sys_prompt=f"你是{cname}基金的主理人。具备大局观、风控意识，不盲从任何单一指标。你需要综合技术面、宏观面、量化指标、跨市场联动和情绪面的全部信息，做出最终决策。（注：深度学习预测模型已暂停使用）",
        **_kw_advanced
    )
    risk_agent = RiskControlAgent(
        name="风控合规官",
        sys_prompt=f"你是独立的风控合规审计官。你不参与前期讨论，唯一任务是对首席策略官关于{cname}的最终决策进行严格的风险审查，确保策略不会让基金暴露在过度风险中。",
        **_kw_advanced
    )

    # ================= receive user question =================
    default_question = f"关注到今天行情异动，看看最近{cname}走势，有没有类似头肩底？帮我全面诊脉。"
    try:
        user_input = input(f"\n请输入分析问题（直接回车使用默认）:\n> ").strip()
        question = user_input if user_input else default_question
    except (EOFError, KeyboardInterrupt):
        question = default_question
    print(f"\n分析问题: {question}\n")

    user_issue = Msg(name="Manager", content=question, role="user")

    # ================= round 1: 6 specialists report in parallel =================
    print("\n=== 圆桌会议开始 (8位专家) ===")
    print("\n--- 第一轮：各领域专家独立报告（并行执行）---")
    # Note: DLPredictorAgent temporarily disabled due to model ineffectiveness in bull market

    agents_round1 = [
        analyst_agent,
        macro_agent,
        quant_agent,
        # dl_agent,  # ⚠️ 暂时禁用：LSTM 模型在牛市环境下严重失效（回测 -48% vs 买入持有 +80%）
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
                print(f"\n[{agent.name} 报告]:\n{reply.content}")
            except Exception as e:
                fallback = Msg(name=agent.name, role="assistant",
                               content=f"【{agent.name}分析失败】{e}")
                round1_replies[idx] = fallback
                print(f"\n[{agent.name}] 报告失败: {e}")

    # ================= round 2: chief strategist synthesises =================
    print("\n--- 第二轮：首席策略官综合研判 ---")
    pm_reply = pm_agent(round1_replies)
    print(f"\n{'=' * 50}\n[首席策略官 最终决议]:\n{pm_reply.content}\n{'=' * 50}")

    # ================= round 3: risk officer independent review =================
    print("\n--- 第三轮：风控合规官独立审查 ---")
    risk_reply = risk_agent(pm_reply)
    print(f"\n{'=' * 50}\n[风控合规官 审查结论]:\n{risk_reply.content}\n{'=' * 50}")

    # ================= save report =================
    try:
        saved_path = _save_report(commodity, question, round1_replies, pm_reply, risk_reply)
        print(f"\n报告已保存至: {saved_path}")
    except Exception as e:
        print(f"\n报告保存失败: {e}")


if __name__ == "__main__":
    main()
