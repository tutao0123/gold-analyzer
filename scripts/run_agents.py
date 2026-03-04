import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置工作目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import Msg
from rag.engine import GoldMultimodalRAG
from core.config import REPORT_DIR


def _save_report(user_question: str, round1_replies: list, pm_reply, risk_reply) -> str:
    """将本次分析结果保存到 reports/ 目录，返回文件路径。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORT_DIR, f"analysis_{timestamp}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("黄金分析报告\n")
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
    print("=== 初始化 AgentScope 与黄金分析智能体框架 (多角色版) ===")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    _model = "qwen-plus"
    _kw = dict(model_name=_model, api_key=api_key)

    print("=== 正在加载本地多模态图文库 ===")
    rag = GoldMultimodalRAG()

    # ---------------- 导入各领域专属智能体 ----------------
    from agents import (
        RAGAnalystAgent, MacroAnalystAgent, QuantEngineerAgent,
        DLPredictorAgent, PortfolioManagerAgent,
        CrossMarketAgent, RiskControlAgent, SentimentAgent
    )

    analyst_agent = RAGAnalystAgent(
        name="技术面研究员",
        sys_prompt="你是专门分析黄金行情的智能体。使用提供给你的【历史K线记忆库匹配】，得出技术分析结论。",
        rag_db=rag,
        **_kw
    )
    macro_agent = MacroAnalystAgent(
        name="宏观基本面分析师",
        sys_prompt="你是黄金交易团队的宏观分析师。基于提供给你的实时宏观数据，并利用联网能力补充最新情报，判断宏观环境对黄金的影响方向。",
        enable_search=True,
        **_kw
    )
    quant_agent = QuantEngineerAgent(
        name="量化工程师",
        sys_prompt="你是黄金交易团队的底层数据工程师。只负责客观陈述数据，绝不带主观情绪。",
        **_kw
    )
    dl_agent = DLPredictorAgent(
        name="算法预测师",
        sys_prompt="你是基于神经网络的时间序列预测AI。负责将底层数理模型（LSTM + Transformer 双模型）预测的目标点位和回测绩效反馈给策略官。",
        **_kw
    )
    cross_market_agent = CrossMarketAgent(
        name="跨市场联动分析师",
        sys_prompt="你是跨市场联动专家。你负责追踪美元指数(DXY)、美国10年期国债收益率、VIX恐慌指数、原油等与黄金高度相关的品种，分析它们的联动关系对金价的影响。",
        **_kw
    )
    sentiment_agent = SentimentAgent(
        name="情绪分析师",
        sys_prompt="你是市场情绪面分析专家。你通过联网搜索社交媒体和财经论坛上关于黄金的讨论热度和情绪倾向，为团队提供散户情绪温度计和反向指标参考。",
        enable_search=True,
        **_kw
    )
    pm_agent = PortfolioManagerAgent(
        name="首席策略官",
        sys_prompt="你是黄金基金的主理人。具备大局观、风控意识，不盲从任何单一指标。你需要综合技术面、宏观面、量化指标、深度学习预测、跨市场联动和情绪面的全部信息，做出最终决策。",
        **_kw
    )
    risk_agent = RiskControlAgent(
        name="风控合规官",
        sys_prompt="你是独立的风控合规审计官。你不参与前期讨论，唯一任务是对首席策略官的最终决策进行严格的风险审查，确保策略不会让基金暴露在过度风险中。",
        **_kw
    )

    # ================= 接收用户问题 =================
    default_question = "关注到今天行情异动，看看最近黄金走势，有没有类似头肩底？帮我全面诊脉。"
    try:
        user_input = input(f"\n请输入分析问题（直接回车使用默认）:\n> ").strip()
        question = user_input if user_input else default_question
    except (EOFError, KeyboardInterrupt):
        question = default_question
    print(f"\n分析问题: {question}\n")

    user_issue = Msg(name="Manager", content=question, role="user")

    # ================= 第一轮：6 位专家并行报告 =================
    print("\n=== 圆桌会议开始 (8位专家) ===")
    print("\n--- 第一轮：各领域专家独立报告（并行执行）---")

    agents_round1 = [
        analyst_agent,
        macro_agent,
        quant_agent,
        dl_agent,
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

    # ================= 第二轮：首席策略官综合决策 =================
    print("\n--- 第二轮：首席策略官综合研判 ---")
    pm_reply = pm_agent(round1_replies)
    print(f"\n{'=' * 50}\n[首席策略官 最终决议]:\n{pm_reply.content}\n{'=' * 50}")

    # ================= 第三轮：风控合规官独立审查 =================
    print("\n--- 第三轮：风控合规官独立审查 ---")
    risk_reply = risk_agent(pm_reply)
    print(f"\n{'=' * 50}\n[风控合规官 审查结论]:\n{risk_reply.content}\n{'=' * 50}")

    # ================= 保存报告 =================
    try:
        saved_path = _save_report(question, round1_replies, pm_reply, risk_reply)
        print(f"\n报告已保存至: {saved_path}")
    except Exception as e:
        print(f"\n报告保存失败: {e}")


if __name__ == "__main__":
    main()
