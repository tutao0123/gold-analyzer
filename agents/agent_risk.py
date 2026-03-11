"""
Risk & compliance officer: independently reviews the chief strategist's decisions,
checking whether positions exceed limits, whether stop-losses are reasonable,
and whether the trade resembles historical loss patterns.
"""
from agents.base_agent import LLMAgent, Msg


class RiskControlAgent(LLMAgent):
    """
    Performs an independent audit of the chief strategist's decision after it is made.
    Does not participate in earlier discussions; only conducts a compliance review at the end.
    """
    def reply(self, x: dict = None) -> dict:
        if x is None: return super().reply(x)
        print(f"\n[{self.name}] Running independent risk-control review of chief strategist's decision...")
        
        content = x.content if hasattr(x, 'content') else str(x)
        
        audit_prompt = (
            f"【风控审查任务】\n"
            f"以下是首席策略官刚刚做出的交易决策：\n\n{content}\n\n"
            "请你作为独立风控合规官，从以下维度进行灵活审查：\n"
            "1. 仓位风险：建议仓位是否超过总资金的 50%？（高胜率信号可放宽至 50%，常规 20%-30%）\n"
            "2. 止损合理性：止损幅度是否在 3-5% 以内？（根据波动率动态调整，低波动可收紧，高波动可放宽）\n"
            "3. 逻辑一致性：多头和空头理由是否严重自相矛盾？（轻微矛盾可接受，市场本就复杂）\n"
            "4. 黑天鹅预案：是否有基本的极端行情应对意识？（不要求完美预案，有止损即可）\n"
            "5. 历史教训：是否与过去典型亏损模式高度相似？（仅在大方向错误时才否决）\n\n"
            "【审查原则】\n"
            "- 宁可适度冒险，不可过度保守错失良机\n"
            "- 高胜率 + 低波动 → 🟢通过\n"
            "- 中等信号 + 可控风险 → 🟡警告但通过\n"
            "- 仅在大方向错误或风险失控时 → 🔴否决\n\n"
            "【输出要求】\n"
            "给出 🟢通过 / 🟡警告 / 🔴否决 的审查结论，并附上修改建议（如有）。(150 字内)"
        )
        
        msg = Msg(name=x.name, role=x.role, content=audit_prompt)
        return super().reply(msg)
