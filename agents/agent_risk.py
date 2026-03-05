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
            "请你作为独立风控合规官，从以下维度进行严格审查：\n"
            "1. 仓位风险：建议仓位是否超过总资金的10%？是否存在过度集中风险？\n"
            "2. 止损合理性：止损幅度是否在2-3%以内？距当前价是否太远或太近？\n"
            "3. 逻辑一致性：多头和空头理由是否自相矛盾？\n"
            "4. 黑天鹅预案：是否考虑了极端行情（如闪崩、跳空）的应对？\n"
            "5. 历史教训：是否与过去典型亏损模式（如逆大趋势做空牛市）相似？\n\n"
            "【输出要求】\n"
            "给出 🟢通过 / 🟡警告 / 🔴否决 的审查结论，并附上修改建议（如有）。(150字内)"
        )
        
        msg = Msg(name=x.name, role=x.role, content=audit_prompt)
        return super().reply(msg)
