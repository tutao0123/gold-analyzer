from agents.base_agent import LLMAgent, Msg

class PortfolioManagerAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        print(f"\n[{self.name}] Synthesising all expert opinions, drafting final trade report ...")
        # x is a list from the upstream pipeline containing replies from all specialists.
        # Assemble them into meeting minutes and pass to the portfolio manager.
        if isinstance(x, list):
            meeting_notes = "\n\n".join([f"【{msg.name} 的观点】:\n{msg.content}" for msg in x])
            prompt = (f"你是这支黄金基金的主理人。请你综合一下各位专家的意见，做出最后的交易风控决策。\n\n"
                      f"{meeting_notes}\n\n"
                      "【要求】\n给出明确的操作方向（做多/做空/观望），分仓建议，止盈止损点，并说明你的核心顾虑。(200字内)")
            x = Msg(name="System", role="user", content=prompt)
        
        return super().reply(x)
