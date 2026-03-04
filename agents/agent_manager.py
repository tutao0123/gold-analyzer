from agents.base_agent import LLMAgent, Msg

class PortfolioManagerAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        print(f"\n[{self.name}] 正在统筹各方意见，撰写最终交易报告 ...")
        # x 是来自上流 Pipeline 的一个列表，保存了三个/四个人的回答。
        # 拼装成一份会议纪要交给他。
        if isinstance(x, list):
            meeting_notes = "\n\n".join([f"【{msg.name} 的观点】:\n{msg.content}" for msg in x])
            prompt = (f"你是这支黄金基金的主理人。请你综合一下各位专家的意见，做出最后的交易风控决策。\n\n"
                      f"{meeting_notes}\n\n"
                      "【要求】\n给出明确的操作方向（做多/做空/观望），分仓建议，止盈止损点，并说明你的核心顾虑。(200字内)")
            x = Msg(name="System", role="user", content=prompt)
        
        return super().reply(x)
