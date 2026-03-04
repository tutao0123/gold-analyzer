"""
情绪分析师：利用大模型联网能力搜索社交媒体和论坛上的黄金讨论情绪
判断散户是极度贪婪还是恐慌，提供反向指标参考
"""
from agents.base_agent import LLMAgent, Msg


class SentimentAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        if x is None: return super().reply(x)
        print(f"\n[{self.name}] 正在通过联网搜索分析市场情绪面...")
        
        query = x.content if hasattr(x, 'content') else str(x)
        
        sentiment_prompt = (
            "【工作指令】请你利用联网搜索能力，完成以下情绪面分析：\n\n"
            "1. 搜索最近48小时内 Twitter/X、Reddit(r/Gold, r/WallStreetBets)、"
            "东方财富股吧、雪球等平台上关于黄金投资的热门讨论\n"
            "2. 判断当前散户情绪处于哪个阶段：极度贪婪 / 乐观 / 中性 / 悲观 / 极度恐慌\n"
            "3. 是否出现以下危险信号：\n"
            "   - 大量散户晒单做多（可能见顶信号）\n"
            "   - 铺天盖地的恐慌言论（可能见底信号）\n"
            "   - 黄金相关话题突然冲上热搜（情绪极端化）\n"
            "4. 给出情绪面对黄金的影响判断：利好 / 利空 / 中性\n\n"
            "【输出格式】\n"
            "情绪温度计：🟢贪婪/🟡中性/🔴恐慌\n"
            "反向指标建议：（如果散户极度看多，则提醒可能见顶）\n"
            "一句话结论（50字内）"
        )
        
        msg = Msg(name=x.name, role=x.role, content=sentiment_prompt)
        return super().reply(msg)
