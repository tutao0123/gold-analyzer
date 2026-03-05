import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import LLMAgent, Msg

class RAGAnalystAgent(LLMAgent):
    def __init__(self, name, sys_prompt, rag_db, **kwargs):
        super().__init__(name, sys_prompt, **kwargs)
        self.rag_db = rag_db

    def _get_current_price(self):
        """Fetch the current price to use as the analysis baseline."""
        try:
            import pandas as pd
            commodity_key = self.commodity["key"]
            if commodity_key == "gold":
                csv_name = "gc_f_full_history.csv"
            else:
                csv_name = f"{commodity_key}_full_history.csv"
            csv_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", csv_name
            )
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                latest_price = df['Close'].iloc[-1]
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                return latest_price, latest_date
        except Exception:
            pass
        return None, None

    def reply(self, x: dict = None) -> dict:
        if x is None: return super().reply(x)

        query = x.content if hasattr(x, 'content') else str(x)
        print(f"\n[{self.name}] Searching knowledge base for chart pattern memories ...")

        try:
            commodity_key = self.commodity["key"]
            commodity_name = self.commodity["name"]

            # get current price
            current_price, latest_date = self._get_current_price()
            price_info = ""
            if current_price:
                price_info = f"【当前最新价格】{current_price:.2f} {self.commodity['unit']}（截至 {latest_date}）\n"

            # for non-gold commodities or when the RAG store is empty, skip RAG and use LLM technical analysis directly
            if commodity_key != "gold":
                context = f"【当前品种 {commodity_name} 暂无历史图形记忆库】请基于价格走势直接进行技术形态分析。"
            else:
                # retrieve historical chart pattern memories from RAG (top 3)
                results = self.rag_db.search(query_text=query, top_k=3)
                context = ""
                if results:
                    matched = [r for r in results if r['distance'] < 1.0]
                    if matched:
                        for i, r in enumerate(matched):
                            context += f"【历史记忆 {i+1}】{r['text']}\n"
                            if r['image']:
                                context += f"  参考图表: {r['image']}\n"
                    else:
                        context = "【知识库暂无高度匹配的图形记忆】\n"
                else:
                    context = "【知识库暂无匹配图形记忆】\n"

            prompt = (
                f"【工作指令】主理人关心的方向：{query}\n"
                f"{price_info}"
                f"我为你从历史图形记忆库中检索到了以下参考信息：\n{context}\n"
                f"⚠️ 注意：以上为历史参考，请务必基于【当前最新价格】进行分析，"
                "不要直接搬用历史价格作为当前技术位。"
                f"请根据历史形态的相似性，推算当前价格结构下的关键支撑/阻力位(100字内)"
            )
            new_msg = Msg(name=x.name, role=x.role, content=prompt)
        except Exception as e:
            new_msg = x

        return super().reply(new_msg)
