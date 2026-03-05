"""
Cross-market correlation analyst: tracks the linkage between the dollar index,
US Treasury yields, crude oil, VIX, and gold.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import LLMAgent, Msg

class CrossMarketAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        if x is None: return super().reply(x)
        print(f"\n[{self.name}] Fetching cross-market correlation data (DXY/Treasuries/VIX/Crude)...")
        
        try:
            import yfinance as yf
            import pandas as pd
            
            # fetch the last 5 days of data for correlated instruments
            tickers = {
                "DX-Y.NYB": "美元指数(DXY)",
                "^TNX": "美国10年期国债收益率",
                "^VIX": "VIX恐慌指数",
                "CL=F": "WTI原油",
                self.commodity["symbol"]: f"{self.commodity['name']}(当前分析标的)",
            }
            
            data_lines = []
            for ticker, name in tickers.items():
                try:
                    df = yf.download(ticker, period="5d", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    if not df.empty and len(df) >= 2:
                        latest = df['Close'].iloc[-1]
                        prev = df['Close'].iloc[-2]
                        change = (latest - prev) / prev * 100
                        data_lines.append(f"  {name}: {latest:.2f} ({change:+.2f}%)")
                    else:
                        data_lines.append(f"  {name}: 数据暂缺")
                except Exception:
                    data_lines.append(f"  {name}: 获取失败")
            
            cross_data = "【跨市场联动实时数据】\n" + "\n".join(data_lines)
            
            # append analysis prompt
            prompt = (
                f"【工作指令】以下是跨市场联动数据：\n{cross_data}\n\n"
                f"请你作为跨市场联动分析师，分析以上品种之间的联动关系对{self.commodity['name']}的影响。"
                f"重点关注：美元与{self.commodity['name']}的负相关是否成立？VIX 是否暗示避险需求？美债收益率走势如何影响持仓成本？(100字内)"
            )
            msg = Msg(name=x.name, role=x.role, content=prompt)
            return super().reply(msg)
            
        except Exception as e:
            msg = Msg(name=x.name, role=x.role, content=f"跨市场数据获取异常：{e}，请基于经验给出分析。")
            return super().reply(msg)
