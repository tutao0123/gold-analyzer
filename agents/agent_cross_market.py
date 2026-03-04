"""
跨市场联动分析师：追踪美元指数、美债收益率、原油、VIX 与黄金的联动关系
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import LLMAgent, Msg

class CrossMarketAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        if x is None: return super().reply(x)
        print(f"\n[{self.name}] 正在抓取跨市场联动数据 (DXY/美债/VIX/原油)...")
        
        try:
            import yfinance as yf
            import pandas as pd
            
            # 抓取关联品种最近 5 天的数据
            tickers = {
                "DX-Y.NYB": "美元指数(DXY)",
                "^TNX": "美国10年期国债收益率",
                "^VIX": "VIX恐慌指数",
                "CL=F": "WTI原油",
                "GC=F": "黄金(基准)",
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
            
            # 附加分析提示
            prompt = (
                f"【工作指令】以下是跨市场联动数据：\n{cross_data}\n\n"
                "请你作为跨市场联动分析师，分析以上品种之间的联动关系对黄金的影响。"
                "重点关注：美元与黄金的负相关是否成立？VIX 是否暗示避险需求？美债收益率走势如何影响持金成本？(100字内)"
            )
            msg = Msg(name=x.name, role=x.role, content=prompt)
            return super().reply(msg)
            
        except Exception as e:
            msg = Msg(name=x.name, role=x.role, content=f"跨市场数据获取异常：{e}，请基于经验给出分析。")
            return super().reply(msg)
