"""
Macro fundamental analyst: fetches hard data such as the yield curve and bond
markets, then uses LLM web search to supplement the latest Fed policy,
economic data, and geopolitical developments.
"""
import logging
from agents.base_agent import LLMAgent, Msg

logger = logging.getLogger(__name__)

# Macro indicators: yield curve, bond markets, risk appetite
_MACRO_TICKERS = {
    "^IRX":  "美国3个月国债收益率(短端)",
    "^FVX":  "美国5年期国债收益率",
    "^TNX":  "美国10年期国债收益率",
    "^TYX":  "美国30年期国债收益率",
    "TLT":   "20年+长期国债ETF(TLT)",
    "HYG":   "高收益债ETF(HYG/风险偏好)",
    "^GSPC": "标普500(风险资产基准)",
}


def _fetch_macro_data() -> str:
    """Fetch hard macro data via yfinance and return a formatted string. Returns empty string on failure."""
    try:
        import yfinance as yf
        import pandas as pd

        lines = []
        for ticker, name in _MACRO_TICKERS.items():
            try:
                df = yf.download(ticker, period="5d", interval="1d",
                                 progress=False, auto_adjust=True)
                if df.empty or len(df) < 2:
                    continue
                latest = float(df["Close"].iloc[-1].item() if hasattr(df["Close"].iloc[-1], 'item') else df["Close"].iloc[-1])
                prev   = float(df["Close"].iloc[-2].item() if hasattr(df["Close"].iloc[-2], 'item') else df["Close"].iloc[-2])
                chg    = (latest - prev) / prev * 100
                lines.append(f"  {name}: {latest:.3f} ({chg:+.2f}%)")
            except Exception as e:
                logger.debug(f"Skipping {ticker}: {e}")

        if not lines:
            return ""

        # yield curve shape analysis
        try:
            irx_df  = yf.download("^IRX",  period="2d", progress=False, auto_adjust=True)
            tnx_df  = yf.download("^TNX",  period="2d", progress=False, auto_adjust=True)
            if not irx_df.empty and not tnx_df.empty:
                irx = float(irx_df["Close"].iloc[-1].item() if hasattr(irx_df["Close"].iloc[-1], 'item') else irx_df["Close"].iloc[-1])
                tnx = float(tnx_df["Close"].iloc[-1].item() if hasattr(tnx_df["Close"].iloc[-1], 'item') else tnx_df["Close"].iloc[-1])
                spread = tnx - irx
                if spread < 0:
                    curve_desc = f"⚠️ 倒挂 (10y-3m={spread:+.2f}bp，衰退信号，历史上利好黄金)"
                elif spread < 0.5:
                    curve_desc = f"趋于平坦 (10y-3m={spread:+.2f}bp)"
                else:
                    curve_desc = f"正常陡峭 (10y-3m={spread:+.2f}bp)"
                lines.append(f"  收益率曲线形态: {curve_desc}")
        except Exception:
            pass

        return "\n".join(lines)

    except ImportError:
        return ""
    except Exception as e:
        logger.warning(f"Failed to fetch macro data: {e}")
        return ""


class MacroAnalystAgent(LLMAgent):
    """
    Macro fundamental analyst: first fetches hard data (yield curve, etc.) via yfinance,
    then lets the LLM supplement with the latest Fed policy, economic releases, and geopolitics.
    """

    def reply(self, x=None):
        if x is None:
            return super().reply(x)

        print(f"\n[{self.name}] Fetching hard macro data (yield curve / bonds / risk appetite)...")
        macro_data = _fetch_macro_data()

        query = x.content if hasattr(x, "content") else str(x)

        if macro_data:
            data_section = f"【宏观市场实时数据】\n{macro_data}\n\n"
        else:
            data_section = "【宏观实时数据暂时不可用，请依赖联网搜索】\n\n"

        prompt = (
            f"【工作指令】主理人关心：{query}\n\n"
            f"{data_section}"
            "请你基于以上实时数据，并利用联网搜索能力，补充近24小时内：\n"
            "1. 美联储最新表态（降息/加息预期变化、官员讲话或会议纪要）\n"
            "2. 近期重要经济数据（CPI、非农、GDP 修正值等）的实际值 vs 预期值\n"
            "3. 地缘政治风险（中东、俄乌、中美贸易等）有无新增催化剂\n\n"
            "综合以上，判断当前宏观环境对黄金的影响方向（利好/利空/中性），"
            "并说明核心逻辑。(150字内)"
        )

        new_msg = Msg(name=x.name, role=x.role, content=prompt)
        return super().reply(new_msg)
