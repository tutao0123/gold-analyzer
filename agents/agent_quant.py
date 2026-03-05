import logging
import pandas as pd
import yfinance as yf
from agents.base_agent import LLMAgent, Msg
from core.analyzer import GoldPriceAnalyzer

logger = logging.getLogger(__name__)


class QuantEngineerAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        if x is None:
            return super().reply(x)
        print(f"\n[{self.name}] Calling analyzer.py to compute local market indicators ...")
        try:
            ticker = self.commodity["symbol"]
            df = yf.download(ticker, period="40d", interval="1d", progress=False, auto_adjust=True)

            if df.empty:
                return Msg(name=self.name, role="assistant", content=f"【量化指标获取失败】yfinance 返回空数据 ({ticker})")

            # flatten multi-level column index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # convert to the [{price, date}] format expected by GoldPriceAnalyzer
            data = []
            for idx, row in df.iterrows():
                close_val = row["Close"]
                if pd.isna(close_val):
                    continue
                price = float(close_val.iloc[0] if hasattr(close_val, "__len__") else close_val)
                data.append({"date": str(idx.date()), "price": price})

            if not data:
                return Msg(name=self.name, role="assistant", content="【量化指标获取失败】价格数据为空")

            analyzer = GoldPriceAnalyzer(data)
            res = analyzer.analyze()

            # format moving averages
            ma_str = "  ".join([f"{k}=${v:.2f}" for k, v in res.ma_analysis.items()])

            # format support / resistance levels
            support_str = (
                ", ".join([f"${s:.2f}" for s in res.support_resistance.support_levels])
                if res.support_resistance.support_levels else "暂无"
            )
            resistance_str = (
                ", ".join([f"${r:.2f}" for r in res.support_resistance.resistance_levels])
                if res.support_resistance.resistance_levels else "暂无"
            )

            # RSI status
            if res.rsi > 70:
                rsi_status = "超买"
            elif res.rsi < 30:
                rsi_status = "超卖"
            else:
                rsi_status = "中性"

            commodity_name = self.commodity["name"]
            unit = self.commodity["unit"]
            quant_str = (
                f"【实时量化计算结果】品种: {commodity_name}\n"
                f"当前价格:    {res.current_price:.2f} {unit}\n"
                f"涨跌幅:      1日 {res.price_change_1d:+.2f}%  "
                f"7日 {res.price_change_7d:+.2f}%  "
                f"30日 {res.price_change_30d:+.2f}%\n"
                f"RSI(14):     {res.rsi:.1f}  [{rsi_status}]\n"
                f"波动率:      {res.volatility.volatility_level.upper()}  "
                f"(年化 {res.volatility.annualized_volatility:.1%}  "
                f"日均波幅 {res.volatility.avg_daily_range:.2f}%)\n"
                f"趋势:        {res.trend.description}\n"
                f"均线:        {ma_str}\n"
                f"支撑位:      {support_str}\n"
                f"阻力位:      {resistance_str}\n"
                f"量化建议:    {res.recommendation}"
            )

            # return structured data directly, skipping a second LLM pass
            return Msg(name=self.name, role="assistant", content=quant_str)

        except Exception as e:
            logger.error(f"Failed to compute quant indicators: {e}", exc_info=True)
            return Msg(name=self.name, role="assistant", content=f"【量化指标获取失败】{e}")
