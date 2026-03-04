import logging
from agents.base_agent import LLMAgent, Msg
from core.analyzer import GoldPriceAnalyzer
from core.data_fetcher import GoldDataFetcher

logger = logging.getLogger(__name__)


class QuantEngineerAgent(LLMAgent):
    def reply(self, x: dict = None) -> dict:
        if x is None:
            return super().reply(x)
        print(f"\n[{self.name}] 正在调用 analyzer.py 计算本地行情指标 ...")
        try:
            fetcher = GoldDataFetcher()
            data = fetcher.generate_historical_data(30)
            analyzer = GoldPriceAnalyzer(data)
            res = analyzer.analyze()

            # 格式化均线
            ma_str = "  ".join([f"{k}=${v:.2f}" for k, v in res.ma_analysis.items()])

            # 格式化支撑/阻力位
            support_str = (
                ", ".join([f"${s:.2f}" for s in res.support_resistance.support_levels])
                if res.support_resistance.support_levels else "暂无"
            )
            resistance_str = (
                ", ".join([f"${r:.2f}" for r in res.support_resistance.resistance_levels])
                if res.support_resistance.resistance_levels else "暂无"
            )

            # RSI 状态
            if res.rsi > 70:
                rsi_status = "超买"
            elif res.rsi < 30:
                rsi_status = "超卖"
            else:
                rsi_status = "中性"

            quant_str = (
                f"【实时量化计算结果】\n"
                f"当前金价:    ${res.current_price:.2f}\n"
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

            # 直接返回结构化数据，不经过 LLM 二次加工
            return Msg(name=self.name, role="assistant", content=quant_str)

        except Exception as e:
            logger.error(f"量化指标计算失败: {e}", exc_info=True)
            return Msg(name=self.name, role="assistant", content=f"【量化指标获取失败】{e}")
