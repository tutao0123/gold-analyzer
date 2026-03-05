"""
Gold price analysis module
Provides trend analysis, volatility calculation, technical indicators, and more
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    direction: str  # "up", "down", "sideways"
    strength: float  # 0-1 strength
    description: str


@dataclass
class VolatilityAnalysis:
    """Volatility analysis result"""
    daily_volatility: float  # daily volatility
    annualized_volatility: float  # annualized volatility
    volatility_level: str  # "low", "medium", "high"
    avg_daily_range: float  # average daily range


@dataclass
class SupportResistance:
    """Support and resistance levels"""
    support_levels: List[float]
    resistance_levels: List[float]


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    current_price: float
    price_change_1d: float
    price_change_7d: float
    price_change_30d: float
    trend: TrendAnalysis
    volatility: VolatilityAnalysis
    support_resistance: SupportResistance
    ma_analysis: Dict[str, float]  # moving average analysis
    rsi: float
    recommendation: str
    summary: str


class PriceAnalyzer:
    """Gold price analyzer"""

    def __init__(self, data: List[Dict]):
        """
        Initialize the analyzer.
        Args:
            data: list of price data dicts, each containing price, date, etc.
        """
        self.data = sorted(data, key=lambda x: x.get("date", x.get("timestamp", "")))
        self.prices = [d["price"] for d in self.data if "price" in d]
        self.dates = [d.get("date", d.get("timestamp", "")) for d in self.data]

        if len(self.prices) < 2:
            raise ValueError("Insufficient data points: at least 2 price records are required")

    def calculate_change(self, days: int) -> float:
        """Calculate price change percentage over the specified number of days"""
        if len(self.prices) < days + 1:
            days = len(self.prices) - 1
        if days <= 0:
            return 0.0

        current = self.prices[-1]
        past = self.prices[-(days + 1)]
        return (current - past) / past * 100

    def analyze_trend(self, short_window: int = 5, long_window: int = 20) -> TrendAnalysis:
        """
        Analyze price trend.
        Determines trend direction using the relationship between short-term and long-term moving averages.
        """
        if len(self.prices) < long_window:
            long_window = len(self.prices)
        if len(self.prices) < short_window:
            short_window = len(self.prices)

        # Calculate moving averages
        short_ma = mean(self.prices[-short_window:])
        long_ma = mean(self.prices[-long_window:])

        # Calculate trend strength
        diff_ratio = abs(short_ma - long_ma) / long_ma
        strength = min(diff_ratio * 100, 1.0)  # normalize to 0-1

        # Determine trend direction
        if short_ma > long_ma * 1.001:
            direction = "up"
            description = f"上涨趋势 (MA{short_window}=${short_ma:.2f} > MA{long_window}=${long_ma:.2f})"
        elif short_ma < long_ma * 0.999:
            direction = "down"
            description = f"下跌趋势 (MA{short_window}=${short_ma:.2f} < MA{long_window}=${long_ma:.2f})"
        else:
            direction = "sideways"
            description = f"横盘整理 (MA{short_window}=${short_ma:.2f} ≈ MA{long_window}=${long_ma:.2f})"

        return TrendAnalysis(direction, strength, description)

    def calculate_volatility(self) -> VolatilityAnalysis:
        """Calculate volatility metrics"""
        if len(self.prices) < 2:
            return VolatilityAnalysis(0, 0, "unknown", 0)

        # Calculate daily returns
        returns = []
        daily_ranges = []

        for i in range(1, len(self.data)):
            prev_price = self.data[i - 1]["price"]
            curr_price = self.data[i]["price"]
            daily_return = (curr_price - prev_price) / prev_price
            returns.append(daily_return)

            # Calculate daily range (High - Low) / Open
            high = self.data[i].get("high", curr_price)
            low = self.data[i].get("low", curr_price)
            open_price = self.data[i].get("open", curr_price)
            if open_price and open_price > 0:
                daily_range = (high - low) / open_price
                daily_ranges.append(daily_range)

        if not returns:
            return VolatilityAnalysis(0, 0, "unknown", 0)

        # Daily volatility (standard deviation)
        daily_vol = stdev(returns) if len(returns) > 1 else 0

        # Annualized volatility (assuming 252 trading days)
        annualized_vol = daily_vol * (252 ** 0.5)

        # Volatility level
        if annualized_vol < 0.1:
            vol_level = "low"
        elif annualized_vol < 0.2:
            vol_level = "medium"
        else:
            vol_level = "high"

        # Average daily range
        avg_range = mean(daily_ranges) * 100 if daily_ranges else 0

        return VolatilityAnalysis(
            daily_volatility=daily_vol,
            annualized_volatility=annualized_vol,
            volatility_level=vol_level,
            avg_daily_range=avg_range,
        )

    def find_support_resistance(self, window: int = 5) -> SupportResistance:
        """
        Find support and resistance levels.
        Uses local minima/maxima method.
        """
        prices = self.prices
        if len(prices) < window * 2 + 1:
            return SupportResistance([], [])

        supports = []
        resistances = []

        for i in range(window, len(prices) - window):
            # 检查是否是局部最小值 (支撑)
            is_support = all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and \
                        all(prices[i] <= prices[i + j] for j in range(1, window + 1))

            # 检查是否是局部最大值 (阻力)
            is_resistance = all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and \
                           all(prices[i] >= prices[i + j] for j in range(1, window + 1))

            if is_support:
                supports.append(prices[i])
            if is_resistance:
                resistances.append(prices[i])

        # 聚类相近的水平
        supports = self._cluster_levels(supports, threshold=0.01)
        resistances = self._cluster_levels(resistances, threshold=0.01)

        # 只保留最近的3个
        current_price = prices[-1]
        supports = sorted([s for s in supports if s < current_price], reverse=True)[:3]
        resistances = sorted([r for r in resistances if r > current_price])[:3]

        return SupportResistance(supports, resistances)

    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """聚类相近的价格水平"""
        if not levels:
            return []

        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for i in range(1, len(levels)):
            if (levels[i] - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(levels[i])
            else:
                clusters.append(mean(current_cluster))
                current_cluster = [levels[i]]

        clusters.append(mean(current_cluster))
        return clusters

    def calculate_moving_averages(self) -> Dict[str, float]:
        """计算各种移动平均线"""
        mas = {}
        periods = [5, 10, 20, 30]

        for period in periods:
            if len(self.prices) >= period:
                mas[f"MA{period}"] = round(mean(self.prices[-period:]), 2)

        return mas

    def calculate_rsi(self, period: int = 14) -> float:
        """
        计算 RSI (相对强弱指标)
        RSI > 70: 超买, RSI < 30: 超卖
        """
        if len(self.prices) < period + 1:
            period = len(self.prices) - 1

        if period <= 0:
            return 50.0

        gains = []
        losses = []

        for i in range(1, period + 1):
            change = self.prices[-i] - self.prices[-(i + 1)]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = mean(gains) if gains else 0
        avg_loss = mean(losses) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    def generate_recommendation(self, analysis: AnalysisResult) -> str:
        """生成交易建议"""
        signals = []

        # 趋势信号
        if analysis.trend.direction == "up":
            signals.append("bullish")
        elif analysis.trend.direction == "down":
            signals.append("bearish")
        else:
            signals.append("neutral")

        # RSI 信号
        if analysis.rsi > 70:
            signals.append("overbought")
        elif analysis.rsi < 30:
            signals.append("oversold")

        # 波动率信号
        if analysis.volatility.volatility_level == "high":
            signals.append("high_volatility")

        # 综合建议
        if "bullish" in signals and "oversold" in signals:
            return "强烈买入 - 上涨趋势且超卖"
        elif "bullish" in signals:
            return "买入 - 上涨趋势"
        elif "bearish" in signals and "overbought" in signals:
            return "强烈卖出 - 下跌趋势且超买"
        elif "bearish" in signals:
            return "卖出 - 下跌趋势"
        elif "oversold" in signals:
            return "观望买入 - 超卖状态"
        elif "overbought" in signals:
            return "观望卖出 - 超买状态"
        else:
            return "观望 - 趋势不明"

    def analyze(self) -> AnalysisResult:
        """执行完整分析"""
        logger.info("开始分析黄金价格数据...")

        current_price = self.prices[-1]

        # 计算涨跌幅
        change_1d = self.calculate_change(1)
        change_7d = self.calculate_change(7)
        change_30d = self.calculate_change(30)

        # 趋势分析
        trend = self.analyze_trend()

        # 波动率分析
        volatility = self.calculate_volatility()

        # 支撑阻力
        sr = self.find_support_resistance()

        # 移动平均线
        mas = self.calculate_moving_averages()

        # RSI
        rsi = self.calculate_rsi()

        # 创建结果对象
        result = AnalysisResult(
            current_price=current_price,
            price_change_1d=change_1d,
            price_change_7d=change_7d,
            price_change_30d=change_30d,
            trend=trend,
            volatility=volatility,
            support_resistance=sr,
            ma_analysis=mas,
            rsi=rsi,
            recommendation="",
            summary="",
        )

        # 生成建议
        result.recommendation = self.generate_recommendation(result)

        # 生成总结
        result.summary = self._generate_summary(result)

        logger.info("分析完成")
        return result

    def _generate_summary(self, result: AnalysisResult) -> str:
        """生成分析总结"""
        summary_parts = [
            f"当前金价: ${result.current_price:.2f}",
            f"1日涨跌: {result.price_change_1d:+.2f}%",
            f"7日涨跌: {result.price_change_7d:+.2f}%",
            f"30日涨跌: {result.price_change_30d:+.2f}%",
            f"趋势: {result.trend.description}",
            f"RSI: {result.rsi:.1f}",
            f"波动率: {result.volatility.volatility_level.upper()} "
            f"({result.volatility.annualized_volatility * 100:.1f}% 年化)",
        ]

        if result.support_resistance.support_levels:
            supports = ", ".join([f"${s:.2f}" for s in result.support_resistance.support_levels])
            summary_parts.append(f"支撑位: {supports}")

        if result.support_resistance.resistance_levels:
            resistances = ", ".join([f"${r:.2f}" for r in result.support_resistance.resistance_levels])
            summary_parts.append(f"阻力位: {resistances}")

        summary_parts.append(f"建议: {result.recommendation}")

        return "\n".join(summary_parts)


if __name__ == "__main__":
    # 测试代码
    test_data = [
        {"date": "2024-01-01", "price": 2000 + i * 5 + (i % 3) * 10}
        for i in range(30)
    ]

    analyzer = PriceAnalyzer(test_data)
    result = analyzer.analyze()
    print(result.summary)
