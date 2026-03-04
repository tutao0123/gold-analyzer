"""
LLM 智能分析模块
使用阿里云百炼大模型 API 进行黄金价格投资分析
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

from openai import OpenAI
from config import BAILIAN_API_KEY, BAILIAN_BASE_URL, DEFAULT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """LLM 分析结果"""
    market_sentiment: str  # 市场情绪
    risk_assessment: str   # 风险评估
    action_advice: str     # 操作建议
    future_prediction: str # 未来预测
    detailed_analysis: str # 详细分析


class LLMAnalyzer:
    """百炼大模型分析器"""

    def __init__(self):
        self.api_key = BAILIAN_API_KEY
        self.base_url = BAILIAN_BASE_URL
        self.model = DEFAULT_MODEL
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    def _build_prompt(self, technical_data: Dict[str, Any]) -> str:
        """构建 LLM 提示词"""

        # 提取技术指标数据
        current_price = technical_data.get('current_price', 0)
        price_change_1d = technical_data.get('price_change_1d', 0)
        price_change_7d = technical_data.get('price_change_7d', 0)
        price_change_30d = technical_data.get('price_change_30d', 0)
        trend = technical_data.get('trend', {})
        volatility = technical_data.get('volatility', {})
        rsi = technical_data.get('rsi', 50)
        ma_analysis = technical_data.get('ma_analysis', {})
        support_levels = technical_data.get('support_resistance', {}).get('support_levels', [])
        resistance_levels = technical_data.get('support_resistance', {}).get('resistance_levels', [])
        technical_recommendation = technical_data.get('recommendation', '')

        prompt = f"""你是一位专业的黄金投资分析师，拥有丰富的金融市场经验。
请基于以下技术指标数据，进行专业的投资分析和预测。

【技术指标数据】
- 当前金价: ${current_price:.2f} USD/oz
- 1日涨跌幅: {price_change_1d:+.2f}%
- 7日涨跌幅: {price_change_7d:+.2f}%
- 30日涨跌幅: {price_change_30d:+.2f}%
- 趋势方向: {trend.get('direction', 'unknown')}
- 趋势强度: {trend.get('strength', 0):.1%}
- 趋势描述: {trend.get('description', '')}
- RSI指标: {rsi:.1f} (70以上超买，30以下超卖)
- 波动率等级: {volatility.get('volatility_level', 'unknown')}
- 年化波动率: {volatility.get('annualized_volatility', 0)*100:.1f}%
- 移动平均线: {json.dumps(ma_analysis, ensure_ascii=False)}
- 支撑位: {support_levels}
- 阻力位: {resistance_levels}
- 技术分析建议: {technical_recommendation}

【输出格式要求】
请用中文输出以下四个部分，每部分简洁专业（每部分100-200字）：

1. 市场情绪分析：基于当前价格和近期走势，分析市场参与者的情绪状态
2. 风险评估：识别当前投资黄金的主要风险因素，包括技术面和宏观面
3. 操作建议：给出具体的操作策略，包括入场点、止损位、目标位
4. 未来预测：预测未来1-4周的价格走势，给出可能的价格区间

请确保分析专业、客观，既有技术面依据，也考虑宏观经济因素。"""

        return prompt

    def analyze(self, technical_result: Any) -> LLMAnalysisResult:
        """
        调用百炼 API 进行智能分析

        Args:
            technical_result: 技术分析结果对象 (AnalysisResult)

        Returns:
            LLMAnalysisResult: LLM 分析结果
        """
        logger.info("开始调用百炼大模型进行智能分析...")
        return self._do_analyze(technical_result, enable_search=False)

    def analyze_with_search(self, technical_result: Any) -> LLMAnalysisResult:
        """
        调用百炼 API 进行具备联网搜索能力的智能分析
        """
        logger.info("开始调用百炼大模型进行智能分析(启用联网增强)...")
        # 联网搜索建议用plus以上模型，已经在 config 设定为 qwen3.5-plus
        return self._do_analyze(technical_result, enable_search=True)
        
    def analyze_document(self, file_path: str, query: str = "这篇文章讲了什么相关信息？可以结合当前市场行情进行分析吗？") -> str:
        """
        上传文件并使用大模型进行长文档分析理解
        
        Args:
            file_path: 文档所在路径
            query: 查询/提问的话术
        """
        if not self.client:
            return "未配置 BAILIAN_API_KEY，无法使用文档分析功能"
            
        try:
            # 1. 上传文件获取 id
            logger.info(f"正在上传并处理文档: {file_path}")
            file_object = self.client.files.create(file=Path(file_path), purpose="file-extract")
            logger.info(f"文档上传成功，fileID: {file_object.id}")
            
            # 2. 调用模型分析
            completion = self.client.chat.completions.create(
                model="qwen-long",  # 明确限制为文档理解模型 qwen-long
                messages=[
                    {'role': 'system', 'content': f'fileid://{file_object.id}'},
                    {'role': 'user', 'content': query}
                ]
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"文档分析失败: {e}")
            return f"文档分析失败：{str(e)}"

    def _do_analyze(self, technical_result: Any, enable_search: bool = False) -> LLMAnalysisResult:
        """通用核心分析逻辑"""
        # 如果没有配置 API Key，返回模拟数据
        if not self.client:
            logger.warning("未配置 BAILIAN_API_KEY，使用模拟 LLM 分析结果")
            return self._get_mock_analysis()

        # 准备技术指标数据(为安全，用__dict__或转换字典保护)
        technical_data = {
            'current_price': technical_result.current_price,
            'price_change_1d': technical_result.price_change_1d,
            'price_change_7d': technical_result.price_change_7d,
            'price_change_30d': technical_result.price_change_30d,
            'trend': {
                'direction': technical_result.trend.direction,
                'strength': technical_result.trend.strength,
                'description': technical_result.trend.description,
            },
            'volatility': {
                'volatility_level': technical_result.volatility.volatility_level,
                'annualized_volatility': technical_result.volatility.annualized_volatility,
                'daily_volatility': technical_result.volatility.daily_volatility,
            },
            'rsi': technical_result.rsi,
            'ma_analysis': technical_result.ma_analysis,
            'support_resistance': {
                'support_levels': technical_result.support_resistance.support_levels,
                'resistance_levels': technical_result.support_resistance.resistance_levels,
            },
            'recommendation': technical_result.recommendation,
        }

        # 构建提示词
        prompt = self._build_prompt(technical_data)

        try:
            extra_body = {"enable_search": True} if enable_search else None
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的黄金投资分析师，擅长技术分析和基本面分析。请结合当前网络最新金融资讯（如果可用）提供客观、专业的投资建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                extra_body=extra_body
            )
            content = completion.choices[0].message.content
            
            logger.info("百炼大模型分析完成")

            # 解析 LLM 输出
            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM 分析出错: {e}")
            return self._get_fallback_analysis(technical_data)

    def _parse_llm_response(self, content: str) -> LLMAnalysisResult:
        """解析 LLM 的响应内容"""

        # 尝试按章节分割
        sections = {
            'market_sentiment': '',
            'risk_assessment': '',
            'action_advice': '',
            'future_prediction': ''
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 识别章节标题
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ['市场情绪', 'sentiment', '情绪分析']):
                current_section = 'market_sentiment'
                continue
            elif any(keyword in lower_line for keyword in ['风险评估', 'risk', '风险分析']):
                current_section = 'risk_assessment'
                continue
            elif any(keyword in lower_line for keyword in ['操作建议', 'action', '操作', '建议']):
                current_section = 'action_advice'
                continue
            elif any(keyword in lower_line for keyword in ['未来预测', 'prediction', '预测', '走势']):
                current_section = 'future_prediction'
                continue

            # 累积内容
            if current_section and not line.startswith('1.') and not line.startswith('2.') and not line.startswith('3.') and not line.startswith('4.'):
                if sections[current_section]:
                    sections[current_section] += ' ' + line
                else:
                    sections[current_section] = line

        # 如果解析失败，使用整个内容作为详细分析
        if not any(sections.values()):
            return LLMAnalysisResult(
                market_sentiment="AI分析：" + content[:200] + "...",
                risk_assessment="请查看详细分析",
                action_advice="请查看详细分析",
                future_prediction="请查看详细分析",
                detailed_analysis=content
            )

        return LLMAnalysisResult(
            market_sentiment=sections['market_sentiment'] or "分析数据不足",
            risk_assessment=sections['risk_assessment'] or "请查看详细分析",
            action_advice=sections['action_advice'] or "请查看详细分析",
            future_prediction=sections['future_prediction'] or "请查看详细分析",
            detailed_analysis=content
        )

    def _get_mock_analysis(self) -> LLMAnalysisResult:
        """获取模拟分析结果（用于测试）"""
        return LLMAnalysisResult(
            market_sentiment="【模拟数据】当前市场情绪中性偏谨慎。近期金价波动幅度适中，投资者观望情绪较浓，等待更明确的方向信号。",
            risk_assessment="【模拟数据】主要风险包括：1) 美联储货币政策不确定性；2) 地缘政治风险变化；3) 美元汇率波动。建议控制仓位，设置止损。",
            action_advice="【模拟数据】建议采取区间操作策略。在支撑位附近轻仓试多，阻力位附近减仓。单次交易风险不超过本金的2%。",
            future_prediction="【模拟数据】预计未来1-4周金价将在当前价位附近震荡整理，波动区间约为±3%。突破关键阻力/支撑后将出现趋势性行情。",
            detailed_analysis="【模拟数据】这是模拟的LLM分析结果。如需真实AI分析，请配置 BAILIAN_API_KEY 环境变量。"
        )

    def _get_fallback_analysis(self, technical_data: Dict) -> LLMAnalysisResult:
        """获取备用分析结果（API调用失败时使用）"""

        trend_direction = technical_data.get('trend', {}).get('direction', 'unknown')
        rsi = technical_data.get('rsi', 50)
        volatility = technical_data.get('volatility', {}).get('volatility_level', 'medium')

        # 基于技术指标生成简单分析
        if trend_direction == 'up':
            sentiment = "当前市场情绪偏多，价格处于上升通道。"
            prediction = "预计短期将维持上涨态势，但需警惕超买回调风险。"
        elif trend_direction == 'down':
            sentiment = "当前市场情绪偏空，价格处于下降通道。"
            prediction = "预计短期将维持弱势震荡，关注超卖反弹机会。"
        else:
            sentiment = "当前市场情绪中性，价格处于横盘整理阶段。"
            prediction = "预计短期将继续震荡整理，等待方向突破。"

        if rsi > 70:
            sentiment += "RSI显示超买，需警惕回调。"
        elif rsi < 30:
            sentiment += "RSI显示超卖，可能存在反弹机会。"

        if volatility == 'high':
            risk = "当前市场波动率较高，风险较大，建议降低仓位。"
        elif volatility == 'low':
            risk = "当前市场波动率较低，风险可控，但需警惕波动率突然放大。"
        else:
            risk = "当前市场波动率适中，风险水平正常。"

        return LLMAnalysisResult(
            market_sentiment=sentiment,
            risk_assessment=risk,
            action_advice="建议观望或轻仓操作，严格设置止损止盈。",
            future_prediction=prediction,
            detailed_analysis="基于技术指标的备用分析（LLM API调用失败）"
        )


def analyze_with_llm(technical_result: Any) -> LLMAnalysisResult:
    """
    便捷函数：使用 LLM 分析技术分析结果

    Args:
        technical_result: 技术分析结果

    Returns:
        LLMAnalysisResult: LLM 智能分析结果
    """
    analyzer = LLMAnalyzer()
    return analyzer.analyze(technical_result)


if __name__ == "__main__":
    # 测试代码
    from dataclasses import dataclass

    @dataclass
    class MockTrend:
        direction: str = "up"
        strength: float = 0.75
        description: str = "上涨趋势 (MA5=$2050.20 > MA20=$2020.50)"

    @dataclass
    class MockVolatility:
        daily_volatility: float = 0.008
        annualized_volatility: float = 0.15
        volatility_level: str = "medium"
        avg_daily_range: float = 1.2

    @dataclass
    class MockSR:
        support_levels: list = None
        resistance_levels: list = None

        def __post_init__(self):
            if self.support_levels is None:
                self.support_levels = [2000, 1980, 1950]
            if self.resistance_levels is None:
                self.resistance_levels = [2100, 2150, 2200]

    @dataclass
    class MockResult:
        current_price: float = 2050.0
        price_change_1d: float = 0.5
        price_change_7d: float = 1.2
        price_change_30d: float = 3.5
        trend: MockTrend = None
        volatility: MockVolatility = None
        support_resistance: MockSR = None
        ma_analysis: dict = None
        rsi: float = 58.0
        recommendation: str = "买入 - 上涨趋势"

        def __post_init__(self):
            if self.trend is None:
                self.trend = MockTrend()
            if self.volatility is None:
                self.volatility = MockVolatility()
            if self.support_resistance is None:
                self.support_resistance = MockSR()
            if self.ma_analysis is None:
                self.ma_analysis = {"MA5": 2050.20, "MA10": 2035.50, "MA20": 2020.50}

    mock_result = MockResult()
    llm_result = analyze_with_llm(mock_result)

    print("=" * 50)
    print("LLM 智能分析结果")
    print("=" * 50)
    print(f"\n【市场情绪】\n{llm_result.market_sentiment}\n")
    print(f"【风险评估】\n{llm_result.risk_assessment}\n")
    print(f"【操作建议】\n{llm_result.action_advice}\n")
    print(f"【未来预测】\n{llm_result.future_prediction}\n")
