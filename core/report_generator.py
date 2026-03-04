"""
HTML 报表生成模块
生成包含图表和AI分析的交互式金价分析报告
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from core.config import REPORT_DIR, REPORT_TITLE, REPORT_DATE_FORMAT

logger = logging.getLogger(__name__)


class ReportGenerator:
    """HTML 报表生成器"""

    def __init__(self):
        self.report_file = f"{REPORT_DIR}/gold_analysis_report.html"

    def generate(self, analysis_result, historical_data: List[Dict], llm_result=None) -> str:
        """生成 HTML 报表"""
        
        # 准备图表数据
        dates = [d['date'] for d in historical_data]
        prices = [d['price'] for d in historical_data]
        
        # AI 分析板块（如果有）
        ai_section = ""
        if llm_result:
            ai_section = f"""
        <div class="card ai-analysis">
            <h2 style="margin-bottom: 20px;">🤖 AI 智能分析</h2>
            
            <div class="ai-section">
                <h3>📊 市场情绪</h3>
                <p>{llm_result.market_sentiment}</p>
            </div>
            
            <div class="ai-section">
                <h3>⚠️ 风险评估</h3>
                <p>{llm_result.risk_assessment}</p>
            </div>
            
            <div class="ai-section">
                <h3>💡 操作建议</h3>
                <p>{llm_result.action_advice}</p>
            </div>
            
            <div class="ai-section">
                <h3>🔮 未来预测</h3>
                <p>{llm_result.future_prediction}</p>
            </div>
            
            {f'<details style="margin-top: 15px; color: #666;"><summary>查看详细分析</summary><pre style="white-space: pre-wrap; margin-top: 10px;">{llm_result.detailed_analysis}</pre></details>' if llm_result.detailed_analysis else ''}
        </div>
"""
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{REPORT_TITLE}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            padding: 40px 0;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .ai-analysis {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        .ai-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
        }}
        .ai-section h3 {{
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .ai-section p {{
            line-height: 1.6;
            opacity: 0.95;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            margin-top: 8px;
        }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            color: white;
            opacity: 0.8;
            padding: 20px;
        }}
        details summary {{
            cursor: pointer;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }}
        details pre {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {REPORT_TITLE}</h1>
            <p>生成时间：{datetime.now().strftime(REPORT_DATE_FORMAT)}</p>
            {'<p style="margin-top: 10px; font-size: 0.9em;">🤖 Powered by 阿里云百炼大模型</p>' if llm_result else ''}
        </div>

        <div class="card">
            <h2 style="margin-bottom: 20px;">💰 核心指标</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">${analysis_result.current_price:.2f}</div>
                    <div class="metric-label">当前金价</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if analysis_result.price_change_1d >= 0 else 'negative'}">{analysis_result.price_change_1d:+.2f}%</div>
                    <div class="metric-label">1日涨跌</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if analysis_result.price_change_7d >= 0 else 'negative'}">{analysis_result.price_change_7d:+.2f}%</div>
                    <div class="metric-label">7日涨跌</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if analysis_result.price_change_30d >= 0 else 'negative'}">{analysis_result.price_change_30d:+.2f}%</div>
                    <div class="metric-label">30日涨跌</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-bottom: 20px;">📈 价格走势</h2>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-bottom: 20px;">🔍 技术分析</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{analysis_result.trend.direction}</div>
                    <div class="metric-label">趋势方向</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_result.trend.strength:.1%}</div>
                    <div class="metric-label">趋势强度</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_result.volatility.volatility_level}</div>
                    <div class="metric-label">波动率等级</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_result.rsi:.1f}</div>
                    <div class="metric-label">RSI指标</div>
                </div>
            </div>
            <p style="margin-top: 15px; color: #666;">
                <strong>趋势分析：</strong>{analysis_result.trend.description}
            </p>
        </div>

        {ai_section}

        <div class="recommendation">
            <h3>💡 投资建议</h3>
            <p style="font-size: 1.2em; margin-top: 10px;">{analysis_result.recommendation}</p>
            <p style="margin-top: 15px; opacity: 0.9;">{analysis_result.summary}</p>
        </div>

        <div class="footer">
            <p>Generated by Gold Price Analyzer Agent | Data source: Multiple APIs</p>
            {'<p style="margin-top: 5px;">AI Analysis powered by 阿里云百炼</p>' if llm_result else ''}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('priceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [{{
                    label: '黄金价格 (USD)',
                    data: {json.dumps(prices)},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: true }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{ display: true, text: '价格 (USD)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

        # 保存文件
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"报表已生成: {self.report_file}")
        return self.report_file
