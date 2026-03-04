"""
黄金价格分析器配置文件
"""

import os
from datetime import datetime

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 报告输出目录
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# API 配置
# 使用 GoldAPI.io
GOLD_API_KEY = os.getenv("GOLD_API_KEY", "goldapi-1ajarprsmma81n7v-io")
GOLD_API_URL = "https://www.goldapi.io/api/XAU/USD"

# 阿里云百炼大模型 API 配置
# 从环境变量读取 DASHSCOPE_API_KEY
BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"

# 备用 API (无需 API Key)
BACKUP_APIS = [
    "https://api.metals.live/v1/spot",
    "https://api.exchangerate-api.com/v4/latest/USD",
]

# 分析配置
ANALYSIS_DAYS = 30  # 分析最近30天的数据
PRICE_THRESHOLD = 0.02  # 价格波动阈值 (2%)

# 报告配置
REPORT_TITLE = "黄金价格分析报告"
REPORT_DATE_FORMAT = "%Y年%m月%d日 %H:%M"

# 日志配置
LOG_FILE = os.path.join(BASE_DIR, "gold_analyzer.log")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# 商品期货配置 (Yahoo Finance 代码)
COMMODITY_SYMBOLS = {
    # 贵金属
    "gold": {"symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"},
    "silver": {"symbol": "SI=F", "name": "白银期货", "unit": "USD/oz"},
    "platinum": {"symbol": "PL=F", "name": "铂金期货", "unit": "USD/oz"},
    "palladium": {"symbol": "PA=F", "name": "钯金期货", "unit": "USD/oz"},
    # 原油
    "wti_oil": {"symbol": "CL=F", "name": "WTI原油期货", "unit": "USD/桶"},
    "brent_oil": {"symbol": "BZ=F", "name": "布伦特原油期货", "unit": "USD/桶"},
    # 能源
    "natural_gas": {"symbol": "NG=F", "name": "天然气期货", "unit": "USD/MMBtu"},
    # 农产品
    "corn": {"symbol": "ZC=F", "name": "玉米期货", "unit": "USD/蒲式耳"},
    "wheat": {"symbol": "ZW=F", "name": "小麦期货", "unit": "USD/蒲式耳"},
    "soybean": {"symbol": "ZS=F", "name": "大豆期货", "unit": "USD/蒲式耳"},
    # 指数
    "sp500": {"symbol": "^GSPC", "name": "标普500指数", "unit": "points"},
    "nasdaq": {"symbol": "^IXIC", "name": "纳斯达克指数", "unit": "points"},
    "vix": {"symbol": "^VIX", "name": "波动率指数", "unit": "points"},
    # 外汇
    "dxy": {"symbol": "DX-Y.NYB", "name": "美元指数", "unit": "index"},
    # 债券收益率
    "tnx": {"symbol": "^TNX", "name": "10年期美债收益率", "unit": "%"},
}
