"""
Gold price analyzer configuration file
"""

import os
from datetime import datetime

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Report output directory
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# API configuration
# Using GoldAPI.io
GOLD_API_KEY = os.getenv("GOLD_API_KEY", "")
GOLD_API_URL = "https://www.goldapi.io/api/XAU/USD"

# Alibaba Cloud Bailian LLM API configuration
# Read DASHSCOPE_API_KEY from environment variable
BAILIAN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"

# Backup APIs (no API Key required)
BACKUP_APIS = [
    "https://api.metals.live/v1/spot",
    "https://api.exchangerate-api.com/v4/latest/USD",
]

# Analysis configuration
ANALYSIS_DAYS = 30  # analyze the most recent 30 days of data
PRICE_THRESHOLD = 0.02  # price movement threshold (2%)

# Report configuration
REPORT_TITLE = "黄金价格分析报告"
REPORT_DATE_FORMAT = "%Y年%m月%d日 %H:%M"

# Logging configuration
LOG_FILE = os.path.join(BASE_DIR, "gold_analyzer.log")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Commodity futures configuration (Yahoo Finance symbols)
COMMODITY_SYMBOLS = {
    # Precious metals
    "gold": {"symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"},
    "silver": {"symbol": "SI=F", "name": "白银期货", "unit": "USD/oz"},
    "platinum": {"symbol": "PL=F", "name": "铂金期货", "unit": "USD/oz"},
    "palladium": {"symbol": "PA=F", "name": "钯金期货", "unit": "USD/oz"},
    # Crude oil
    "wti_oil": {"symbol": "CL=F", "name": "WTI原油期货", "unit": "USD/桶"},
    "brent_oil": {"symbol": "BZ=F", "name": "布伦特原油期货", "unit": "USD/桶"},
    # Energy
    "natural_gas": {"symbol": "NG=F", "name": "天然气期货", "unit": "USD/MMBtu"},
    # Industrial metals
    "copper":   {"symbol": "HG=F",   "name": "铜期货",   "unit": "USD/磅"},
    "aluminum": {"symbol": "ALI=F",  "name": "铝期货",   "unit": "USD/磅"},
    # Agricultural products
    "corn": {"symbol": "ZC=F", "name": "玉米期货", "unit": "USD/蒲式耳"},
    "wheat": {"symbol": "ZW=F", "name": "小麦期货", "unit": "USD/蒲式耳"},
    "soybean": {"symbol": "ZS=F", "name": "大豆期货", "unit": "USD/蒲式耳"},
    # Indices
    "sp500": {"symbol": "^GSPC", "name": "标普500指数", "unit": "points"},
    "nasdaq": {"symbol": "^IXIC", "name": "纳斯达克指数", "unit": "points"},
    "vix": {"symbol": "^VIX", "name": "波动率指数", "unit": "points"},
    # Forex
    "dxy": {"symbol": "DX-Y.NYB", "name": "美元指数", "unit": "index"},
    # Bond yields
    "tnx": {"symbol": "^TNX", "name": "10年期美债收益率", "unit": "%"},
}

# Commodities suitable for full-pipeline analysis and prediction (excludes indices/forex/bonds)
ANALYZABLE_COMMODITIES = [
    "gold", "silver", "platinum", "palladium",
    "wti_oil", "copper", "aluminum",
    "natural_gas", "corn", "wheat", "soybean",
]
