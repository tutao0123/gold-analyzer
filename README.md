# Gold Analyzer - 黄金价格智能分析系统

基于多智能体协作的黄金价格分析与预测系统，集成技术分析、深度学习预测、多模态RAG知识库和大语言模型智能分析能力。

## 项目特点

- **多数据源获取**: 支持从多个免费API源获取实时金价数据（GoldAPI、Metals.live、Yahoo Finance等）
- **技术指标分析**: 提供趋势分析、波动率计算、支撑阻力位识别、RSI指标等
- **深度学习预测**: 支持 LSTM 和 Transformer 双模型价格预测，并内置回测引擎
- **多智能体协作**: 使用AgentScope框架实现8个专业智能体协同分析
- **多模态RAG**: 基于Faiss和DashScope的图文知识库，支持K线图相似性检索
- **跨市场联动**: 追踪美元指数、美债收益率、VIX、原油等与黄金高度相关品种
- **情绪面分析**: 联网搜索社交媒体情绪，提供散户情绪温度计和反向指标
- **独立风控审核**: 对最终决策进行合规审查，输出通过/警告/否决结论
- **LLM智能分析**: 集成阿里云百炼大模型，提供专业的投资分析和建议

## 项目结构

```
gold_analyzer/
├── core/                    # 核心模块
│   ├── config.py           # 配置文件（含商品期货符号表）
│   ├── analyzer.py         # 技术分析器
│   ├── data_fetcher.py     # 数据获取器
│   ├── llm_analyzer.py     # LLM智能分析器
│   └── report_generator.py # 报告生成器
├── agents/                  # 智能体模块（8个专业智能体）
│   ├── agent_analyst.py    # 技术面研究员（RAG K线形态匹配）
│   ├── agent_macro.py      # 宏观基本面分析师（联网搜索）
│   ├── agent_quant.py      # 量化工程师（技术指标计算）
│   ├── agent_dl.py         # 算法预测师（LSTM+Transformer预测）
│   ├── agent_cross_market.py # 跨市场联动分析师（DXY/美债/VIX/原油）
│   ├── agent_sentiment.py  # 情绪分析师（社交媒体情绪面）
│   ├── agent_risk.py       # 风控合规官（独立风控审核）
│   └── agent_manager.py    # 首席策略官（综合决策）
├── dl/                      # 深度学习模块
│   ├── trainer.py          # LSTM模型训练
│   ├── transformer_model.py # Transformer预测模型
│   ├── predictor.py        # 双模型价格预测器
│   ├── backtester.py       # 滑动窗口回测引擎
│   ├── download_history.py # 黄金历史数据下载
│   └── commodity_downloader.py # 通用商品数据下载器
├── rag/                     # RAG知识库模块
│   ├── engine.py           # 多模态RAG引擎
│   ├── build_weekly.py     # 构建周度知识库
│   └── build_events.py     # 构建事件知识库
├── scripts/                 # 脚本
│   ├── run_agents.py       # 运行多智能体分析（主入口）
│   ├── run_legacy.py       # 运行传统分析流程
│   ├── test_llm.py         # 测试LLM功能
│   └── test_rag.py         # 测试RAG功能
├── data/                    # 数据存储
│   ├── gold_price_history.json
│   └── gc_f_full_history.csv
├── models/                  # 模型存储
│   ├── gold_lstm_weights.pth
│   └── scaler.pkl
├── kline_weekly_images/     # K线图存储
├── reports/                 # 分析报告输出
└── build_weekly_dataset.py  # 构建周度数据集
```

## 安装

### 环境要求

- Python 3.9+
- CUDA（可选，用于GPU加速训练）

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

```bash
# 阿里云百炼大模型 API Key (必需，用于LLM分析和RAG)
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# GoldAPI Key (可选，用于获取更准确的实时金价)
export GOLD_API_KEY="your_goldapi_key"
```

## 使用方法

### 1. 运行多智能体分析系统

```bash
python scripts/run_agents.py
```

系统将依次调用8个智能体进行协作分析：

| 智能体 | 角色 | 能力 |
|--------|------|------|
| 技术面研究员 | RAG K线形态匹配 | 检索历史相似形态并给出技术判断 |
| 宏观基本面分析师 | 联网搜索 | 搜索美联储、非农、地缘政治最新动态 |
| 量化工程师 | 指标计算 | RSI、波动率、支撑阻力位等客观数据 |
| 算法预测师 | LSTM+Transformer | 双模型预测未来价格走势及回测绩效 |
| 跨市场联动分析师 | 多品种关联 | 分析DXY、美债、VIX、原油对金价的影响 |
| 情绪分析师 | 舆情监控 | 社交媒体散户情绪温度计和反向指标 |
| 首席策略官 | 综合决策 | 汇总各方观点，给出最终交易建议 |
| 风控合规官 | 独立审核 | 对决策进行仓位/止损/逻辑一致性审查 |

### 2. 构建多模态知识库

```bash
# 构建周度K线图文知识库（需要DASHSCOPE_API_KEY）
python build_weekly_dataset.py
```

### 3. 训练深度学习模型

```bash
# 下载黄金历史数据
python dl/download_history.py

# 训练LSTM模型
python dl/trainer.py

# 运行回测（评估模型交易绩效）
python dl/backtester.py
```

### 4. 下载多品种商品数据

```bash
# 下载黄金、白银、原油、标普500等多品种历史数据
python dl/commodity_downloader.py
```

### 5. 单独运行技术分析

```python
from core.data_fetcher import GoldDataFetcher
from core.analyzer import GoldPriceAnalyzer

# 获取数据
fetcher = GoldDataFetcher()
data = fetcher.generate_historical_data(days=30)

# 技术分析
analyzer = GoldPriceAnalyzer(data)
result = analyzer.analyze()
print(result.summary)
```

## 核心功能

### 技术分析指标

| 指标 | 说明 |
|------|------|
| 趋势分析 | 基于短期/长期移动平均线判断趋势方向和强度 |
| 波动率 | 计算日波动率和年化波动率 |
| RSI | 相对强弱指标，判断超买超卖 |
| 支撑阻力位 | 使用局部极值法识别关键价位 |
| 移动平均线 | MA5/MA10/MA20/MA30 |

### 深度学习模型

| 模型 | 说明 |
|------|------|
| GoldLSTM | 双层LSTM网络，基于多特征时序预测次日收盘价 |
| GoldTransformer | Transformer Encoder架构，含正弦/余弦位置编码 |
| 回测引擎 | 滑动窗口模拟交易，输出累计收益、最大回撤、夏普比率、胜率 |

### 智能体协作流程

```
用户问题
    │
    ├──► 技术面研究员 (RAG检索历史K线形态)
    │
    ├──► 宏观基本面分析师 (联网搜索最新新闻)
    │
    ├──► 量化工程师 (计算技术指标)
    │
    ├──► 算法预测师 (LSTM+Transformer双模型预测)
    │
    ├──► 跨市场联动分析师 (DXY/美债/VIX/原油关联分析)
    │
    ├──► 情绪分析师 (社交媒体情绪面分析)
    │
    └──► 首席策略官 (综合决策)
            │
            ▼
        最终交易建议
            │
            ▼
        风控合规官 (独立审核: 🟢通过 / 🟡警告 / 🔴否决)
```

### 跨市场数据支持

| 品种 | 代码 | 说明 |
|------|------|------|
| 黄金 | GC=F | 主分析标的 |
| 白银/铂金/钯金 | SI=F / PL=F / PA=F | 贵金属板块 |
| WTI/布伦特原油 | CL=F / BZ=F | 能源品种 |
| 美元指数 | DX-Y.NYB | 与黄金负相关 |
| 10年期美债收益率 | ^TNX | 影响持金成本 |
| VIX恐慌指数 | ^VIX | 避险情绪指标 |
| 标普500/纳斯达克 | ^GSPC / ^IXIC | 风险资产对比 |
| 农产品 | ZC=F / ZW=F / ZS=F | 商品通胀参考 |

## API说明

### 数据获取 API

系统支持多个数据源，按优先级依次尝试：

1. **GoldAPI.io** - 专业贵金属API（需要API Key）
2. **Metals.live** - 免费现货金属数据
3. **Yahoo Finance** - 通过yfinance获取黄金期货数据
4. **网页爬虫** - 从Coinbase、Kitco等网站爬取

### 大模型 API

使用阿里云百炼(DashScope)提供的模型：

- **qwen-plus**: 通用对话模型，支持联网搜索（多智能体默认使用）
- **qwen-long**: 长文档理解模型
- **qwen3-vl-embedding**: 多模态Embedding模型（RAG向量化）

## 依赖库

- `agentscope`: 多智能体框架
- `torch`: 深度学习框架（LSTM + Transformer）
- `yfinance`: Yahoo Finance多品种数据获取
- `mplfinance`: K线图绘制
- `faiss-cpu`: 向量检索
- `dashscope`: 阿里云百炼SDK
- `openai`: OpenAI兼容API
- `requests`: HTTP请求
- `beautifulsoup4`: 网页解析
- `pandas/numpy`: 数据处理

## 注意事项

1. 本项目仅供学习和研究使用，不构成任何投资建议
2. 金融投资有风险，请谨慎决策
3. 使用前请确保已配置 `DASHSCOPE_API_KEY`
4. 部分API可能有请求频率限制

## License

MIT License
