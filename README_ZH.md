# 多品种期货智能分析系统

基于多智能体协作的期货价格分析与预测系统，集成技术分析、深度学习预测、多模态 RAG 知识库和大语言模型智能分析能力。

> For the English version, see [README_EN.md](./README_EN.md)

---

## 项目特点

- **多品种支持** — 支持 11 种期货品种全流程分析：黄金、白银、铂金、钯金、WTI 原油、铜、铝、天然气、玉米、小麦、大豆；启动时通过交互菜单选品种
- **8 位专家圆桌** — 技术面研究员、宏观基本面分析师、量化工程师、算法预测师、跨市场联动分析师、情绪分析师、首席策略官、独立风控合规官
- **LSTM + Transformer 双模型预测** — 7 维技术指标特征工程（RSI、MACD、布林带、成交量）+ 内置回测引擎（累计收益、最大回撤、夏普比率、胜率）
- **多模态 RAG 知识库** — 基于 Faiss 和 DashScope `qwen3-vl-embedding` 的 K 线形态检索；黄金已内置多年周线快照库
- **并行执行架构** — 第一轮 6 位专家通过 `ThreadPoolExecutor` 并发执行，三轮设计分离分析、综合、审核职责
- **轻量级 LLM Agent 基类** — 直接调用 DashScope OpenAI 兼容接口，无 agentscope 依赖
- **独立风控审核** — 风控合规官在所有分析完成后独立审查，输出 🟢通过 / 🟡警告 / 🔴否决

---

## 架构设计

### 系统总览

```
┌──────────────────────────────────────────────────────────┐
│                       用户交互层                          │
│              品种选择  ·  自然语言问题输入                 │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                       Agent 层                            │
│                                                           │
│  第一轮 — 6 位专家 并行 执行                               │
│  ┌────────┬────────┬──────┬──────┬───────┬──────────┐    │
│  │技术面  │宏观基  │量化  │算法  │跨市场 │情绪      │    │
│  │研究员  │本面    │工程师│预测师│联动   │分析师    │    │
│  └────────┴────────┴──────┴──────┴───────┴──────────┘    │
│                          │                                │
│  第二轮 — 首席策略官 汇总所有报告                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                   首席策略官                         │ │
│  │        操作方向 · 分仓建议 · 止盈止损                │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                          │                                │
│  第三轮 — 风控合规官 独立审核                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │        风控合规官  🟢通过 / 🟡警告 / 🔴否决          │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                       数据层                              │
│  yfinance 实时行情 │ Faiss RAG │ DL 模型 │ 联网搜索       │
└──────────────────────────────────────────────────────────┘
```

### Agent 设计 — LLMAgent 基类

所有 8 个 Agent 均继承 `agents/base_agent.py` 中的 `LLMAgent`：

```
LLMAgent
├── __init__(name, sys_prompt, model_name, api_key, commodity, ...)
│     commodity = {"key": "gold", "symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"}
│     每个 Agent 从构造时即感知当前分析品种。
│
├── reply(msg) → Msg
│     默认行为：拼接 sys_prompt → 调用 DashScope OpenAI 兼容 API → 返回 Msg
│     子类通过 override reply() 在 LLM 调用前注入实时数据。
│
└── __call__(msg) → Msg   (reply 的别名，使 Agent 可直接被调用)
```

需要实时数据的 Agent 遵循 **获取 → 注入 → LLM** 模式：

```
CrossMarketAgent.reply()
  1. 从 yfinance 获取 DXY / TNX / VIX / CL=F / {commodity.symbol} 最新行情
  2. 将结构化数据格式化为纯文本数据块
  3. 拼接到用户消息前
  4. 调用 super().reply() → LLM 基于注入数据进行分析
```

### RAG 架构

```
构建阶段（只需运行一次）              查询阶段（每次分析）
────────────────────────            ─────────────────────────
周线 K 线图 PNG                      用户问题
  + LLM 生成的新闻摘要                │
  │                                  ▼
  ▼                            向量化 query
DashScope qwen3-vl-embedding         │ (qwen3-vl-embedding)
  │                                  ▼
  ▼                            Faiss 余弦相似度检索
Faiss 索引 ←─────────────────── Top-k 相关片段
(存储在 rag/)                         │
                                      ▼
                                注入到 prompt
                                      │
                                      ▼
                              技术面研究员 → LLM 分析
```

两个知识库：
- **周线快照库** — 约 5 年周线 K 线图 + LLM 生成的市场摘要（`rag/build_weekly.py`）
- **重大事件库** — 7 个重大历史行情事件，联网搜索自动扩写（`rag/build_events.py`）

### 深度学习流水线

```
历史 CSV（OHLCV）
        │
        ▼
  compute_features()          每日 7 维特征：
  ─────────────────           Close、RSI(14)、MACD、MACD_Signal、
                              BB_Upper、BB_Lower、Volume
        │
        ▼
  MinMaxScaler                按列独立归一化至 [0, 1]
        │
        ▼
  滑动窗口                    seq_length = 60 天 → 预测第 61 天收盘价
        │
        ├──► PriceLSTM        2 层 LSTM，hidden=128，dropout=0.2
        │                     FC: 128→64→1
        │
        └──► PriceTransformer input_proj → PositionalEncoding
                              → 3× TransformerEncoderLayer（d_model=64，nhead=4）
                              → GlobalAvgPool → LayerNorm → 64→1
        │
        ▼
  回测引擎                    滑动窗口模拟交易
                              指标：总收益、年化收益、最大回撤、夏普比率、胜率（多空）
```

---

## 项目结构

```
├── agents/
│   ├── base_agent.py             # LLMAgent 基类（品种感知）
│   ├── agent_analyst.py          # 技术面研究员（RAG K线形态匹配）
│   ├── agent_macro.py            # 宏观基本面分析师（收益率曲线 + 联网搜索）
│   ├── agent_quant.py            # 量化工程师（RSI/均线/波动率/支撑阻力位）
│   ├── agent_dl.py               # 算法预测师（LSTM + Transformer）
│   ├── agent_cross_market.py     # 跨市场联动分析师（DXY/美债/VIX/原油）
│   ├── agent_sentiment.py        # 情绪分析师（社交媒体联网搜索）
│   ├── agent_risk.py             # 风控合规官（独立合规审查）
│   └── agent_manager.py          # 首席策略官（综合决策）
├── core/
│   ├── config.py                 # 配置：API Key、COMMODITY_SYMBOLS、ANALYZABLE_COMMODITIES
│   └── analyzer.py               # 技术指标计算引擎（RSI、均线、支阻位、波动率）
├── dl/
│   ├── trainer.py                # 模型训练（--commodity / --model / --epochs）
│   ├── transformer_model.py      # PriceTransformer Encoder 架构
│   ├── predictor.py              # 推理：按品种 key 查找 CSV + 权重文件
│   ├── backtester.py             # 滑动窗口回测引擎
│   ├── download_history.py       # 历史数据下载（--commodity）
│   └── commodity_downloader.py   # 批量下载器（含相关性计算工具）
├── rag/
│   ├── engine.py                 # 多模态 RAG 引擎（Faiss + DashScope Embedding）
│   ├── build_weekly.py           # 构建周度图文知识库（异步并发）
│   └── build_events.py           # 构建重大历史事件知识库
├── scripts/
│   └── run_agents.py             # 主入口
├── data/                         # 本地 CSV 行情数据（git 忽略）
├── models/                       # 模型权重 + Scaler（git 忽略）
├── reports/                      # 分析报告输出（git 忽略）
├── .env                          # API Key 配置（git 忽略）
└── requirements.txt
```

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/tutao0123/gold-analyzer.git
cd gold-analyzer

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件（已被 git 忽略）：

```bash
# 阿里云百炼（必需，用于所有 LLM 调用和 RAG Embedding）
DASHSCOPE_API_KEY=your_dashscope_api_key

# GoldAPI.io（可选，用于获取更精准的实时金价备用数据源）
GOLD_API_KEY=your_goldapi_key
```

`run_agents.py` 启动时自动调用 `load_dotenv()`，有 `.env` 文件即可，无需手动 `export`。

### 3. 运行多智能体分析

```bash
python scripts/run_agents.py
```

启动时显示品种选择菜单：

```
可分析品种：
   1. 黄金期货 (GC=F)
   2. 白银期货 (SI=F)
   ...
  11. 大豆期货 (ZS=F)

请选择品种（默认1，直接回车选黄金）: >
```

选品种后，8 位专家并行运行，分析报告自动保存至 `reports/analysis_YYYYMMDD_HHMMSS.txt`。

---

## 深度学习流水线（可选）

DL 预测模块与主流程完全解耦。若所选品种没有训练好的模型，算法预测师会输出提示并跳过，其余 Agent 正常运行。

```bash
# 第一步：下载历史 OHLCV 数据
python dl/download_history.py --commodity gold
python dl/download_history.py --commodity silver

# 第二步：训练模型（默认 LSTM；加 --model transformer 训练 Transformer）
python dl/trainer.py --commodity gold
python dl/trainer.py --commodity silver --model transformer --epochs 50

# 第三步：运行回测评估模型绩效
python dl/backtester.py
```

模型文件保存至 `models/`：
- `{commodity}_lstm_weights.pth`
- `{commodity}_scaler.pkl`
- `{commodity}_transformer_weights.pth`（如训练了 Transformer）

---

## 构建 RAG 知识库（可选，仅黄金）

```bash
# 前提：黄金历史 CSV 已存在
python dl/download_history.py --commodity gold

# 构建周度 K 线图文知识库（约 5 年 / ~260 次 API 调用，异步并发）
python rag/build_weekly.py

# 构建重大历史事件案例库（7 个事件，联网搜索自动扩写）
python rag/build_events.py
```

---

## 支持品种一览

| Key | 代码 | 名称 | 单位 |
|-----|------|------|------|
| `gold` | GC=F | 黄金期货 | USD/oz |
| `silver` | SI=F | 白银期货 | USD/oz |
| `platinum` | PL=F | 铂金期货 | USD/oz |
| `palladium` | PA=F | 钯金期货 | USD/oz |
| `wti_oil` | CL=F | WTI 原油期货 | USD/桶 |
| `copper` | HG=F | 铜期货 | USD/磅 |
| `aluminum` | ALI=F | 铝期货 | USD/磅 |
| `natural_gas` | NG=F | 天然气期货 | USD/MMBtu |
| `corn` | ZC=F | 玉米期货 | USD/蒲式耳 |
| `wheat` | ZW=F | 小麦期货 | USD/蒲式耳 |
| `soybean` | ZS=F | 大豆期货 | USD/蒲式耳 |

---

## 主要依赖

| 包 | 用途 |
|----|------|
| `torch` | LSTM + Transformer 模型训练与推理 |
| `yfinance` | 实时行情与历史数据获取 |
| `faiss-cpu` | 向量相似性检索（RAG） |
| `dashscope` | 阿里云百炼 LLM + 多模态 Embedding API |
| `openai` | DashScope OpenAI 兼容客户端 |
| `mplfinance` | K 线图绘制 |
| `pandas / numpy` | 数据处理 |
| `scikit-learn` | MinMaxScaler 特征归一化 |
| `python-dotenv` | `.env` 文件自动加载 |

---

## 参与贡献

欢迎提交 Pull Request。对于重大改动，请先开 Issue 说明你的想法。

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/amazing-feature`）
3. 提交改动
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 发起 Pull Request

---

## 免责声明

本项目仅供**学习和研究使用，不构成任何投资建议**。金融市场存在风险，请独立判断，审慎决策。

## License

MIT License
