# 多品种期货智能分析系统

基于多智能体协作的期货价格分析与预测系统，集成技术分析、深度学习预测、多模态 RAG 知识库和大语言模型智能分析能力。

> For the English version, see [README_EN.md](./README_EN.md)

---

## 项目特点

- **多品种支持** — 支持 11 种期货品种全流程分析：黄金、白银、铂金、钯金、WTI 原油、铜、铝、天然气、玉米、小麦、大豆；启动时通过交互菜单选品种
- **8 位专家圆桌** — 技术面研究员、宏观基本面分析师、量化工程师、算法预测师、跨市场联动分析师、情绪分析师、首席策略官、独立风控合规官
- **LSTM + Transformer 双模型预测** — 滑动窗口时序预测 + 内置回测引擎（累计收益、最大回撤、夏普比率、胜率）
- **多模态 RAG 知识库** — 基于 Faiss 和 DashScope `qwen3-vl-embedding` 的 K 线形态检索；黄金已内置多年周线快照库
- **并行执行架构** — 第一轮 6 位专家通过 `ThreadPoolExecutor` 并发执行，显著缩短总耗时
- **轻量级 LLM Agent 基类** — 直接调用 DashScope OpenAI 兼容接口，无 agentscope 依赖
- **独立风控审核** — 风控合规官在所有分析完成后独立审查，输出 🟢通过 / 🟡警告 / 🔴否决

---

## 项目结构

```
├── agents/                       # 智能体模块
│   ├── base_agent.py             # LLMAgent 基类（支持 commodity 字段）
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
│   ├── analyzer.py               # 技术指标计算引擎
│   ├── data_fetcher.py           # 黄金实时数据获取器
│   └── report_generator.py       # 报告格式化
├── dl/
│   ├── trainer.py                # 模型训练（--commodity / --model / --epochs）
│   ├── transformer_model.py      # Transformer Encoder 架构
│   ├── predictor.py              # 推理：按品种 key 查找 CSV + 权重文件
│   ├── backtester.py             # 滑动窗口回测引擎
│   ├── download_history.py       # 历史数据下载（--commodity）
│   └── commodity_downloader.py   # 通用批量下载器（含相关性计算工具）
├── rag/
│   ├── engine.py                 # 多模态 RAG 引擎（Faiss + DashScope Embedding）
│   ├── build_weekly.py           # 构建周度图文知识库（异步并发）
│   └── build_events.py           # 构建历史重大事件知识库
├── scripts/
│   └── run_agents.py             # 主入口
├── data/                         # 本地 CSV 行情数据（git 忽略）
├── models/                       # 模型权重 + Scaler（git 忽略）
├── reports/                      # 分析报告输出（git 忽略）
├── .env                          # API Key 配置（git 忽略，参见示例注释）
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入真实 Key：

```bash
# 阿里云百炼（必需，用于所有 LLM 调用和 RAG Embedding）
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# GoldAPI.io（可选，用于获取更精准的实时金价备用数据源）
export GOLD_API_KEY="your_goldapi_key"
```

### 3. 运行多智能体分析

```bash
python scripts/run_agents.py
```

启动时显示品种选择菜单：

```
可分析品种：
   1. 黄金期货 (GC=F)
   2. 白银期货 (SI=F)
   3. 铂金期货 (PL=F)
   ...
  11. 大豆期货 (ZS=F)

请选择品种（默认1，直接回车选黄金）: >
```

选品种后，8 位专家并行运行，分析报告自动保存至 `reports/`。

---

## 深度学习流水线（可选）

DL 预测模块与主流程解耦。若所选品种没有训练好的模型，算法预测师会输出提示并跳过，其余 Agent 正常运行。

```bash
# 第一步：下载历史数据
python dl/download_history.py --commodity gold     # 默认黄金
python dl/download_history.py --commodity silver   # 白银

# 第二步：训练 LSTM 模型（默认 30 轮）
python dl/trainer.py --commodity gold
python dl/trainer.py --commodity silver --epochs 50

# 第三步：运行回测评估模型绩效
python dl/backtester.py
```

模型文件保存至 `models/`：
- `{commodity}_lstm_weights.pth`
- `{commodity}_scaler.pkl`
- `{commodity}_transformer_weights.pth`（如训练了 Transformer）

黄金保持原有文件名（`gold_lstm_weights.pth` / `scaler.pkl`）向后兼容。

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

## 智能体协作流程

```
用户问题
    │
    ├──► 技术面研究员     （RAG 检索历史 K 线形态）    ─┐
    ├──► 宏观基本面分析师  （收益率曲线 + 联网搜索）     │  并行
    ├──► 量化工程师       （RSI / 均线 / 波动率 / 支阻） │
    ├──► 算法预测师       （LSTM + Transformer 双模型）  │
    ├──► 跨市场联动分析师  （DXY / 美债 / VIX / 原油）   │
    └──► 情绪分析师       （社交媒体联网搜索）          ─┘
                │
                ▼
         首席策略官  （汇总 → 最终交易建议）
                │
                ▼
         风控合规官  （独立审核：🟢通过 / 🟡警告 / 🔴否决）
                │
                ▼
         报告保存至 reports/analysis_YYYYMMDD_HHMMSS.txt
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

---

## 免责声明

本项目仅供学习和研究使用，**不构成任何投资建议**。金融市场存在风险，请独立判断，审慎决策。

## License

MIT License
