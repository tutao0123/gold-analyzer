# Multi-Commodity Futures Analyzer

A multi-agent collaborative system for futures price analysis and prediction, integrating technical analysis, deep learning forecasting, multimodal RAG knowledge retrieval, and large language model intelligence.

> 中文文档请见 [README_ZH.md](./README_ZH.md)

---

## Features

- **Multi-commodity support** — Analyse any of 11 futures: gold, silver, platinum, palladium, WTI crude, copper, aluminium, natural gas, corn, wheat, soybean; commodity is selected at startup via an interactive menu
- **8-agent round-table** — Specialists for technical analysis, macro fundamentals, quant indicators, deep learning prediction, cross-market correlation, market sentiment, chief strategy officer, and independent risk control
- **LSTM + Transformer dual-model prediction** — Sliding-window time-series prediction with a built-in backtesting engine (cumulative return, max drawdown, Sharpe ratio, win rate)
- **Multimodal RAG knowledge base** — Chart pattern retrieval backed by Faiss and DashScope `qwen3-vl-embedding`; gold has a pre-built weekly snapshot library
- **Parallel execution** — All round-1 agents run concurrently via `ThreadPoolExecutor`
- **Lightweight LLM agent base class** — Direct DashScope OpenAI-compatible API calls; no agentscope dependency
- **Independent risk review** — The risk officer audits the chief strategist's decision last, outputting 🟢 Pass / 🟡 Warning / 🔴 Veto

---

## Project Structure

```
├── agents/                       # Agent modules
│   ├── base_agent.py             # LLMAgent base class (commodity-aware)
│   ├── agent_analyst.py          # Technical analyst   (RAG chart pattern matching)
│   ├── agent_macro.py            # Macro analyst       (live yield curve + web search)
│   ├── agent_quant.py            # Quant engineer      (RSI, MA, volatility, S/R levels)
│   ├── agent_dl.py               # DL predictor        (LSTM + Transformer)
│   ├── agent_cross_market.py     # Cross-market analyst(DXY / Treasuries / VIX / crude)
│   ├── agent_sentiment.py        # Sentiment analyst   (social media web search)
│   ├── agent_risk.py             # Risk officer        (independent compliance review)
│   └── agent_manager.py          # Chief strategist    (synthesis & final decision)
├── core/
│   ├── config.py                 # Config: API keys, COMMODITY_SYMBOLS, ANALYZABLE_COMMODITIES
│   ├── analyzer.py               # Technical indicator engine
│   ├── data_fetcher.py           # Gold real-time data fetcher
│   └── report_generator.py       # Report formatter
├── dl/
│   ├── trainer.py                # Model training  (--commodity, --model, --epochs)
│   ├── transformer_model.py      # Transformer Encoder architecture
│   ├── predictor.py              # Inference: commodity-aware CSV + weight file lookup
│   ├── backtester.py             # Sliding-window backtester
│   ├── download_history.py       # Historical data downloader (--commodity)
│   └── commodity_downloader.py   # Generic bulk downloader with correlation tools
├── rag/
│   ├── engine.py                 # Multimodal RAG engine (Faiss + DashScope embeddings)
│   ├── build_weekly.py           # Build weekly chart + news knowledge base (async)
│   └── build_events.py           # Build historical event knowledge base
├── scripts/
│   └── run_agents.py             # Main entry point
├── data/                         # Local CSV price data (git-ignored)
├── models/                       # Model weights + scalers (git-ignored)
├── reports/                      # Analysis report output (git-ignored)
├── .env                          # API keys (git-ignored; see .env.example)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Set API keys

Copy `.env` and fill in your keys:

```bash
# Alibaba Cloud Bailian (required — used for all LLM calls and RAG embeddings)
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# GoldAPI.io (optional — real-time gold spot price backup)
export GOLD_API_KEY="your_goldapi_key"
```

### 3. Run the multi-agent analysis

```bash
python scripts/run_agents.py
```

On startup you will see a commodity selection menu:

```
Available commodities:
   1. 黄金期货 (GC=F)
   2. 白银期货 (SI=F)
   3. 铂金期货 (PL=F)
   ...
  11. 大豆期货 (ZS=F)

Select commodity (default 1, press Enter for gold): >
```

All 8 agents then run in parallel and produce a consolidated report saved to `reports/`.

---

## Deep Learning Pipeline (optional)

The DL prediction module is independent of the main agent run. If no trained model exists for the selected commodity, the DL agent reports a warning and the rest of the analysis continues normally.

```bash
# Step 1: download historical data
python dl/download_history.py --commodity gold     # default
python dl/download_history.py --commodity silver

# Step 2: train LSTM model (30 epochs by default)
python dl/trainer.py --commodity gold
python dl/trainer.py --commodity silver --epochs 50

# Step 3: run backtest to evaluate model performance
python dl/backtester.py
```

Model files are saved to `models/`:
- `{commodity}_lstm_weights.pth`
- `{commodity}_scaler.pkl`
- `{commodity}_transformer_weights.pth` (if Transformer is trained)

Gold keeps its original filename (`gold_lstm_weights.pth` / `scaler.pkl`) for backward compatibility.

---

## Building the RAG Knowledge Base (optional, gold only)

```bash
# Prerequisite: gold historical CSV must already exist
python dl/download_history.py --commodity gold

# Build weekly chart + LLM news snapshots (~5 years, async, ~260 API calls)
python rag/build_weekly.py

# Build major historical event cases (7 events, web-search powered)
python rag/build_events.py
```

---

## Agent Collaboration Flow

```
User question
    │
    ├──► Technical Analyst     (RAG chart pattern retrieval)      ─┐
    ├──► Macro Analyst         (live yield curve + web search)     │  parallel
    ├──► Quant Engineer        (RSI / MA / volatility / S&R)       │
    ├──► DL Predictor          (LSTM + Transformer dual model)      │
    ├──► Cross-market Analyst  (DXY / Treasuries / VIX / crude)    │
    └──► Sentiment Analyst     (social media web search)          ─┘
                │
                ▼
         Chief Strategist  (synthesis → final trade recommendation)
                │
                ▼
         Risk Officer      (independent review: 🟢 Pass / 🟡 Warning / 🔴 Veto)
                │
                ▼
         Report saved to reports/analysis_YYYYMMDD_HHMMSS.txt
```

---

## Supported Commodities

| Key | Symbol | Name | Unit |
|-----|--------|------|------|
| `gold` | GC=F | Gold Futures | USD/oz |
| `silver` | SI=F | Silver Futures | USD/oz |
| `platinum` | PL=F | Platinum Futures | USD/oz |
| `palladium` | PA=F | Palladium Futures | USD/oz |
| `wti_oil` | CL=F | WTI Crude Oil | USD/bbl |
| `copper` | HG=F | Copper Futures | USD/lb |
| `aluminum` | ALI=F | Aluminium Futures | USD/lb |
| `natural_gas` | NG=F | Natural Gas | USD/MMBtu |
| `corn` | ZC=F | Corn Futures | USD/bu |
| `wheat` | ZW=F | Wheat Futures | USD/bu |
| `soybean` | ZS=F | Soybean Futures | USD/bu |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | LSTM + Transformer model training and inference |
| `yfinance` | Real-time and historical market data |
| `faiss-cpu` | Vector similarity search for RAG |
| `dashscope` | Alibaba Cloud Bailian LLM + multimodal embedding API |
| `openai` | OpenAI-compatible client for DashScope |
| `mplfinance` | Candlestick chart rendering |
| `pandas / numpy` | Data processing |
| `scikit-learn` | MinMaxScaler for feature normalisation |

---

## Disclaimer

This project is for **educational and research purposes only** and does not constitute investment advice. Financial markets carry risk — always make decisions independently and responsibly.

## License

MIT License
