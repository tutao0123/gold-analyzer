# Multi-Commodity Futures Analyzer

A multi-agent collaborative system for futures price analysis and prediction, integrating technical analysis, deep learning forecasting, multimodal RAG knowledge retrieval, and large language model intelligence.

> 中文文档请见 [README_ZH.md](./README_ZH.md)

---

## Features

- **Multi-commodity support** — Analyse any of 11 futures: gold, silver, platinum, palladium, WTI crude, copper, aluminium, natural gas, corn, wheat, soybean; commodity is selected at startup via an interactive menu
- **8-agent round-table** — Specialists for technical analysis, macro fundamentals, quant indicators, deep learning prediction, cross-market correlation, market sentiment, chief strategy officer, and independent risk control
- **LSTM + Transformer dual-model prediction** — 7-dimensional feature engineering (RSI, MACD, Bollinger Bands, Volume) with a built-in backtesting engine (cumulative return, max drawdown, Sharpe ratio, win rate)
- **Multimodal RAG knowledge base** — Chart pattern retrieval backed by Faiss and DashScope `qwen3-vl-embedding`; gold has a pre-built weekly snapshot library
- **Parallel execution** — All round-1 agents run concurrently via `ThreadPoolExecutor`; the 3-round design separates analysis, synthesis, and audit
- **Lightweight LLM agent base class** — Direct DashScope OpenAI-compatible API calls; no agentscope dependency
- **Independent risk review** — The risk officer audits the chief strategist's decision last, outputting 🟢 Pass / 🟡 Warning / 🔴 Veto

---

## Architecture Design

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                      User Interface                       │
│          Commodity selection · Natural-language question  │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                      Agent Layer                          │
│                                                           │
│  Round 1 — 6 specialists run in PARALLEL                  │
│  ┌──────────┬──────────┬────────┬──────┬───────┬──────┐  │
│  │Technical │  Macro   │ Quant  │  DL  │Cross- │Senti-│  │
│  │Analyst   │ Analyst  │Engineer│Pred. │market │ment  │  │
│  └──────────┴──────────┴────────┴──────┴───────┴──────┘  │
│                          │                                │
│  Round 2 — Chief Strategist synthesises all reports       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                  Chief Strategist                    │ │
│  │  direction (long/short/hold) · position · SL/TP     │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                          │                                │
│  Round 3 — Risk Officer independent audit                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │         Risk Officer  🟢 Pass / 🟡 Warn / 🔴 Veto   │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                      Data Layer                           │
│   yfinance    │   Faiss RAG    │  DL models  │ Web search │
│  (live prices)│(chart patterns)│(LSTM / TF)  │ (DashScope)│
└──────────────────────────────────────────────────────────┘
```

### Agent Design — LLMAgent Base Class

All 8 agents inherit from `LLMAgent` in `agents/base_agent.py`:

```
LLMAgent
├── __init__(name, sys_prompt, model_name, api_key, commodity, ...)
│     commodity = {"key": "gold", "symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"}
│     Every agent is commodity-aware from construction.
│
├── reply(msg) → Msg
│     Default: prepend sys_prompt → call DashScope OpenAI-compatible API → return Msg
│     Subclasses override reply() to inject real-time data before the LLM call.
│
└── __call__(msg) → Msg   (alias for reply)
```

Agents that need live data follow the **Fetch → Inject → LLM** pattern:

```
CrossMarketAgent.reply()
  1. Fetch DXY / TNX / VIX / CL=F / {commodity.symbol} from yfinance
  2. Build a structured data block as plain text
  3. Prepend it to the user message
  4. Call super().reply() → LLM analyses the injected data
```

### RAG Architecture

```
Build-time (run once)                Query-time (every analysis)
─────────────────────                ────────────────────────────
Weekly chart PNG                     User question
    + LLM news summary               │
    │                                ▼
    ▼                          Embed query
DashScope qwen3-vl-embedding         │ (qwen3-vl-embedding)
    │                                ▼
    ▼                          Faiss cosine search
Faiss index  ←────────────────── Top-k chunks
(stored in rag/)                     │
                                     ▼
                               Inject into prompt
                                     │
                                     ▼
                               TechnicalAnalystAgent → LLM
```

Two knowledge bases:
- **Weekly snapshots** — ~5 years of weekly candlestick charts + LLM-generated market summaries (`rag/build_weekly.py`)
- **Historical events** — 7 major market events with web-search-enriched case descriptions (`rag/build_events.py`)

### Deep Learning Pipeline

```
Historical CSV (OHLCV)
        │
        ▼
  compute_features()          7-dim feature vector per day:
  ─────────────────           Close, RSI(14), MACD, MACD_Signal,
                              BB_Upper, BB_Lower, Volume
        │
        ▼
  MinMaxScaler                per-feature normalisation → [0, 1]
        │
        ▼
  Sliding window              seq_length = 60 days → predict day 61
        │
        ├──► PriceLSTM        2-layer LSTM, hidden=128, dropout=0.2
        │                     FC: 128→64→1
        │
        └──► PriceTransformer input_proj → PositionalEncoding
                              → 3× TransformerEncoderLayer (d_model=64, nhead=4)
                              → GlobalAvgPool → LayerNorm → 64→1
        │
        ▼
  Backtest engine             sliding-window simulation on held-out data
                              metrics: total/annual return, max drawdown,
                                       Sharpe ratio, win rate (L/S)
```

---

## Project Structure

```
├── agents/
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
│   ├── config.py                 # API keys, COMMODITY_SYMBOLS, ANALYZABLE_COMMODITIES
│   └── analyzer.py               # Technical indicator engine (RSI, MA, S/R, volatility)
├── dl/
│   ├── trainer.py                # Model training  (--commodity, --model, --epochs)
│   ├── transformer_model.py      # PriceTransformer Encoder architecture
│   ├── predictor.py              # Inference: commodity-aware CSV + weight file lookup
│   ├── backtester.py             # Sliding-window backtester
│   ├── download_history.py       # Historical data downloader (--commodity)
│   └── commodity_downloader.py   # Bulk downloader with cross-asset correlation tools
├── rag/
│   ├── engine.py                 # Multimodal RAG engine (Faiss + DashScope embeddings)
│   ├── build_weekly.py           # Build weekly chart + news knowledge base (async)
│   └── build_events.py           # Build historical event knowledge base
├── scripts/
│   └── run_agents.py             # Main entry point
├── data/                         # Local CSV price data       (git-ignored)
├── models/                       # Model weights + scalers    (git-ignored)
├── reports/                      # Analysis report output     (git-ignored)
├── .env                          # API keys                   (git-ignored)
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/tutao0123/gold-analyzer.git
cd gold-analyzer

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root (it is git-ignored):

```bash
# Alibaba Cloud Bailian — required for all LLM calls and RAG embeddings
DASHSCOPE_API_KEY=your_dashscope_api_key

# GoldAPI.io — optional real-time gold spot price backup source
GOLD_API_KEY=your_goldapi_key
```

`run_agents.py` calls `load_dotenv()` automatically, so just having the `.env` file is enough.

### 3. Run the multi-agent analysis

```bash
python scripts/run_agents.py
```

On startup you will see a commodity selection menu:

```
Available commodities:
   1. 黄金期货 (GC=F)
   2. 白银期货 (SI=F)
   ...
  11. 大豆期货 (ZS=F)

Select commodity (default 1, press Enter for gold): >
```

All 8 agents then run and produce a report saved to `reports/analysis_YYYYMMDD_HHMMSS.txt`.

---

## Deep Learning Pipeline (optional)

The DL module is fully decoupled from the main agent run. If no trained model exists for the selected commodity, the DL agent reports a warning and the rest of the analysis continues normally.

```bash
# Step 1: download historical OHLCV data
python dl/download_history.py --commodity gold
python dl/download_history.py --commodity silver

# Step 2: train (LSTM by default; add --model transformer for Transformer)
python dl/trainer.py --commodity gold
python dl/trainer.py --commodity silver --model transformer --epochs 50

# Step 3: evaluate with the backtest engine
python dl/backtester.py
```

Model files are saved to `models/`:
- `{commodity}_lstm_weights.pth`
- `{commodity}_scaler.pkl`
- `{commodity}_transformer_weights.pth` (if Transformer is trained)

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
| `torch` | LSTM + Transformer training and inference |
| `yfinance` | Real-time and historical market data |
| `faiss-cpu` | Vector similarity search for RAG |
| `dashscope` | Alibaba Cloud Bailian LLM + multimodal embedding API |
| `openai` | OpenAI-compatible client for DashScope |
| `mplfinance` | Candlestick chart rendering |
| `pandas / numpy` | Data processing |
| `scikit-learn` | MinMaxScaler for feature normalisation |
| `python-dotenv` | `.env` file auto-loading |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Disclaimer

This project is for **educational and research purposes only** and does not constitute investment advice. Financial markets carry risk — always make decisions independently and responsibly.

## License

MIT License
