# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent futures analysis system for 11 commodity types (gold, silver, platinum, palladium, WTI oil, copper, aluminum, natural gas, corn, wheat, soybean). Uses 8 specialized AI agents coordinated through a 3-round architecture: parallel specialist analysis → chief strategist synthesis → independent risk review.

## Commands

```bash
# Run the main multi-agent analysis (prompts for commodity selection)
python scripts/run_agents.py

# Deep learning pipeline (optional, per-commodity)
python dl/download_history.py --commodity gold    # Download historical OHLCV data
python dl/trainer.py --commodity gold             # Train LSTM model (default)
python dl/trainer.py --commodity gold --model transformer --epochs 50  # Train Transformer variant
python dl/backtester.py                           # Run backtesting

# RAG knowledge base (optional, gold only currently)
python rag/build_weekly.py    # Build weekly chart-text knowledge base
python rag/build_events.py    # Build major historical events knowledge base
```

## Environment Setup

Create a `.env` file in the project root (git-ignored):
```
DASHSCOPE_API_KEY=your_dashscope_api_key    # Required for LLM + RAG embedding
GOLD_API_KEY=your_goldapi_key               # Optional, backup real-time gold price
```

## Architecture

### Agent System (3-Round Design)

**Round 1**: 5-6 specialists execute in parallel via `ThreadPoolExecutor`:
- `RAGAnalystAgent` - Technical analysis with K-line pattern matching via RAG
- `MacroAnalystAgent` - Macro fundamentals with web search (`enable_search=True`)
- `QuantEngineerAgent` - Real-time technical indicators (RSI, MA, support/resistance)
- `CrossMarketAgent` - DXY, Treasury yields, VIX, crude oil correlation analysis
- `SentimentAgent` - Social media sentiment via web search

**Round 2**: `PortfolioManagerAgent` synthesizes all specialist reports into a final decision.

**Round 3**: `RiskControlAgent` independently reviews the decision (🟢pass / 🟡warning / 🔴veto).

### LLMAgent Base Class

All agents inherit from `agents/base_agent.py:LLMAgent`:
- Constructor receives `commodity` dict with `{"key": "gold", "symbol": "GC=F", "name": "黄金期货", "unit": "USD/oz"}`
- Override `reply()` to inject real-time data before calling `super().reply()` for LLM inference
- Uses DashScope OpenAI-compatible API (`qwen-plus` model by default)

### Data-Injecting Agents Pattern

Agents that need real data follow this pattern in `reply()`:
1. Fetch data (yfinance, web search, local CSV)
2. Format as structured text
3. Inject into prompt or return directly as `Msg` content

Example: `QuantEngineerAgent` fetches 40 days of OHLCV via yfinance, computes indicators via `core/analyzer.py:PriceAnalyzer`, and returns formatted metrics without a secondary LLM call.

## Key Directories

- `agents/` - 8 specialized agent implementations
- `core/` - `config.py` (API keys, commodity symbols) and `analyzer.py` (technical indicators)
- `dl/` - LSTM/Transformer price prediction models
- `rag/` - Faiss + DashScope multimodal RAG engine
- `data/` - Local CSV historical data (git-ignored)
- `models/` - Trained model weights and scalers (git-ignored)
- `reports/` - Generated analysis reports (git-ignored)

## Commodity Configuration

Supported commodities are defined in `core/config.py`:
- `COMMODITY_SYMBOLS` - All tradeable symbols (includes indices, forex, bonds)
- `ANALYZABLE_COMMODITIES` - Subset suitable for full pipeline (excludes indices/forex)

## Dependencies

Core: `torch`, `yfinance`, `faiss-cpu`, `dashscope`, `openai`, `pandas`, `numpy`, `scikit-learn`, `python-dotenv`

## Notes

- The DL predictor agent is currently disabled in `run_agents.py` due to model underperformance in bull markets
- RAG knowledge base only has pre-built data for gold; other commodities would need `rag/build_*.py` adaptation
- Reports are saved to `reports/analysis_YYYYMMDD_HHMMSS.txt`