# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Pure Python CLI application — no Docker, no database, no web server. All external dependencies are cloud APIs (DashScope for LLM, Yahoo Finance for market data via `yfinance`).

### Services

| Service | How to run | Notes |
|---|---|---|
| Multi-agent analysis | `python3 scripts/run_agents.py` | Interactive CLI; requires `DASHSCOPE_API_KEY`. Prompts for commodity selection, then runs 8 agents in 3 rounds. |
| DL data download | `python3 dl/download_history.py --commodity gold` | Downloads OHLCV from Yahoo Finance to `data/`. No API key needed. |
| DL training | `python3 dl/trainer.py --commodity gold` | Trains LSTM model. Requires downloaded CSV in `data/`. |
| Quant analysis (standalone test) | See `agents/agent_quant.py` | Fetches live data from yfinance and computes technical indicators. No LLM API key needed. |

### Key caveats

- The main entry point (`scripts/run_agents.py`) is **interactive** — it reads from stdin for commodity selection and question input. When automating, pipe input: `echo -e "1\n" | python3 scripts/run_agents.py`.
- `DASHSCOPE_API_KEY` environment variable is **required** for any LLM-based agent to function. Without it, `run_agents.py` exits immediately with an error message. Set it in a `.env` file at the project root (auto-loaded via `python-dotenv`).
- The `python` command is not available; use `python3` instead.
- No linting or automated test suite is configured in this repository.
- The DL predictor agent is disabled in `run_agents.py` (commented out) due to model underperformance — the LSTM consistently goes short in bull markets (see backtest: -38% vs buy-and-hold +53%).
- RAG knowledge base is only pre-built for gold. The `rag/` directory builders require `DASHSCOPE_API_KEY`.
- The `QuantEngineerAgent` does **not** call the LLM — it fetches live data via yfinance and returns computed indicators directly. Useful for testing without API keys.
- Historical CSV data is stored in `data/` (git-ignored). Run `python3 dl/download_history.py --commodity <key>` to populate before training or backtesting.

### Commands reference

See `CLAUDE.md` for the full command reference.
