# Multi-Commodity Futures Analyzer · 多品种期货智能分析系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange" />
  <img src="https://img.shields.io/badge/LLM-DashScope%20qwen--plus-purple" />
  <img src="https://img.shields.io/badge/RAG-Faiss%20%2B%20Multimodal-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

A multi-agent collaborative system for futures price analysis and prediction — powered by LLM, dual deep learning models, and multimodal RAG.

---

**Documentation / 文档**

- [English README](./README_EN.md)
- [中文文档](./README_ZH.md)

---

## Overview

8 AI agents collaborate across 3 rounds to analyse any of 11 futures commodities (gold, silver, copper, crude oil, and more):

| Agent | Role |
|-------|------|
| Technical Analyst | RAG-based candlestick pattern retrieval |
| Macro Analyst | Live yield curve + web search |
| Quant Engineer | RSI, MA, volatility, support/resistance |
| DL Predictor | LSTM + Transformer dual-model forecasting |
| Cross-market Analyst | DXY / Treasuries / VIX / crude correlation |
| Sentiment Analyst | Social media sentiment via web search |
| Chief Strategist | Synthesis → final trade recommendation |
| Risk Officer | Independent compliance review (🟢/🟡/🔴) |

## Quick Start

```bash
git clone https://github.com/tutao0123/gold-analyzer.git
cd gold-analyzer
pip install -r requirements.txt

cp .env .env.local          # fill in your API key
export DASHSCOPE_API_KEY="your_key"

python scripts/run_agents.py
```

## License

MIT License
