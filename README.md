# Multi-Commodity Futures Analyzer · 多品种期货智能分析系统

A multi-agent collaborative system for futures price analysis and prediction, powered by LLM, deep learning, and multimodal RAG.

---

**Documentation / 文档**

- [English README](./README_EN.md)
- [中文文档](./README_ZH.md)

---

## Overview

8 AI agents collaborate in parallel to analyse any of 11 futures commodities (gold, silver, copper, crude oil, and more), each covering a specialised domain:

| Agent | Role |
|-------|------|
| Technical Analyst | RAG-based candlestick pattern retrieval |
| Macro Analyst | Live yield curve data + web search |
| Quant Engineer | RSI, MA, volatility, support/resistance |
| DL Predictor | LSTM + Transformer dual-model forecasting |
| Cross-market Analyst | DXY / Treasuries / VIX / crude correlation |
| Sentiment Analyst | Social media sentiment via web search |
| Chief Strategist | Synthesis and final trade recommendation |
| Risk Officer | Independent compliance review (🟢/🟡/🔴) |

## Quick Start

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="your_key"
python scripts/run_agents.py
```

## License

MIT License
