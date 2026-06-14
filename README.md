# 🪙 BTC Trading Signal Bot — Expert Edition v3.0 (Fx4Fun)

Phần mềm phân tích và đưa ra tín hiệu trading BTC/USDT Futures **cấp độ chuyên gia**, kết hợp 5 nguồn phân tích:

1. **📈 Technical Analysis** — RSI, MACD, EMA 21/50, Bollinger Bands, Volume Analysis
2. **📰 News Sentiment** — RSS feeds từ CoinDesk, CoinTelegraph + NLP sentiment (free, không cần key)
3. **🤖 ML/AI Prediction** — Ensemble XGBoost + LSTM dự báo trend
4. **🧠 Expert Analysis** — Market Structure, S/R, Divergence, Wyckoff Phase, Multi-timeframe H1/H4
5. **💎 SMC/ICT** — Order Blocks, Fair Value Gaps, Liquidity Zones, BOS/CHoCH, Premium/Discount, Liquidity Sweep

## ✨ Features

- 🌐 **Web Dashboard** — Dark theme, realtime charts (Bollinger Bands + MACD, H1/H4)
- 📱 **Telegram Bot** — Gửi tín hiệu tự động + interactive commands
- ⚡ **Futures x10** — Entry/SL/TP cụ thể, Liquidation price, ROE%, PnL
- 📐 **Multi-timeframe** — H1 + H4 real data từ Binance Futures
- 🔄 **Anti-whipsaw** — Tránh flip signal liên tục khi sideway
- 📊 **Dynamic Weights** — Tự redistribute khi 1 source không có data
- 💰 **Funding Rate** — Theo dõi market positioning (Long/Short bias)
- 🛡️ **Max SL Cap** — Giới hạn loss tối đa 30% margin cho Futures x10

## 🏗️ Data Sources

| Source | Type | Key Required | 
|--------|------|:---:|
| Binance Futures API | Price, OHLCV, Funding Rate | ❌ Free |
| CoinDesk RSS | News | ❌ Free |
| CoinTelegraph RSS | News | ❌ Free |
| NewsAPI.org | News (optional) | ✅ Optional |

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
python -m textblob.download_corpora

# Config (optional - app chạy được không cần key nào)
copy .env.example .env
```

## 📖 Sử dụng

```bash
# Chạy 1 lần
python main.py

# Train ML + chạy
python main.py --train

# Web Dashboard (http://localhost:5000)
python main.py --web

# Telegram alerts
python main.py --telegram

# Production (Web + Telegram + Loop 5 phút)
python main.py --train --all
```

## 🏗️ Architecture

```
Fx4Fun/
├── main.py                 # Entry point + Rich terminal UI
├── config.py               # Tất cả cấu hình (weights, thresholds)
├── data_fetcher.py         # Binance Futures API (OHLCV + Funding Rate)
├── technical_analysis.py   # RSI, MACD, EMA, BB + Volume Analysis
├── news_sentiment.py       # Multi-source news (RSS + NewsAPI) + NLP
├── ml_predictor.py         # XGBoost + LSTM ensemble
├── expert_analysis.py      # Market Structure, S/R, Divergence, Wyckoff
├── smc_ict_analysis.py     # SMC: OB, FVG, Liquidity, BOS/CHoCH, OTE
├── signal_engine.py        # Combines all → final signal
├── telegram_bot.py         # Telegram integration
├── web_dashboard.py        # Flask + SocketIO web UI
├── templates/
│   ├── dashboard.html      # Main dashboard (signal + charts)
│   └── charts.html         # Dedicated charts page
└── models/                 # Saved ML models (auto)
```

## 📊 Signal Logic

| Score | Signal | Futures Action |
|-------|--------|---------------|
| ≥ 0.6 | STRONG BUY 🟢🟢 | Long x10, full size |
| 0.3 → 0.6 | BUY 🟢 | Long x10, half size |
| -0.3 → 0.3 | HOLD 🟡 | No new position |
| -0.6 → -0.3 | SELL 🔴 | Short x10, half size |
| ≤ -0.6 | STRONG SELL 🔴🔴 | Short x10, full size |

## 🧠 SMC/ICT Methodology

- **Order Blocks**: Demand/Supply zones (2-candle displacement confirmation)
- **Fair Value Gaps**: Unfilled imbalances (price tends to fill)
- **Liquidity Zones**: Equal Highs/Lows where stops cluster
- **BOS/CHoCH**: Structure continuation vs reversal
- **Premium/Discount**: Fibonacci OTE (61.8-78.6%)
- **Liquidity Sweep**: Stop hunt detection with volume confirmation

## ⚡ Futures x10 Risk Management

- Max SL: 3% price move (= 30% ROE loss cap)
- Liquidation formula: Binance-accurate (IMR + MMR)
- Position size suggestion based on risk %
- Multi-target TP: Conservative, 3:1 RR, 5:1 RR

## 🔧 Technical Improvements (v3.0)

- Dynamic weight allocation (no dead weight from unavailable sources)
- Anti-whipsaw logic (requires stronger score to flip direction)
- Volume analysis integrated (vol_ratio for breakout confirmation)
- EMA 21/50 (suitable for H4 swing trading, not day-trading noise)
- Funding rate monitoring (detect overcrowded positions)
- Confidence score: X/N sources agree (not just HIGH/MED/LOW)
- Binance Futures API: 1200 req/min, no rate limit issues
- Smart caching: 30s price, 2min klines, 5min market data

## ⚠️ Disclaimer

Tool hỗ trợ phân tích, **KHÔNG phải lời khuyên tài chính**.
Past performance ≠ Future results. Always DYOR.
