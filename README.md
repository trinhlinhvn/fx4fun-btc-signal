# 🪙 Fx4Fun v4.1 STABLE — BTC/USDT Signal Bot

## Backtest-optimized | PF 2.03 | RR 2:1 | H4 trend + M15 entry

---

## Tham số (backtest-optimized)

```
SL:  1.5%   (~$950)
TP1: 3.0%   (~$1,900)  R:R 2:1
TP2: 4.5%   (~$2,800)  R:R 3:1
TP3: 7.5%   (~$4,700)  R:R 5:1

Quét: mỗi 5 phút
Timeframe: H4 (trend) + M15 (entry)
Alert: Telegram khi score >= 0.2
```

---

## Kiến trúc

```
 Binance Futures API (free)
       │
       ├── H4 klines (200 candles, trend)
       ├── M15 klines (100 candles, entry)
       ├── Funding Rate + Open Interest
       └── Taker Buy Volume (order flow)
       │
       ▼
 ┌─────────────────────────────────────┐
 │      9 ANALYSIS ENGINES             │
 │                                     │
 │  TA (2.5x) + SMC (2.0x)            │
 │  Volume Profile (2.2x)             │
 │  Order Flow (2.2x)                 │
 │  Money Flow (1.8x)                 │
 │  ML/AI (1.5x) + On-chain (1.5x)   │
 │  KOL (1.2x) + F&G (1.0x)          │
 │  News (0.8x)                       │
 └──────────────────┬──────────────────┘
                    │
 ┌──────────────────┴──────────────────┐
 │      CONFLUENCE VOTING              │
 │                                     │
 │  Mỗi source VOTE: BUY/SELL/NEUTRAL │
 │  75%+ agree → Amplify 2x           │
 │  Split → Dampen 0.8x               │
 └──────────────────┬──────────────────┘
                    │
 ┌──────────────────┴──────────────────┐
 │      OUTPUT                         │
 │                                     │
 │  Signal: LONG / SHORT / HOLD        │
 │  Entry Zone (±0.15%)                │
 │  SL: 1.5%                           │
 │  TP1: 3.0% | TP2: 4.5% | TP3: 7.5% │
 │  3 lý do ngắn gọn                  │
 └──────────────────┬──────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Web (5173)   Terminal    Telegram
```

---

## 9 Engines

| # | Engine | Vai trò |
|---|--------|---------|
| 1 | Technical Analysis | RSI, MACD, EMA 21/50, BB, Volume Ratio |
| 2 | SMC / ICT | Order Blocks, FVG, BOS/CHoCH, Liquidity |
| 3 | Volume Profile | POC, Value Area, HVN/LVN |
| 4 | Order Flow | Delta, Cum. Delta, Absorption, Imbalance |
| 5 | Money Flow | MFI, CMF, Smart Money Detection |
| 6 | Fear & Greed | Contrarian (Fear=BUY, Greed=SELL) |
| 7 | ML/AI | XGBoost + LSTM ensemble |
| 8 | On-chain | Whale Alert, Funding, L/S ratio, OI |
| 9 | News + KOL | RSS feeds + Trump/Musk/CZ tweets |

---

## Chạy

| File | Mô tả |
|------|--------|
| `start.bat` | Web Dashboard http://localhost:5173 |
| `start_all.bat` | Web + Telegram + Loop 5 min |
| `start_telegram.bat` | Telegram alerts only (nhẹ) |
| `start_backtest.bat` | Kiểm tra hiệu suất strategy |

---

## Telegram Alert

```
🟢🟢 STRONG LONG 🚀
━━━━━━━━━━━━━━━━━━━
BTC/USDT | $63,055 (-1.8%)

Entry Zone: $62,961 - $63,150
SL: $62,109 (1.5%)
TP1: $64,947 (3.0%)
TP2: $65,892 (4.5%)
TP3: $67,784 (7.5%)
R:R 2.0:1 | Score +0.45

Ly do:
1. Volume Profile below VA + Order Flow delta positive
2. SMC: Bullish OB + Discount zone (35% range)
3. TA: RSI oversold + MACD bullish cross

06:30 19/06 | H4/M15 | Refresh 5 min
DYOR.
```

---

## Backtest (60 days)

```
Trades:        9
Win Rate:      55.6%
Profit Factor: 2.03
PnL:           +6.2% (annualized ~37%)
Max Drawdown:  -3.0%
```

---

## Files

```
Fx4Fun/
├── config.py               # Tham số SL/TP/interval
├── signal_engine.py        # Core: 9 engines + Confluence Voting
├── data_fetcher.py         # Binance Futures API
├── technical_analysis.py   # RSI, MACD, EMA, BB
├── smc_ict_analysis.py     # OB, FVG, BOS, Liquidity
├── volume_profile.py       # VP + Order Flow (NEW)
├── money_flow.py           # MFI, CMF, Smart Money
├── fear_greed.py           # F&G + Liquidation Heatmap
├── onchain_monitor.py      # Whale, Funding, L/S
├── kol_monitor.py          # KOL tweets
├── expert_analysis.py      # S/R, Divergence, MTF
├── pro_trading.py          # Grade, Kill Zone, Candle
├── fund_manager.py         # Signal filter
├── backtester.py           # Backtest engine
├── telegram_bot.py         # Alert
├── web_dashboard.py        # Flask + Charts
├── main.py                 # CLI entry
├── templates/dashboard.html # TradingView Candlestick UI
└── start*.bat              # Shortcuts
```

---

⚠️ DYOR. Tool hỗ trợ, không phải lời khuyên tài chính.
