"""
Fund Manager Module — Inspired by top traders worldwide
========================================================

Combines best practices from:
- Paul Tudor Jones: Risk management, position sizing
- Stanley Druckenmiller: Conviction-based sizing
- Linda Raschke: Market regime detection
- Mark Minervini: Trend Template + drawdown protection
- Jim Simons: Statistical signal quality
- Ray Dalio: Risk parity principles
- Ed Seykota: Risk no more than you can afford
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional


class FundManager:
    """
    Quản lý vốn $1000 như một Fund Director chuyên nghiệp.
    Quyết định position size, signal quality, cooldown, drawdown protection.
    """

    STATE_FILE = "fund_state.json"

    def __init__(self, capital_usd: float = 1000.0):
        self.initial_capital = capital_usd
        self.state = self._load_state()
        if "capital" not in self.state:
            self.state["capital"] = capital_usd
            self.state["peak_capital"] = capital_usd
            self.state["trades"] = []
            self.state["last_signal_time"] = None
            self.state["last_signal_type"] = None
            self.state["consecutive_losses"] = 0
            self._save_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[FUND] State save error: {e}")

    # =========================================================
    # VÒNG 1: Paul Tudor Jones — Position Sizing
    # =========================================================
    def calculate_position_size(self, signal_score: float, confidence: str,
                                 risk_pct: float, leverage: int = 10) -> dict:
        """
        Position sizing theo Kelly Criterion + risk management.
        
        Rules:
        - Max risk per trade: 2% capital ($20 with $1000)
        - Position size scales with conviction (Druckenmiller)
        - Reduce size after consecutive losses (PTJ)
        - Stop trading if drawdown >15%
        """
        capital = self.state.get("capital", self.initial_capital)
        peak = self.state.get("peak_capital", self.initial_capital)
        consecutive_losses = self.state.get("consecutive_losses", 0)

        # Drawdown check (Mark Minervini's rule)
        drawdown = (peak - capital) / peak if peak > 0 else 0

        if drawdown > 0.15:
            return {
                "should_trade": False,
                "reason": f"⛔ Drawdown {drawdown*100:.1f}% — STOP TRADING (PTJ rule)",
                "suggested_margin": 0,
                "max_risk_usd": 0,
            }

        # Conviction-based sizing
        abs_score = abs(signal_score)
        if abs_score >= 0.5:
            conviction_multiplier = 1.0
            conviction_label = "HIGH CONVICTION 🔥"
        elif abs_score >= 0.3:
            conviction_multiplier = 0.7
            conviction_label = "MEDIUM CONVICTION ⚡"
        elif abs_score >= 0.15:
            conviction_multiplier = 0.4
            conviction_label = "LOW CONVICTION 💧"
        else:
            return {
                "should_trade": False,
                "reason": "Score < 0.15 — không đủ conviction",
                "suggested_margin": 0,
                "max_risk_usd": 0,
            }

        # Confidence multiplier
        if "HIGH" in confidence:
            conf_mult = 1.0
        elif "MEDIUM" in confidence:
            conf_mult = 0.7
        else:
            conf_mult = 0.4

        # Drawdown reduction (PTJ: reduce size when losing)
        if consecutive_losses >= 3:
            loss_mult = 0.3  # Cut size to 30% after 3 losses
        elif consecutive_losses == 2:
            loss_mult = 0.6
        elif consecutive_losses == 1:
            loss_mult = 0.8
        else:
            loss_mult = 1.0

        # Max risk per trade: 2% capital (Ed Seykota)
        max_risk_pct = 0.02
        max_risk_usd = capital * max_risk_pct * conviction_multiplier * conf_mult * loss_mult

        # Calculate margin needed for this risk
        # If risk_pct is the SL distance %, then:
        # margin_required = max_risk_usd / (risk_pct * leverage / 100)
        if risk_pct > 0:
            suggested_margin = max_risk_usd / (risk_pct / 100)
            # Cap at 30% of capital max (don't risk too much in 1 trade)
            suggested_margin = min(suggested_margin, capital * 0.30)
        else:
            suggested_margin = capital * 0.10

        position_size_usd = suggested_margin * leverage

        return {
            "should_trade": True,
            "capital": round(capital, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "consecutive_losses": consecutive_losses,
            "conviction": conviction_label,
            "max_risk_usd": round(max_risk_usd, 2),
            "max_risk_pct_capital": round(max_risk_usd / capital * 100, 2),
            "suggested_margin": round(suggested_margin, 2),
            "position_size_usd": round(position_size_usd, 2),
            "leverage": leverage,
            "size_modifiers": {
                "conviction": conviction_multiplier,
                "confidence": conf_mult,
                "loss_protection": loss_mult,
            },
        }

    # =========================================================
    # VÒNG 2: Druckenmiller — Strong Signal Filter
    # =========================================================
    def is_strong_signal(self, result: dict) -> tuple[bool, str]:
        """
        Alert khi có signal đủ mạnh (nới lỏng so với v3):
        1. Score >= 0.2 (thay vì 0.5)
        2. Signal phải actionable (BUY/SELL)
        3. R:R >= 1.5:1 (thay vì 2:1)
        """
        score = result.get("final_score", 0)
        signal = result.get("signal", "")
        risk_reward = result.get("risk_reward", {})

        abs_score = abs(score)
        if abs_score < 0.2:
            return False, f"Score {abs_score:.2f} < 0.2"

        if "BUY" not in signal and "SELL" not in signal:
            return False, "No actionable direction"

        rr = risk_reward.get("risk_reward_ratio", 0)
        if rr < 1.5:
            return False, f"R:R {rr}:1 < 1.5:1"

        return True, "✅ Signal passed filters"

    # =========================================================
    # VÒNG 3: Linda Raschke — Market Regime Detection
    # =========================================================
    def detect_market_regime(self, df) -> dict:
        """
        Phát hiện market regime để adjust strategy.
        - TRENDING: trade trend continuation, hold longer
        - RANGING: mean reversion, tighter targets
        - VOLATILE: smaller size, wider SL
        - QUIET: smaller size, take quick profits
        """
        if df is None or df.empty or len(df) < 30:
            return {"regime": "UNKNOWN", "strategy_hint": "Insufficient data"}

        import numpy as np

        # ATR (volatility)
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(abs(high[1:] - close[:-1]),
                                   abs(low[1:] - close[:-1])))
        atr = np.mean(tr[-14:])
        atr_pct = atr / close[-1] * 100

        # ADX-like trend strength (simplified)
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:]) if len(close) >= 50 else ma20
        trend_strength = abs(ma20 - ma50) / ma50 * 100 if ma50 > 0 else 0

        # Range detection
        recent_range = (max(high[-20:]) - min(low[-20:])) / close[-1] * 100

        if trend_strength > 3 and atr_pct > 1.5:
            regime = "TRENDING"
            hint = "📈 Trade trend continuation. Hold longer. TP3 priority."
            sl_multiplier = 2.5  # Wider SL
            tp_multiplier = 1.2  # Larger TP
        elif recent_range < 4 and atr_pct < 1.5:
            regime = "RANGING"
            hint = "↔️ Mean reversion. TP1 only. Tighter SL."
            sl_multiplier = 1.5
            tp_multiplier = 0.7
        elif atr_pct > 3:
            regime = "VOLATILE"
            hint = "⚡ High volatility. Smaller size. Wider SL."
            sl_multiplier = 3.0
            tp_multiplier = 1.5
        else:
            regime = "QUIET"
            hint = "😴 Quiet market. Smaller size. Take quick profits."
            sl_multiplier = 1.5
            tp_multiplier = 0.8

        return {
            "regime": regime,
            "atr_pct": round(atr_pct, 2),
            "trend_strength": round(trend_strength, 2),
            "range_pct": round(recent_range, 2),
            "strategy_hint": hint,
            "sl_multiplier": sl_multiplier,
            "tp_multiplier": tp_multiplier,
        }

    # =========================================================
    # VÒNG 4: Mark Minervini — Trend Template
    # =========================================================
    def check_trend_template(self, df) -> dict:
        """
        Mark Minervini's Trend Template (adapted for crypto):
        1. Price > EMA 50 (trend up)
        2. EMA 21 > EMA 50 (short term aligned)
        3. Price not too extended (within 25% of 50 EMA)
        4. Recent low > previous low (uptrend confirmed)
        
        Nếu không match → KHÔNG nên LONG.
        """
        if df is None or df.empty or len(df) < 50:
            return {"qualifies_long": False, "qualifies_short": False, "score": 0, "details": "Insufficient data"}

        import numpy as np
        close = df["close"].values
        low = df["low"].values
        high = df["high"].values

        ema21 = df["ema_short"].iloc[-1] if "ema_short" in df.columns else np.mean(close[-21:])
        ema50 = df["ema_long"].iloc[-1] if "ema_long" in df.columns else np.mean(close[-50:])
        current = close[-1]

        long_score = 0
        short_score = 0
        details = []

        # Check 1: Price vs EMA50
        if current > ema50:
            long_score += 1
            details.append(f"✓ Price > EMA50 ({current:.0f} > {ema50:.0f})")
        else:
            short_score += 1
            details.append(f"✓ Price < EMA50 ({current:.0f} < {ema50:.0f})")

        # Check 2: EMA alignment
        if ema21 > ema50:
            long_score += 1
        else:
            short_score += 1

        # Check 3: Not too extended
        distance_from_ema50 = abs(current - ema50) / ema50 * 100
        if distance_from_ema50 < 15:
            details.append(f"✓ Not extended ({distance_from_ema50:.1f}%)")
            if current > ema50:
                long_score += 1
            else:
                short_score += 1
        else:
            details.append(f"⚠ Extended {distance_from_ema50:.1f}% — risky")

        # Check 4: Higher low / lower high
        recent_low = min(low[-10:])
        prev_low = min(low[-20:-10]) if len(low) >= 20 else recent_low
        recent_high = max(high[-10:])
        prev_high = max(high[-20:-10]) if len(high) >= 20 else recent_high

        if recent_low > prev_low:
            long_score += 1
            details.append(f"✓ Higher low ({recent_low:.0f} > {prev_low:.0f})")
        elif recent_high < prev_high:
            short_score += 1
            details.append(f"✓ Lower high ({recent_high:.0f} < {prev_high:.0f})")

        return {
            "qualifies_long": long_score >= 3,
            "qualifies_short": short_score >= 3,
            "long_score": long_score,
            "short_score": short_score,
            "details": details[:4],
        }

    # =========================================================
    # VÒNG 5: Jim Simons — Signal Cooldown & Quality
    # =========================================================
    def should_send_alert(self, result: dict, cooldown_minutes: int = 60) -> tuple[bool, str]:
        """
        Anti-spam logic:
        - Không gửi cùng 1 signal type trong X phút
        - Không gửi nếu signal flip quá nhanh
        - Track signal history
        
        Renaissance Tech: only act on signals with statistical edge.
        """
        signal = result.get("signal", "")
        last_time_str = self.state.get("last_signal_time")
        last_type = self.state.get("last_signal_type", "")

        # Different signal type → always send
        if signal != last_type:
            self._update_signal_state(signal)
            return True, "New signal direction"

        # Same signal — check cooldown
        if last_time_str:
            try:
                last_time = datetime.fromisoformat(last_time_str)
                elapsed = (datetime.now() - last_time).total_seconds() / 60
                if elapsed < cooldown_minutes:
                    return False, f"Cooldown active ({elapsed:.0f}/{cooldown_minutes} min)"
            except Exception:
                pass

        self._update_signal_state(signal)
        return True, "Cooldown elapsed"

    def _update_signal_state(self, signal: str):
        self.state["last_signal_time"] = datetime.now().isoformat()
        self.state["last_signal_type"] = signal
        self._save_state()

    # =========================================================
    # Performance tracking
    # =========================================================
    def record_trade(self, pnl_usd: float, signal: str):
        """Record completed trade for performance tracking."""
        self.state["capital"] = self.state.get("capital", self.initial_capital) + pnl_usd
        if self.state["capital"] > self.state.get("peak_capital", self.initial_capital):
            self.state["peak_capital"] = self.state["capital"]

        if pnl_usd < 0:
            self.state["consecutive_losses"] = self.state.get("consecutive_losses", 0) + 1
        else:
            self.state["consecutive_losses"] = 0

        self.state["trades"].append({
            "time": datetime.now().isoformat(),
            "signal": signal,
            "pnl_usd": pnl_usd,
            "capital_after": self.state["capital"],
        })
        self._save_state()

    def get_performance_summary(self) -> dict:
        """Get fund performance metrics."""
        trades = self.state.get("trades", [])
        capital = self.state.get("capital", self.initial_capital)
        peak = self.state.get("peak_capital", self.initial_capital)

        wins = sum(1 for t in trades if t["pnl_usd"] > 0)
        losses = sum(1 for t in trades if t["pnl_usd"] < 0)
        total = len(trades)

        return {
            "initial_capital": self.initial_capital,
            "current_capital": round(capital, 2),
            "peak_capital": round(peak, 2),
            "total_pnl": round(capital - self.initial_capital, 2),
            "total_pnl_pct": round((capital - self.initial_capital) / self.initial_capital * 100, 2),
            "drawdown_pct": round((peak - capital) / peak * 100, 2) if peak > 0 else 0,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "consecutive_losses": self.state.get("consecutive_losses", 0),
        }
