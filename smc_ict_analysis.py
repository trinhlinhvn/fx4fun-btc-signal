"""
SMC / ICT Analysis Module
===========================
Smart Money Concepts + Inner Circle Trader methodology:

- Order Blocks (OB): Vùng giá nơi institutional orders được đặt
- Fair Value Gaps (FVG/Imbalance): Vùng giá chưa được fill
- Liquidity Zones: Nơi stop loss tập trung (equal highs/lows)
- Break of Structure (BOS) & Change of Character (CHoCH)
- Premium/Discount Zones (Fibonacci OTE)
- Kill Zones (London, NY sessions)
- Inducement & Liquidity Sweeps
"""
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime
from typing import List, Tuple, Optional


class SMCICTAnalyzer:
    """
    Phân tích BTC theo Smart Money Concepts & ICT methodology.
    Xác định vùng giá institutional đang giao dịch.
    """

    def __init__(self):
        pass

    def find_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """
        Tìm Order Blocks — nến cuối cùng trước khi giá move mạnh (BOS).
        
        Bullish OB: Nến đỏ cuối cùng trước khi giá break lên (demand zone)
        Bearish OB: Nến xanh cuối cùng trước khi giá break xuống (supply zone)
        """
        if len(df) < lookback:
            lookback = len(df) - 1

        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        bullish_obs = []
        bearish_obs = []

        data = df.iloc[-lookback:]

        for i in range(3, len(data) - 2):
            # Bullish Order Block: nến đỏ → followed by strong bullish move
            curr_open = data.iloc[i]["open"]
            curr_close = data.iloc[i]["close"]
            curr_high = data.iloc[i]["high"]
            curr_low = data.iloc[i]["low"]

            # Check 2 candles after for displacement (more reliable)
            next_1_close = data.iloc[i + 1]["close"] if i + 1 < len(data) else curr_close
            next_2_high = data.iloc[i + 2]["high"] if i + 2 < len(data) else next_1_close
            prev_high = data.iloc[i - 1]["high"]

            # Bearish candle followed by strong bullish displacement (2 candles)
            if curr_close < curr_open:  # Red candle
                displacement = float(next_2_high) - float(curr_high)
                candle_size = float(curr_high) - float(curr_low)
                if candle_size > 0 and displacement > candle_size * 2.0 and float(next_1_close) > float(prev_high):
                    bullish_obs.append({
                        "type": "BULLISH_OB",
                        "high": float(curr_high),
                        "low": float(curr_low),
                        "midpoint": float((curr_high + curr_low) / 2),
                        "index": i,
                        "strength": "STRONG" if displacement > candle_size * 3.0 else "MODERATE",
                        "status": "UNMITIGATED" if float(df["low"].iloc[-1]) > float(curr_low) else "MITIGATED",
                    })

            # Bearish Order Block: nến xanh → followed by strong bearish move
            if curr_close > curr_open:  # Green candle
                next_2_low = data.iloc[i + 2]["low"] if i + 2 < len(data) else next_1_close
                displacement = float(curr_low) - float(next_2_low)
                candle_size = float(curr_high) - float(curr_low)
                prev_low = data.iloc[i - 1]["low"]
                if candle_size > 0 and displacement > candle_size * 2.0 and float(next_1_close) < float(prev_low):
                    bearish_obs.append({
                        "type": "BEARISH_OB",
                        "high": float(curr_high),
                        "low": float(curr_low),
                        "midpoint": float((curr_high + curr_low) / 2),
                        "index": i,
                        "strength": "STRONG" if displacement > candle_size * 3.0 else "MODERATE",
                        "status": "UNMITIGATED" if float(df["high"].iloc[-1]) < float(curr_high) else "MITIGATED",
                    })

        # Keep only unmitigated OBs (active zones)
        active_bullish = [ob for ob in bullish_obs if ob["status"] == "UNMITIGATED"][-3:]
        active_bearish = [ob for ob in bearish_obs if ob["status"] == "UNMITIGATED"][-3:]

        current_price = float(df["close"].iloc[-1])

        return {
            "bullish_obs": active_bullish,
            "bearish_obs": active_bearish,
            "nearest_demand": next((ob for ob in reversed(active_bullish) if ob["high"] < current_price), None),
            "nearest_supply": next((ob for ob in reversed(active_bearish) if ob["low"] > current_price), None),
        }

    def find_fair_value_gaps(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """
        Tìm Fair Value Gaps (FVG) / Imbalances.
        
        Bullish FVG: Low of candle 3 > High of candle 1 (gap up)
        Bearish FVG: High of candle 3 < Low of candle 1 (gap down)
        
        FVG = vùng giá chưa được trade → giá có xu hướng quay lại fill.
        """
        if len(df) < lookback:
            lookback = len(df) - 1

        data = df.iloc[-lookback:]
        bullish_fvgs = []
        bearish_fvgs = []

        for i in range(2, len(data)):
            candle_1_high = float(data.iloc[i - 2]["high"])
            candle_1_low = float(data.iloc[i - 2]["low"])
            candle_3_high = float(data.iloc[i]["high"])
            candle_3_low = float(data.iloc[i]["low"])

            # Bullish FVG: gap between candle 1 high and candle 3 low
            if candle_3_low > candle_1_high:
                gap_size = candle_3_low - candle_1_high
                midpoint = (candle_3_low + candle_1_high) / 2
                # Check if FVG has been filled
                current_low = float(df["low"].iloc[-1])
                filled = current_low <= midpoint

                bullish_fvgs.append({
                    "type": "BULLISH_FVG",
                    "top": candle_3_low,
                    "bottom": candle_1_high,
                    "midpoint": midpoint,
                    "gap_size": gap_size,
                    "filled": filled,
                    "index": i,
                })

            # Bearish FVG: gap between candle 3 high and candle 1 low
            if candle_3_high < candle_1_low:
                gap_size = candle_1_low - candle_3_high
                midpoint = (candle_1_low + candle_3_high) / 2
                current_high = float(df["high"].iloc[-1])
                filled = current_high >= midpoint

                bearish_fvgs.append({
                    "type": "BEARISH_FVG",
                    "top": candle_1_low,
                    "bottom": candle_3_high,
                    "midpoint": midpoint,
                    "gap_size": gap_size,
                    "filled": filled,
                    "index": i,
                })

        # Keep only unfilled FVGs
        active_bullish = [f for f in bullish_fvgs if not f["filled"]][-3:]
        active_bearish = [f for f in bearish_fvgs if not f["filled"]][-3:]

        current_price = float(df["close"].iloc[-1])

        return {
            "bullish_fvgs": active_bullish,
            "bearish_fvgs": active_bearish,
            "nearest_bullish_fvg": next((f for f in reversed(active_bullish) if f["top"] < current_price), None),
            "nearest_bearish_fvg": next((f for f in reversed(active_bearish) if f["bottom"] > current_price), None),
        }

    def find_liquidity_zones(self, df: pd.DataFrame, window: int = 3) -> dict:
        """
        Tìm Liquidity Zones — nơi stop loss tập trung.
        
        - Equal Highs (EQH): Nhiều đỉnh bằng nhau → sell stops phía trên
        - Equal Lows (EQL): Nhiều đáy bằng nhau → buy stops phía dưới
        - Previous session highs/lows
        
        Smart money sẽ hunt liquidity trước khi move thật.
        """
        highs = df["high"].values
        lows = df["low"].values
        current_price = float(df["close"].iloc[-1])

        # Find equal highs (within 0.3% tolerance)
        equal_highs = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                # Check if there's another similar high
                for j in range(i + window, min(i + 20, len(highs))):
                    if highs[j] == max(highs[j - window:j + window + 1]):
                        if abs(highs[i] - highs[j]) / highs[i] < 0.003:
                            equal_highs.append({
                                "level": float((highs[i] + highs[j]) / 2),
                                "touches": 2,
                                "type": "SELL_SIDE_LIQUIDITY",
                            })
                            break

        # Find equal lows
        equal_lows = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window:i + window + 1]):
                for j in range(i + window, min(i + 20, len(lows))):
                    if lows[j] == min(lows[j - window:j + window + 1]):
                        if abs(lows[i] - lows[j]) / lows[i] < 0.003:
                            equal_lows.append({
                                "level": float((lows[i] + lows[j]) / 2),
                                "touches": 2,
                                "type": "BUY_SIDE_LIQUIDITY",
                            })
                            break

        # Deduplicate and keep nearest
        ssl_above = [h for h in equal_highs if h["level"] > current_price]
        bsl_below = [l for l in equal_lows if l["level"] < current_price]

        # Previous swing high/low as liquidity
        swing_highs = []
        swing_lows = []
        sw = 5
        for i in range(sw, len(highs) - sw):
            if highs[i] == max(highs[i - sw:i + sw + 1]):
                swing_highs.append(float(highs[i]))
            if lows[i] == min(lows[i - sw:i + sw + 1]):
                swing_lows.append(float(lows[i]))

        return {
            "sell_side_liquidity": ssl_above[:3],
            "buy_side_liquidity": bsl_below[:3],
            "swing_high_liquidity": [h for h in swing_highs if h > current_price][:2],
            "swing_low_liquidity": [l for l in swing_lows if l < current_price][-2:],
            "nearest_ssl": ssl_above[0]["level"] if ssl_above else (swing_highs[-1] if swing_highs and swing_highs[-1] > current_price else None),
            "nearest_bsl": bsl_below[-1]["level"] if bsl_below else (swing_lows[-1] if swing_lows else None),
        }

    def detect_bos_choch(self, df: pd.DataFrame) -> dict:
        """
        Detect Break of Structure (BOS) & Change of Character (CHoCH).
        
        BOS: Break theo hướng trend hiện tại (continuation)
        CHoCH: Break ngược hướng trend (reversal signal)
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        window = 5

        # Find swing points
        swing_highs = []
        swing_lows = []

        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == min(lows[i - window:i + window + 1]):
                swing_lows.append((i, float(lows[i])))

        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {"bos": None, "choch": None, "structure": "UNDEFINED"}

        current_price = float(closes[-1])

        # Determine current structure
        last_hh = swing_highs[-1][1] > swing_highs[-2][1]
        last_hl = swing_lows[-1][1] > swing_lows[-2][1]
        last_lh = swing_highs[-1][1] < swing_highs[-2][1]
        last_ll = swing_lows[-1][1] < swing_lows[-2][1]

        if last_hh and last_hl:
            structure = "BULLISH"
        elif last_lh and last_ll:
            structure = "BEARISH"
        else:
            structure = "RANGING"

        bos = None
        choch = None

        # BOS: price breaks previous swing high (bullish) or swing low (bearish)
        prev_swing_high = swing_highs[-2][1]
        prev_swing_low = swing_lows[-2][1]

        if structure == "BULLISH" and current_price > prev_swing_high:
            bos = {
                "type": "BULLISH_BOS",
                "level": prev_swing_high,
                "description": f"Price broke above previous swing high ${prev_swing_high:,.0f} — Bullish continuation",
            }
        elif structure == "BEARISH" and current_price < prev_swing_low:
            bos = {
                "type": "BEARISH_BOS",
                "level": prev_swing_low,
                "description": f"Price broke below previous swing low ${prev_swing_low:,.0f} — Bearish continuation",
            }

        # CHoCH: price breaks against the trend
        if structure == "BULLISH" and current_price < swing_lows[-1][1]:
            choch = {
                "type": "BEARISH_CHOCH",
                "level": swing_lows[-1][1],
                "description": f"CHoCH! Price broke below ${swing_lows[-1][1]:,.0f} — Possible trend reversal to BEARISH",
                "strength": "STRONG",
            }
        elif structure == "BEARISH" and current_price > swing_highs[-1][1]:
            choch = {
                "type": "BULLISH_CHOCH",
                "level": swing_highs[-1][1],
                "description": f"CHoCH! Price broke above ${swing_highs[-1][1]:,.0f} — Possible trend reversal to BULLISH",
                "strength": "STRONG",
            }

        return {
            "structure": structure,
            "bos": bos,
            "choch": choch,
            "last_swing_high": swing_highs[-1][1],
            "last_swing_low": swing_lows[-1][1],
        }

    def calculate_premium_discount(self, df: pd.DataFrame) -> dict:
        """
        Premium/Discount Zones dựa trên swing range.
        
        - Premium Zone (> 50%): Vùng giá cao → sell zone
        - Discount Zone (< 50%): Vùng giá thấp → buy zone
        - Equilibrium (50%): Fair price
        - OTE (Optimal Trade Entry): 62-79% Fibonacci retracement
        """
        window = 10
        highs = df["high"].values
        lows = df["low"].values

        # Find the range (recent swing high to swing low)
        recent_high = float(max(highs[-30:]))
        recent_low = float(min(lows[-30:]))
        current_price = float(df["close"].iloc[-1])

        range_size = recent_high - recent_low
        if range_size == 0:
            return {"zone": "UNDEFINED", "level_pct": 50}

        # Position within range (0% = low, 100% = high)
        position_pct = (current_price - recent_low) / range_size * 100

        # Fibonacci levels
        equilibrium = recent_low + range_size * 0.5
        fib_618 = recent_high - range_size * 0.618  # OTE start
        fib_786 = recent_high - range_size * 0.786  # OTE end (for shorts: premium OTE)

        # For longs (discount OTE)
        long_ote_start = recent_low + range_size * 0.618
        long_ote_end = recent_low + range_size * 0.786

        if position_pct > 70:
            zone = "PREMIUM"
            recommendation = "Vùng Premium — Tìm Short setup. Giá đang ở vùng đắt."
        elif position_pct > 50:
            zone = "SLIGHT_PREMIUM"
            recommendation = "Trên Equilibrium — Cẩn thận Long. Tốt hơn nên chờ pullback."
        elif position_pct > 30:
            zone = "SLIGHT_DISCOUNT"
            recommendation = "Dưới Equilibrium — Tìm Long setup. Giá đang ở vùng rẻ tương đối."
        else:
            zone = "DISCOUNT"
            recommendation = "Vùng Discount — Tốt để Long. Giá đang ở vùng rẻ."

        return {
            "zone": zone,
            "position_pct": round(position_pct, 1),
            "current_price": current_price,
            "range_high": recent_high,
            "range_low": recent_low,
            "equilibrium": round(equilibrium, 2),
            "ote_zone": {
                "description": "Optimal Trade Entry (61.8% - 78.6% Fib)",
                "long_entry_start": round(fib_618, 2),
                "long_entry_end": round(fib_786, 2),
                "in_ote": fib_786 <= current_price <= fib_618,
            },
            "recommendation": recommendation,
        }

    def detect_liquidity_sweep(self, df: pd.DataFrame) -> dict:
        """
        Detect Liquidity Sweep / Stop Hunt.
        
        Xảy ra khi giá sweep qua một level rồi reverse mạnh.
        Đây là dấu hiệu smart money đang vào lệnh.
        """
        if len(df) < 10:
            return {"detected": False}

        recent = df.iloc[-5:]
        current_close = float(recent.iloc[-1]["close"])
        current_low = float(recent.iloc[-1]["low"])
        current_high = float(recent.iloc[-1]["high"])

        # Check last 5 candles for sweep pattern
        prev_lows = df["low"].values[-20:-5]
        prev_highs = df["high"].values[-20:-5]

        prev_swing_low = float(min(prev_lows)) if len(prev_lows) > 0 else current_low
        prev_swing_high = float(max(prev_highs)) if len(prev_highs) > 0 else current_high

        sweep = {"detected": False, "type": None, "description": ""}

        # Bullish sweep: wick below previous low but close above
        for i in range(len(recent)):
            candle = recent.iloc[i]
            if float(candle["low"]) < prev_swing_low and float(candle["close"]) > prev_swing_low:
                sweep = {
                    "detected": True,
                    "type": "BULLISH_SWEEP",
                    "level_swept": prev_swing_low,
                    "description": f"🔄 Liquidity Sweep! Price swept ${prev_swing_low:,.0f} rồi reverse lên — Smart money mua BSL",
                    "signal": "BUY",
                    "strength": "STRONG" if float(candle["close"]) > float(candle["open"]) else "MODERATE",
                }
                break

        # Bearish sweep: wick above previous high but close below
        if not sweep["detected"]:
            for i in range(len(recent)):
                candle = recent.iloc[i]
                if float(candle["high"]) > prev_swing_high and float(candle["close"]) < prev_swing_high:
                    sweep = {
                        "detected": True,
                        "type": "BEARISH_SWEEP",
                        "level_swept": prev_swing_high,
                        "description": f"🔄 Liquidity Sweep! Price swept ${prev_swing_high:,.0f} rồi reverse xuống — Smart money bán SSL",
                        "signal": "SELL",
                        "strength": "STRONG" if float(candle["close"]) < float(candle["open"]) else "MODERATE",
                    }
                    break

        return sweep

    def generate_smc_analysis(self, df: pd.DataFrame) -> dict:
        """
        Tổng hợp toàn bộ SMC/ICT analysis thành signal.
        """
        if df.empty or len(df) < 30:
            return {"error": "Insufficient data for SMC analysis", "score": 0}

        current_price = float(df["close"].iloc[-1])

        # Run all SMC components
        order_blocks = self.find_order_blocks(df)
        fvg = self.find_fair_value_gaps(df)
        liquidity = self.find_liquidity_zones(df)
        structure = self.detect_bos_choch(df)
        premium_discount = self.calculate_premium_discount(df)
        sweep = self.detect_liquidity_sweep(df)

        # Generate SMC score (-1 to 1)
        score = 0.0
        narratives = []

        # 1. Structure bias
        if structure["structure"] == "BULLISH":
            score += 0.2
            narratives.append("📈 SMC Structure: BULLISH (HH + HL)")
        elif structure["structure"] == "BEARISH":
            score -= 0.2
            narratives.append("📉 SMC Structure: BEARISH (LH + LL)")

        # 2. BOS/CHoCH
        if structure["bos"]:
            if "BULLISH" in structure["bos"]["type"]:
                score += 0.15
            else:
                score -= 0.15
            narratives.append(f"💥 {structure['bos']['description']}")

        if structure["choch"]:
            if "BULLISH" in structure["choch"]["type"]:
                score += 0.25
            else:
                score -= 0.25
            narratives.append(f"⚡ {structure['choch']['description']}")

        # 3. Premium/Discount
        if premium_discount["zone"] == "DISCOUNT":
            score += 0.15
            narratives.append(f"💰 Price in DISCOUNT zone ({premium_discount['position_pct']:.0f}%) — Bullish bias")
        elif premium_discount["zone"] == "PREMIUM":
            score -= 0.15
            narratives.append(f"💸 Price in PREMIUM zone ({premium_discount['position_pct']:.0f}%) — Bearish bias")

        if premium_discount["ote_zone"]["in_ote"]:
            narratives.append("🎯 Price in OTE zone (61.8-78.6%) — Optimal entry")

        # 4. Order Blocks
        nearest_demand = order_blocks["nearest_demand"]
        nearest_supply = order_blocks["nearest_supply"]

        if nearest_demand:
            dist_to_demand = (current_price - nearest_demand["high"]) / current_price * 100
            if dist_to_demand < 2:
                score += 0.1
                narratives.append(f"🟢 Near Bullish OB (demand) at ${nearest_demand['midpoint']:,.0f} — Expect bounce")

        if nearest_supply:
            dist_to_supply = (nearest_supply["low"] - current_price) / current_price * 100
            if dist_to_supply < 2:
                score -= 0.1
                narratives.append(f"🔴 Near Bearish OB (supply) at ${nearest_supply['midpoint']:,.0f} — Expect rejection")

        # 5. FVG
        nearest_bull_fvg = fvg["nearest_bullish_fvg"]
        nearest_bear_fvg = fvg["nearest_bearish_fvg"]

        if nearest_bull_fvg:
            narratives.append(f"📊 Unfilled Bullish FVG below at ${nearest_bull_fvg['midpoint']:,.0f} — May act as support")

        if nearest_bear_fvg:
            narratives.append(f"📊 Unfilled Bearish FVG above at ${nearest_bear_fvg['midpoint']:,.0f} — May act as resistance")

        # 6. Liquidity Sweep
        if sweep["detected"]:
            # Volume confirmation: sweep with high volume = stronger signal
            vol_boost = 0.0
            if "vol_ratio" in df.columns:
                recent_vol = df["vol_ratio"].iloc[-3:].mean()
                if recent_vol > 1.5:
                    vol_boost = 0.1  # High volume confirms the sweep
                    narratives.append(f"📊 HIGH VOLUME confirms sweep (vol ratio: {recent_vol:.1f}x)")

            if sweep["signal"] == "BUY":
                score += 0.2 + vol_boost
            else:
                score -= 0.2 + vol_boost
            narratives.append(sweep["description"])

        # 7. Liquidity targets
        if liquidity["nearest_ssl"]:
            narratives.append(f"🎯 SSL target above: ${liquidity['nearest_ssl']:,.0f}")
        if liquidity["nearest_bsl"]:
            narratives.append(f"🎯 BSL target below: ${liquidity['nearest_bsl']:,.0f}")

        # Clamp score
        score = max(-1.0, min(1.0, score))

        # SMC-based entry suggestion
        entry_suggestion = self._generate_entry(
            score, current_price, order_blocks, fvg, premium_discount, liquidity, structure
        )

        return {
            "score": round(score, 4),
            "structure": structure,
            "order_blocks": order_blocks,
            "fair_value_gaps": fvg,
            "liquidity": liquidity,
            "premium_discount": premium_discount,
            "sweep": sweep,
            "narratives": narratives,
            "entry_suggestion": entry_suggestion,
        }

    def _generate_entry(self, score, price, obs, fvg, pd_zone, liquidity, structure):
        """Generate SMC-based entry suggestion."""
        if abs(score) < 0.15:
            return {
                "action": "WAIT",
                "reason": "Không có confluence SMC đủ mạnh. Chờ setup rõ ràng hơn.",
            }

        if score > 0:
            # Long bias
            entry_zone = None
            if obs["nearest_demand"]:
                entry_zone = obs["nearest_demand"]["midpoint"]
            elif fvg["nearest_bullish_fvg"]:
                entry_zone = fvg["nearest_bullish_fvg"]["midpoint"]

            sl = liquidity["nearest_bsl"] if liquidity["nearest_bsl"] else price * 0.97
            tp = liquidity["nearest_ssl"] if liquidity["nearest_ssl"] else price * 1.05

            return {
                "action": "LONG",
                "entry_zone": round(entry_zone, 2) if entry_zone else round(price, 2),
                "stop_loss": round(float(sl) * 0.998, 2),
                "take_profit": round(float(tp), 2),
                "reason": "SMC confluence bullish — OB/FVG + structure + discount zone",
            }
        else:
            # Short bias
            entry_zone = None
            if obs["nearest_supply"]:
                entry_zone = obs["nearest_supply"]["midpoint"]
            elif fvg["nearest_bearish_fvg"]:
                entry_zone = fvg["nearest_bearish_fvg"]["midpoint"]

            sl = liquidity["nearest_ssl"] if liquidity["nearest_ssl"] else price * 1.03
            tp = liquidity["nearest_bsl"] if liquidity["nearest_bsl"] else price * 0.95

            return {
                "action": "SHORT",
                "entry_zone": round(entry_zone, 2) if entry_zone else round(price, 2),
                "stop_loss": round(float(sl) * 1.002, 2),
                "take_profit": round(float(tp), 2),
                "reason": "SMC confluence bearish — OB/FVG + structure + premium zone",
            }
