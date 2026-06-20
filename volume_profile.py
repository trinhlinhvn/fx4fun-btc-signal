"""
Volume Profile & Order Flow Analysis
======================================
Hai phương pháp phân tích chuyên nghiệp nhất hiện nay:

1. VOLUME PROFILE:
   - POC (Point of Control): Vùng giá có volume giao dịch nhiều nhất
   - Value Area (VA): Vùng chứa 70% volume → price tends to return here
   - HVN (High Volume Node): Support/Resistance mạnh
   - LVN (Low Volume Node): Price moves fast through these areas
   
2. ORDER FLOW:
   - Delta (Buy volume - Sell volume): net buying/selling pressure
   - Cumulative Delta: trend of buying/selling over time
   - Absorption: large orders absorbing at a level (support/resistance)
   - Imbalance: aggressive buyers/sellers (ratio > 3:1)
   
Binance cung cấp:
- Taker Buy Volume vs Total Volume (proxy cho order flow)
- Klines có taker_buy_volume → tính delta
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


class VolumeProfileAnalyzer:
    """
    Volume Profile — phân tích phân bổ volume theo giá.
    Xác định POC, Value Area, HVN/LVN.
    """

    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins

    def calculate_volume_profile(self, df: pd.DataFrame) -> dict:
        """
        Tính Volume Profile từ OHLCV data.
        Phân bổ volume vào các price bins.
        """
        if df.empty or "volume" not in df.columns or len(df) < 10:
            return {"error": "Insufficient data"}

        # Price range
        price_low = float(df["low"].min())
        price_high = float(df["high"].max())
        price_range = price_high - price_low

        if price_range == 0:
            return {"error": "No price range"}

        # Create price bins
        bin_size = price_range / self.num_bins
        bins = np.linspace(price_low, price_high, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_volumes = np.zeros(self.num_bins)

        # Distribute volume to bins (VWAP-style distribution per candle)
        for _, row in df.iterrows():
            candle_low = float(row["low"])
            candle_high = float(row["high"])
            candle_vol = float(row["volume"])
            candle_range = candle_high - candle_low

            if candle_range == 0 or candle_vol == 0:
                continue

            # Distribute volume proportionally to bins the candle touches
            for i in range(self.num_bins):
                bin_low = bins[i]
                bin_high = bins[i + 1]

                # Overlap between candle and bin
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)

                if overlap_high > overlap_low:
                    # Proportion of candle in this bin
                    proportion = (overlap_high - overlap_low) / candle_range
                    bin_volumes[i] += candle_vol * proportion

        # POC: Price level with highest volume
        poc_idx = np.argmax(bin_volumes)
        poc_price = float(bin_centers[poc_idx])

        # Value Area: 70% of total volume around POC
        total_volume = bin_volumes.sum()
        va_target = total_volume * 0.70

        # Expand from POC outward
        va_volume = bin_volumes[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while va_volume < va_target:
            # Expand toward higher volume side
            expand_low = bin_volumes[va_low_idx - 1] if va_low_idx > 0 else 0
            expand_high = bin_volumes[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else 0

            if expand_high >= expand_low and va_high_idx < self.num_bins - 1:
                va_high_idx += 1
                va_volume += bin_volumes[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                va_volume += bin_volumes[va_low_idx]
            else:
                break

        va_low = float(bin_centers[va_low_idx])
        va_high = float(bin_centers[va_high_idx])

        # HVN (High Volume Nodes) — top 5 bins
        hvn_indices = np.argsort(bin_volumes)[-5:][::-1]
        hvn_levels = [float(bin_centers[i]) for i in hvn_indices]

        # LVN (Low Volume Nodes) — lowest 5 bins with some volume
        non_zero = bin_volumes > total_volume * 0.005  # At least 0.5% volume
        lvn_candidates = np.where(non_zero, bin_volumes, np.inf)
        lvn_indices = np.argsort(lvn_candidates)[:5]
        lvn_levels = [float(bin_centers[i]) for i in lvn_indices if bin_volumes[i] > 0]

        current_price = float(df["close"].iloc[-1])

        # Signal interpretation
        if current_price < va_low:
            position = "BELOW_VA"
            signal = 0.2  # Price below value → likely to return up
            note = "Price dưới Value Area → xu hướng quay về POC (LONG bias)"
        elif current_price > va_high:
            position = "ABOVE_VA"
            signal = -0.2  # Price above value → likely to return down
            note = "Price trên Value Area → xu hướng quay về POC (SHORT bias)"
        elif abs(current_price - poc_price) / poc_price < 0.005:
            position = "AT_POC"
            signal = 0.0  # At POC = fair value
            note = "Price tại POC — fair value, chờ breakout"
        else:
            position = "IN_VA"
            signal = 0.0
            note = "Price trong Value Area — ranging"

        return {
            "poc": round(poc_price, 2),
            "value_area_high": round(va_high, 2),
            "value_area_low": round(va_low, 2),
            "hvn_levels": [round(h, 2) for h in hvn_levels[:3]],
            "lvn_levels": [round(l, 2) for l in lvn_levels[:3]],
            "current_price": round(current_price, 2),
            "position": position,
            "signal": signal,
            "note": note,
            "total_volume": round(total_volume, 2),
        }


class OrderFlowAnalyzer:
    """
    Order Flow — phân tích áp lực mua/bán từ taker volume.
    Binance cung cấp taker_buy_volume trong klines.
    """

    def calculate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Delta = Taker Buy Volume - Taker Sell Volume.
        Positive delta = aggressive buying. Negative = aggressive selling.
        """
        if "volume" not in df.columns:
            return df

        # Nếu có taker_buy_volume từ Binance
        if "taker_buy_volume" in df.columns:
            # Chuyển sang float nếu cần
            taker_buy = df["taker_buy_volume"].astype(float) if df["taker_buy_volume"].dtype == object else df["taker_buy_volume"]
        else:
            # Estimate: nếu close > open → taker buy dominant
            taker_buy = df["volume"].copy()
            bullish = df["close"] > df["open"]
            # Bullish candle: ~60% taker buy, bearish: ~40%
            taker_buy[bullish] = df["volume"][bullish] * 0.6
            taker_buy[~bullish] = df["volume"][~bullish] * 0.4

        taker_sell = df["volume"] - taker_buy

        df = df.copy()
        df["taker_buy_vol"] = taker_buy
        df["taker_sell_vol"] = taker_sell
        df["delta"] = taker_buy - taker_sell
        df["cumulative_delta"] = df["delta"].cumsum()
        df["delta_ma5"] = df["delta"].rolling(5).mean()

        return df

    def detect_absorption(self, df: pd.DataFrame, window: int = 3) -> list:
        """
        Detect Absorption: Large volume at a level but price doesn't move.
        Sign of institutional orders absorbing retail pressure.
        → Strong S/R level.
        """
        if "delta" not in df.columns or len(df) < window + 2:
            return []

        absorptions = []
        vol_avg = df["volume"].rolling(20).mean()

        for i in range(window, len(df) - 1):
            vol = float(df["volume"].iloc[i])
            avg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else vol
            body = abs(float(df["close"].iloc[i]) - float(df["open"].iloc[i]))
            full_range = float(df["high"].iloc[i]) - float(df["low"].iloc[i])

            if full_range == 0:
                continue

            body_ratio = body / full_range

            # Absorption: high volume + small body (< 30% of range)
            if vol > avg * 1.5 and body_ratio < 0.3:
                level = float(df["close"].iloc[i])
                direction = "SUPPORT" if float(df["delta"].iloc[i]) > 0 else "RESISTANCE"
                absorptions.append({
                    "price": round(level, 2),
                    "type": direction,
                    "volume_ratio": round(vol / avg, 1),
                    "index": i,
                })

        return absorptions[-5:]  # Last 5

    def detect_imbalance(self, df: pd.DataFrame) -> list:
        """
        Detect Order Flow Imbalance: When buy/sell ratio > 3:1.
        Aggressive directional pressure → momentum.
        """
        if "taker_buy_vol" not in df.columns:
            return []

        imbalances = []
        for i in range(len(df) - 5, len(df)):  # Last 5 candles
            if i < 0:
                continue
            buy = float(df["taker_buy_vol"].iloc[i])
            sell = float(df["taker_sell_vol"].iloc[i])

            if sell > 0 and buy / sell > 3:
                imbalances.append({
                    "type": "BUY_IMBALANCE",
                    "ratio": round(buy / sell, 1),
                    "price": round(float(df["close"].iloc[i]), 2),
                    "signal": 0.15,
                })
            elif buy > 0 and sell / buy > 3:
                imbalances.append({
                    "type": "SELL_IMBALANCE",
                    "ratio": round(sell / buy, 1),
                    "price": round(float(df["close"].iloc[i]), 2),
                    "signal": -0.15,
                })

        return imbalances

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Full Order Flow analysis.
        Returns score + components.
        """
        if df.empty or len(df) < 20:
            return {"score": 0, "error": "Insufficient data"}

        # Calculate delta
        df = self.calculate_delta(df)

        # Get latest values
        latest_delta = float(df["delta"].iloc[-1]) if "delta" in df.columns else 0
        cum_delta = float(df["cumulative_delta"].iloc[-1]) if "cumulative_delta" in df.columns else 0
        delta_ma = float(df["delta_ma5"].iloc[-1]) if "delta_ma5" in df.columns and not pd.isna(df["delta_ma5"].iloc[-1]) else 0

        # Delta trend (last 10 candles)
        recent_delta = df["delta"].iloc[-10:] if "delta" in df.columns else pd.Series([0])
        delta_trend = "RISING" if recent_delta.iloc[-1] > recent_delta.iloc[0] else "FALLING"

        # Absorption & Imbalance
        absorptions = self.detect_absorption(df)
        imbalances = self.detect_imbalance(df)

        # Score calculation
        score = 0.0
        narratives = []

        # Delta direction
        avg_vol = df["volume"].mean()
        normalized_delta = latest_delta / avg_vol if avg_vol > 0 else 0

        if normalized_delta > 0.15:
            score += 0.2
            narratives.append(f"📈 Delta dương mạnh ({normalized_delta:.2f}) — buyers aggressive")
        elif normalized_delta < -0.15:
            score -= 0.2
            narratives.append(f"📉 Delta âm mạnh ({normalized_delta:.2f}) — sellers aggressive")

        # Cumulative delta trend
        cum_delta_5 = float(df["cumulative_delta"].iloc[-5]) if len(df) >= 5 else 0
        cum_change = cum_delta - cum_delta_5
        normalized_cum = cum_change / (avg_vol * 5) if avg_vol > 0 else 0

        if normalized_cum > 0.1:
            score += 0.15
            narratives.append(f"📊 Cumulative Delta tăng — net buying 5 candles")
        elif normalized_cum < -0.1:
            score -= 0.15
            narratives.append(f"📊 Cumulative Delta giảm — net selling 5 candles")

        # Imbalances
        for imb in imbalances:
            score += imb["signal"]
            narratives.append(f"⚡ {imb['type']} (ratio {imb['ratio']}:1) tại ${imb['price']:,.0f}")

        # Absorption
        if absorptions:
            last_abs = absorptions[-1]
            narratives.append(f"🛡️ Absorption detected ({last_abs['type']}) tại ${last_abs['price']:,.0f} (vol {last_abs['volume_ratio']}x)")

        score = max(-1.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "delta": round(latest_delta, 2),
            "cumulative_delta_trend": delta_trend,
            "normalized_delta": round(normalized_delta, 4),
            "absorptions": absorptions,
            "imbalances": imbalances,
            "narratives": narratives,
        }


class VolumeOrderFlowEngine:
    """Combines Volume Profile + Order Flow into one analysis."""

    def __init__(self):
        self.vp = VolumeProfileAnalyzer(num_bins=40)
        self.of = OrderFlowAnalyzer()

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Full Volume Profile + Order Flow analysis.
        Returns combined score + all data.
        """
        vp_result = self.vp.calculate_volume_profile(df)
        of_result = self.of.analyze(df)

        vp_score = vp_result.get("signal", 0) if "error" not in vp_result else 0
        of_score = of_result.get("score", 0)

        # Combined: Order Flow 60% + Volume Profile 40%
        combined_score = of_score * 0.6 + vp_score * 0.4

        # Build narratives
        narratives = []
        if "error" not in vp_result:
            narratives.append(f"📊 POC: ${vp_result['poc']:,.0f} | VA: ${vp_result['value_area_low']:,.0f}-${vp_result['value_area_high']:,.0f}")
            narratives.append(f"📍 Position: {vp_result['position']} — {vp_result['note']}")

        narratives.extend(of_result.get("narratives", []))

        return {
            "score": round(combined_score, 4),
            "volume_profile": vp_result,
            "order_flow": of_result,
            "narratives": narratives,
        }
