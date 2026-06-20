"""
Money Flow Indicator — Lux Algo Style
=======================================
Kết hợp Volume + Price Action để detect dòng tiền thông minh.

Bao gồm:
1. Money Flow Index (MFI) — RSI nhưng dùng volume-weighted
2. Chaikin Money Flow (CMF) — Accumulation/Distribution pressure
3. Volume-Weighted Momentum — Lux Algo signature
4. Smart Money Flow (custom) — Detect institutional buying/selling

Ý nghĩa:
- MFI > 80 + giá tăng = Smart money đang distribute (SELL signal)
- MFI < 20 + giá giảm = Smart money đang accumulate (BUY signal)
- CMF positive + volume spike = Strong buying pressure
- Divergence giữa price và money flow = reversal signal mạnh
"""
import numpy as np
import pandas as pd


class MoneyFlowAnalyzer:
    """
    Money Flow analysis inspired by Lux Algo.
    Combines MFI + CMF + Volume Momentum + Smart Money detection.
    """

    def __init__(self):
        self.mfi_period = 14
        self.cmf_period = 20
        self.momentum_period = 10

    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI):
        Like RSI but uses volume to measure buying/selling pressure.
        Range: 0-100. Overbought > 80, Oversold < 20.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        # Positive/Negative flow
        pos_flow = pd.Series(0.0, index=df.index)
        neg_flow = pd.Series(0.0, index=df.index)

        tp_diff = typical_price.diff()
        pos_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
        neg_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]

        pos_sum = pos_flow.rolling(period).sum()
        neg_sum = neg_flow.rolling(period).sum()

        # Avoid division by zero
        money_ratio = pos_sum / neg_sum.replace(0, 1)
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow (CMF):
        Measures accumulation/distribution over a period.
        Range: -1 to +1. Positive = buying pressure, Negative = selling pressure.
        """
        high_low = df["high"] - df["low"]
        # Avoid division by zero
        high_low = high_low.replace(0, 0.01)

        # Money Flow Multiplier
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low

        # Money Flow Volume
        mf_volume = mf_multiplier * df["volume"]

        # CMF = sum(MF Volume) / sum(Volume)
        cmf = mf_volume.rolling(period).sum() / df["volume"].rolling(period).sum()

        return cmf

    def calculate_volume_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Volume-Weighted Momentum (Lux Algo style):
        Price change weighted by relative volume.
        High volume moves have more weight = institutional activity.
        """
        # Relative volume (current vs average)
        vol_sma = df["volume"].rolling(20).mean()
        rel_vol = df["volume"] / vol_sma.replace(0, 1)

        # Price momentum
        price_change = df["close"].pct_change(period)

        # Volume-weighted momentum
        vw_momentum = price_change * rel_vol

        # Smooth
        vw_momentum_smooth = vw_momentum.rolling(3).mean()

        return vw_momentum_smooth

    def detect_smart_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Smart Money activity:
        - Large volume candles with small body (absorption)
        - Volume spike at support/resistance (institutional entry)
        - Divergence between volume trend and price trend
        """
        result = pd.DataFrame(index=df.index)

        # Candle body ratio
        body = abs(df["close"] - df["open"])
        full_range = df["high"] - df["low"]
        body_ratio = body / full_range.replace(0, 1)

        # Volume vs average
        vol_sma = df["volume"].rolling(20).mean()
        vol_ratio = df["volume"] / vol_sma.replace(0, 1)

        # Smart Money Absorption: high volume + small body = institutional order absorbing retail
        result["absorption"] = (vol_ratio > 1.5) & (body_ratio < 0.4)

        # Smart Money Buying: high volume + bullish close in lower half of range
        lower_half = df["close"] < (df["high"] + df["low"]) / 2
        result["smart_buying"] = (vol_ratio > 1.3) & (df["close"] > df["open"]) & lower_half

        # Smart Money Selling: high volume + bearish close in upper half
        upper_half = df["close"] > (df["high"] + df["low"]) / 2
        result["smart_selling"] = (vol_ratio > 1.3) & (df["close"] < df["open"]) & upper_half

        # Cumulative Smart Money Flow
        sm_flow = pd.Series(0.0, index=df.index)
        sm_flow[result["smart_buying"]] = vol_ratio[result["smart_buying"]]
        sm_flow[result["smart_selling"]] = -vol_ratio[result["smart_selling"]]
        result["sm_flow_cumulative"] = sm_flow.cumsum()

        return result

    def detect_mf_divergence(self, df: pd.DataFrame, mfi: pd.Series) -> dict:
        """
        Detect divergence between Price and Money Flow Index.
        MFI divergence is MORE reliable than RSI divergence because it includes volume.
        """
        if len(df) < 30 or mfi.isna().all():
            return {"detected": False}

        window = 5
        prices = df["close"].values
        mfi_vals = mfi.values

        # Find price swing lows
        price_lows = []
        mfi_lows = []
        price_highs = []
        mfi_highs = []

        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i-window:i+window+1]):
                price_lows.append((i, prices[i]))
            if prices[i] == max(prices[i-window:i+window+1]):
                price_highs.append((i, prices[i]))

            if not np.isnan(mfi_vals[i]):
                valid_mfi = [m for m in mfi_vals[i-window:i+window+1] if not np.isnan(m)]
                if valid_mfi and mfi_vals[i] == min(valid_mfi):
                    mfi_lows.append((i, mfi_vals[i]))
                if valid_mfi and mfi_vals[i] == max(valid_mfi):
                    mfi_highs.append((i, mfi_vals[i]))

        # Bullish divergence: Price LL + MFI HL
        if len(price_lows) >= 2 and len(mfi_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and mfi_lows[-1][1] > mfi_lows[-2][1]:
                return {
                    "detected": True,
                    "type": "BULLISH",
                    "description": "Price Lower Low + MFI Higher Low → Smart money accumulating while retail sells",
                    "strength": "STRONG" if mfi_vals[-1] < 30 else "MODERATE",
                }

        # Bearish divergence: Price HH + MFI LH
        if len(price_highs) >= 2 and len(mfi_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and mfi_highs[-1][1] < mfi_highs[-2][1]:
                return {
                    "detected": True,
                    "type": "BEARISH",
                    "description": "Price Higher High + MFI Lower High → Smart money distributing while retail buys",
                    "strength": "STRONG" if mfi_vals[-1] > 70 else "MODERATE",
                }

        return {"detected": False}

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Full Money Flow analysis.
        Returns score (-1 to 1) + all components.
        """
        if df.empty or len(df) < 30 or "volume" not in df.columns:
            return {"score": 0, "error": "Insufficient data or no volume"}

        # Calculate all indicators
        mfi = self.calculate_mfi(df, self.mfi_period)
        cmf = self.calculate_cmf(df, self.cmf_period)
        vol_momentum = self.calculate_volume_momentum(df, self.momentum_period)
        smart_money = self.detect_smart_money_flow(df)
        divergence = self.detect_mf_divergence(df, mfi)

        # Latest values
        latest_mfi = float(mfi.iloc[-1]) if not mfi.isna().iloc[-1] else 50
        latest_cmf = float(cmf.iloc[-1]) if not cmf.isna().iloc[-1] else 0
        latest_vol_mom = float(vol_momentum.iloc[-1]) if not vol_momentum.isna().iloc[-1] else 0

        # Recent smart money activity (last 5 candles)
        recent_sm = smart_money.iloc[-5:]
        sm_buying_count = recent_sm["smart_buying"].sum()
        sm_selling_count = recent_sm["smart_selling"].sum()
        absorption_count = recent_sm["absorption"].sum()

        # === SCORING ===
        score = 0.0
        narratives = []

        # MFI signal
        if latest_mfi < 20:
            score += 0.25
            narratives.append(f"💰 MFI {latest_mfi:.0f} — OVERSOLD (smart money accumulating)")
        elif latest_mfi > 80:
            score -= 0.25
            narratives.append(f"💸 MFI {latest_mfi:.0f} — OVERBOUGHT (smart money distributing)")
        elif latest_mfi < 40:
            score += 0.1
            narratives.append(f"📊 MFI {latest_mfi:.0f} — Leaning oversold")
        elif latest_mfi > 60:
            score -= 0.1
            narratives.append(f"📊 MFI {latest_mfi:.0f} — Leaning overbought")

        # CMF signal
        if latest_cmf > 0.15:
            score += 0.2
            narratives.append(f"🟢 CMF +{latest_cmf:.2f} — Strong buying pressure (accumulation)")
        elif latest_cmf < -0.15:
            score -= 0.2
            narratives.append(f"🔴 CMF {latest_cmf:.2f} — Strong selling pressure (distribution)")
        elif latest_cmf > 0.05:
            score += 0.08
        elif latest_cmf < -0.05:
            score -= 0.08

        # Volume momentum
        if latest_vol_mom > 0.02:
            score += 0.15
            narratives.append(f"📈 Volume momentum BULLISH ({latest_vol_mom:+.3f})")
        elif latest_vol_mom < -0.02:
            score -= 0.15
            narratives.append(f"📉 Volume momentum BEARISH ({latest_vol_mom:+.3f})")

        # Smart money detection
        if sm_buying_count >= 2:
            score += 0.15
            narratives.append(f"🐋 Smart money BUYING detected ({int(sm_buying_count)} signals in 5 candles)")
        elif sm_selling_count >= 2:
            score -= 0.15
            narratives.append(f"🐋 Smart money SELLING detected ({int(sm_selling_count)} signals in 5 candles)")

        if absorption_count >= 2:
            narratives.append(f"⚡ Absorption detected ({int(absorption_count)}x) — institutional orders absorbing")

        # MFI Divergence (strongest signal)
        if divergence["detected"]:
            if divergence["type"] == "BULLISH":
                score += 0.2
            else:
                score -= 0.2
            narratives.append(f"⚡ MF DIVERGENCE {divergence['type']} — {divergence['description'][:80]}")

        score = max(-1.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "mfi": round(latest_mfi, 1),
            "cmf": round(latest_cmf, 4),
            "vol_momentum": round(latest_vol_mom, 4),
            "smart_money_buying": int(sm_buying_count),
            "smart_money_selling": int(sm_selling_count),
            "absorption": int(absorption_count),
            "divergence": divergence,
            "narratives": narratives,
        }
