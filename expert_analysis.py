"""
Expert Trading Analysis Module
================================
Phân tích chuyên sâu như một trader chuyên nghiệp:
- Support/Resistance levels
- Market Structure (Higher Highs, Lower Lows)
- Trend Strength & Phase Detection
- Risk/Reward calculation
- Entry/Exit/Stop-loss recommendations
- Multi-timeframe confluence
- Volume profile analysis
- Divergence detection
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from config import TA_CONFIG


class ExpertAnalyzer:
    """
    Phân tích BTC như một chuyên gia trading thực thụ.
    Đưa ra nhận định market structure, vùng giá quan trọng,
    và khuyến nghị entry/exit cụ thể.
    """

    def __init__(self):
        self.config = TA_CONFIG

    def find_support_resistance(self, df: pd.DataFrame, window: int = 5) -> dict:
        """
        Tìm các vùng Support/Resistance dựa trên swing highs/lows.
        Sử dụng phương pháp pivot points + cluster analysis.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        resistance_levels = []
        support_levels = []

        # Tìm swing highs (đỉnh cục bộ)
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistance_levels.append(float(highs[i]))

        # Tìm swing lows (đáy cục bộ)
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window:i + window + 1]):
                support_levels.append(float(lows[i]))

        # Cluster nearby levels (nhóm các mức gần nhau)
        resistance_levels = self._cluster_levels(resistance_levels, threshold=0.015)
        support_levels = self._cluster_levels(support_levels, threshold=0.015)

        current_price = float(closes[-1])

        # Lọc: resistance phải trên giá hiện tại, support phải dưới
        active_resistance = sorted([r for r in resistance_levels if r > current_price])[:3]
        active_support = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]

        # Tính strength cho mỗi level (số lần test)
        resistance_with_strength = []
        for level in active_resistance:
            touches = sum(1 for h in highs if abs(h - level) / level < 0.01)
            resistance_with_strength.append({"price": level, "touches": touches, "strength": min(touches / 3, 1.0)})

        support_with_strength = []
        for level in active_support:
            touches = sum(1 for l in lows if abs(l - level) / level < 0.01)
            support_with_strength.append({"price": level, "touches": touches, "strength": min(touches / 3, 1.0)})

        return {
            "resistance": resistance_with_strength,
            "support": support_with_strength,
            "current_price": current_price,
            "nearest_resistance": active_resistance[0] if active_resistance else None,
            "nearest_support": active_support[0] if active_support else None,
        }

    def _cluster_levels(self, levels: List[float], threshold: float = 0.015) -> List[float]:
        """Nhóm các mức giá gần nhau thành 1 vùng (lấy trung bình)."""
        if not levels:
            return []

        levels = sorted(levels)
        clusters = [[levels[0]]]

        for level in levels[1:]:
            if (level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        return [np.mean(cluster) for cluster in clusters]

    def detect_market_structure(self, df: pd.DataFrame) -> dict:
        """
        Xác định Market Structure:
        - Uptrend: Higher Highs (HH) + Higher Lows (HL)
        - Downtrend: Lower Highs (LH) + Lower Lows (LL)
        - Ranging: Mixed
        - Break of Structure (BOS) detection
        """
        highs = df["high"].values
        lows = df["low"].values
        window = 5

        # Tìm swing points
        swing_highs = []
        swing_lows = []

        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == min(lows[i - window:i + window + 1]):
                swing_lows.append((i, float(lows[i])))

        # Phân tích HH/HL/LH/LL
        hh_count = 0
        lh_count = 0
        hl_count = 0
        ll_count = 0

        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i - 1][1]:
                hh_count += 1
            else:
                lh_count += 1

        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i - 1][1]:
                hl_count += 1
            else:
                ll_count += 1

        # Xác định structure
        total_high_swings = hh_count + lh_count
        total_low_swings = hl_count + ll_count

        if total_high_swings == 0 or total_low_swings == 0:
            structure = "UNDEFINED"
            trend_strength = 0
        else:
            bullish_score = (hh_count / total_high_swings + hl_count / total_low_swings) / 2
            bearish_score = (lh_count / total_high_swings + ll_count / total_low_swings) / 2

            if bullish_score > 0.65:
                structure = "UPTREND"
                trend_strength = bullish_score
            elif bearish_score > 0.65:
                structure = "DOWNTREND"
                trend_strength = bearish_score
            else:
                structure = "RANGING"
                trend_strength = max(bullish_score, bearish_score)

        # Detect Break of Structure
        bos = None
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_high = swing_highs[-1][1]
            prev_high = swing_highs[-2][1]
            last_low = swing_lows[-1][1]
            prev_low = swing_lows[-2][1]
            current = float(df["close"].iloc[-1])

            if structure == "DOWNTREND" and current > prev_high:
                bos = "BULLISH BOS - Price broke above previous swing high"
            elif structure == "UPTREND" and current < prev_low:
                bos = "BEARISH BOS - Price broke below previous swing low"

        return {
            "structure": structure,
            "trend_strength": round(trend_strength, 3),
            "higher_highs": hh_count,
            "lower_highs": lh_count,
            "higher_lows": hl_count,
            "lower_lows": ll_count,
            "break_of_structure": bos,
            "swing_highs": swing_highs[-3:] if swing_highs else [],
            "swing_lows": swing_lows[-3:] if swing_lows else [],
        }

    def detect_divergence(self, df: pd.DataFrame) -> dict:
        """
        Phát hiện Divergence giữa Price và RSI/MACD.
        - Bullish Divergence: Price Lower Low + RSI Higher Low → reversal up
        - Bearish Divergence: Price Higher High + RSI Lower High → reversal down
        """
        if "rsi" not in df.columns or df["rsi"].isna().all():
            return {"detected": False, "type": None}

        window = 5
        prices = df["close"].values
        rsi = df["rsi"].values

        # Tìm swing lows gần nhất cho price và RSI
        price_lows = []
        rsi_lows = []

        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i - window:i + window + 1]):
                price_lows.append((i, prices[i]))
            if not np.isnan(rsi[i]) and rsi[i] == min([r for r in rsi[i - window:i + window + 1] if not np.isnan(r)]):
                rsi_lows.append((i, rsi[i]))

        # Tìm swing highs
        price_highs = []
        rsi_highs = []

        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i - window:i + window + 1]):
                price_highs.append((i, prices[i]))
            if not np.isnan(rsi[i]) and rsi[i] == max([r for r in rsi[i - window:i + window + 1] if not np.isnan(r)]):
                rsi_highs.append((i, rsi[i]))

        divergence = {"detected": False, "type": None, "description": ""}

        # Bullish Divergence: Price LL + RSI HL
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (price_lows[-1][1] < price_lows[-2][1] and
                    rsi_lows[-1][1] > rsi_lows[-2][1]):
                divergence = {
                    "detected": True,
                    "type": "BULLISH",
                    "description": "Price tạo Lower Low nhưng RSI tạo Higher Low → tín hiệu đảo chiều tăng",
                    "strength": "STRONG" if rsi[-1] < 40 else "MODERATE",
                }

        # Bearish Divergence: Price HH + RSI LH
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and
                    rsi_highs[-1][1] < rsi_highs[-2][1]):
                divergence = {
                    "detected": True,
                    "type": "BEARISH",
                    "description": "Price tạo Higher High nhưng RSI tạo Lower High → tín hiệu đảo chiều giảm",
                    "strength": "STRONG" if rsi[-1] > 60 else "MODERATE",
                }

        return divergence

    def calculate_trend_phase(self, df: pd.DataFrame) -> dict:
        """
        Xác định phase hiện tại theo Wyckoff:
        - Accumulation: sideway sau downtrend (smart money mua)
        - Markup: uptrend mạnh
        - Distribution: sideway sau uptrend (smart money bán)
        - Markdown: downtrend mạnh
        """
        if len(df) < 30:
            return {"phase": "UNKNOWN", "description": "Insufficient data"}

        closes = df["close"].values
        recent = closes[-15:]  # 15 candles gần nhất
        earlier = closes[-30:-15]  # 15 candles trước đó

        recent_return = (recent[-1] - recent[0]) / recent[0]
        earlier_return = (earlier[-1] - earlier[0]) / earlier[0]
        recent_volatility = np.std(recent) / np.mean(recent)
        earlier_volatility = np.std(earlier) / np.mean(earlier)

        # Logic phân phase
        if earlier_return < -0.03 and abs(recent_return) < 0.02 and recent_volatility < earlier_volatility:
            phase = "ACCUMULATION"
            description = "Sideway sau downtrend — Smart money đang tích lũy. Chuẩn bị breakout lên."
            bias = "BULLISH"
        elif recent_return > 0.03 and recent_volatility > 0.01:
            phase = "MARKUP"
            description = "Uptrend mạnh — Momentum tăng rõ ràng. Tìm điểm mua pullback."
            bias = "BULLISH"
        elif earlier_return > 0.03 and abs(recent_return) < 0.02 and recent_volatility < earlier_volatility:
            phase = "DISTRIBUTION"
            description = "Sideway sau uptrend — Smart money đang phân phối. Cẩn thận breakdown."
            bias = "BEARISH"
        elif recent_return < -0.03 and recent_volatility > 0.01:
            phase = "MARKDOWN"
            description = "Downtrend mạnh — Áp lực bán lớn. Chờ tín hiệu đảo chiều."
            bias = "BEARISH"
        else:
            phase = "TRANSITION"
            description = "Đang chuyển giao giữa các phase. Chờ xác nhận rõ ràng hơn."
            bias = "NEUTRAL"

        return {
            "phase": phase,
            "description": description,
            "bias": bias,
            "recent_return": round(recent_return * 100, 2),
            "volatility": round(recent_volatility * 100, 2),
        }

    def calculate_risk_reward(self, df: pd.DataFrame, signal_direction: str) -> dict:
        """
        Tính toán Risk/Reward ratio và đề xuất vị trí Entry/SL/TP
        như một trader chuyên nghiệp.
        """
        sr_levels = self.find_support_resistance(df)
        current_price = sr_levels["current_price"]
        nearest_support = sr_levels["nearest_support"]
        nearest_resistance = sr_levels["nearest_resistance"]

        # ATR-based stop loss
        atr = self._calculate_atr(df, period=14)

        if "BUY" in signal_direction or signal_direction == "UP":
            # Long setup
            entry = current_price
            stop_loss = max(
                current_price - 2 * atr,
                nearest_support * 0.995 if nearest_support else current_price * 0.97
            )
            take_profit_1 = nearest_resistance if nearest_resistance else current_price * 1.03
            take_profit_2 = current_price + 3 * (current_price - stop_loss)  # 3:1 RR
            take_profit_3 = current_price + 5 * (current_price - stop_loss)  # 5:1 RR

            risk = current_price - stop_loss
            reward = take_profit_1 - current_price
            rr_ratio = reward / risk if risk > 0 else 0

            position_type = "LONG"

        elif "SELL" in signal_direction or signal_direction == "DOWN":
            # Short setup
            entry = current_price
            stop_loss = min(
                current_price + 2 * atr,
                nearest_resistance * 1.005 if nearest_resistance else current_price * 1.03
            )
            take_profit_1 = nearest_support if nearest_support else current_price * 0.97
            take_profit_2 = current_price - 3 * (stop_loss - current_price)
            take_profit_3 = current_price - 5 * (stop_loss - current_price)

            risk = stop_loss - current_price
            reward = current_price - take_profit_1
            rr_ratio = reward / risk if risk > 0 else 0

            position_type = "SHORT"
        else:
            return {
                "position_type": "NO TRADE",
                "reason": "Signal is HOLD/NEUTRAL — không có setup rõ ràng",
                "suggestion": "Chờ breakout hoặc tín hiệu mạnh hơn",
            }

        risk_pct = abs(risk) / current_price * 100

        # Futures x10 calculation
        futures = self._calculate_futures(
            position_type, entry, stop_loss,
            [take_profit_1, take_profit_2, take_profit_3],
            leverage=10
        )

        return {
            "position_type": position_type,
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            "take_profit_3": round(take_profit_3, 2),
            "risk_reward_ratio": round(rr_ratio, 2),
            "risk_percent": round(risk_pct, 2),
            "atr": round(atr, 2),
            "position_size_suggestion": self._suggest_position_size(risk_pct),
            "futures": futures,
        }

    def _calculate_futures(self, position_type: str, entry: float, stop_loss: float,
                           take_profits: list, leverage: int = 10) -> dict:
        """
        Tính toán chi tiết cho Futures x10.
        Giới hạn max loss = 30% margin (safety rule).
        """
        margin = 1000.0
        position_size = margin * leverage
        qty = position_size / entry

        # Max SL check: không cho phép loss > 30% margin
        max_sl_pct = 0.03  # 3% price move = 30% ROE with x10
        
        if position_type == "LONG":
            max_sl_price = entry * (1 - max_sl_pct)
            if stop_loss < max_sl_price:
                stop_loss = max_sl_price  # Tighten SL
            
            # Liquidation (Binance formula approximation)
            # Liq = Entry * (1 - IMR + MMR) where IMR=1/lev, MMR~0.4%
            liquidation_price = entry * (1 - 1/leverage + 0.004)
            
            sl_pnl = qty * (stop_loss - entry)
            sl_pct = (stop_loss - entry) / entry * 100
            sl_roe = sl_pct * leverage

            tp_results = []
            for i, tp in enumerate(take_profits):
                tp_pct = (tp - entry) / entry * 100
                tp_pnl = qty * (tp - entry)
                tp_roe = tp_pct * leverage
                tp_results.append({
                    "target": f"TP{i+1}",
                    "price": round(tp, 2),
                    "pnl_usd": round(tp_pnl, 2),
                    "pnl_pct": round(tp_pct, 2),
                    "roe_pct": round(tp_roe, 2),
                })

        else:  # SHORT
            max_sl_price = entry * (1 + max_sl_pct)
            if stop_loss > max_sl_price:
                stop_loss = max_sl_price

            liquidation_price = entry * (1 + 1/leverage - 0.004)

            sl_pnl = qty * (entry - stop_loss)
            sl_pct = (entry - stop_loss) / entry * 100
            sl_roe = sl_pct * leverage

            tp_results = []
            for i, tp in enumerate(take_profits):
                tp_pct = (entry - tp) / entry * 100
                tp_pnl = qty * (entry - tp)
                tp_roe = tp_pct * leverage
                tp_results.append({
                    "target": f"TP{i+1}",
                    "price": round(tp, 2),
                    "pnl_usd": round(tp_pnl, 2),
                    "pnl_pct": round(tp_pct, 2),
                    "roe_pct": round(tp_roe, 2),
                })

        return {
            "leverage": leverage,
            "margin_usd": margin,
            "position_size_usd": position_size,
            "quantity_btc": round(qty, 6),
            "liquidation_price": round(liquidation_price, 2),
            "sl_pnl_usd": round(sl_pnl, 2),
            "sl_roe_pct": round(sl_roe, 2),
            "take_profits": tp_results,
            "max_loss_pct": round(abs(sl_roe), 1),
        }

    def analyze_multi_timeframe(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> dict:
        """
        Phân tích multi-timeframe H1 + H4.
        H4 = xác định trend chính (bias)
        H1 = tìm điểm entry chính xác
        """
        result = {
            "h4": {"trend": "UNKNOWN", "bias": "NEUTRAL"},
            "h1": {"trend": "UNKNOWN", "bias": "NEUTRAL"},
            "confluence": False,
            "recommendation": "",
        }

        # H4 Analysis (trend chính)
        if not df_h4.empty and len(df_h4) >= 20:
            h4_structure = self.detect_market_structure(df_h4)
            h4_phase = self.calculate_trend_phase(df_h4)
            result["h4"] = {
                "trend": h4_structure["structure"],
                "bias": h4_phase["bias"],
                "strength": h4_structure["trend_strength"],
                "phase": h4_phase["phase"],
            }

        # H1 Analysis (entry timing)
        if not df_h1.empty and len(df_h1) >= 20:
            h1_structure = self.detect_market_structure(df_h1)
            h1_phase = self.calculate_trend_phase(df_h1)
            h1_sr = self.find_support_resistance(df_h1, window=3)
            result["h1"] = {
                "trend": h1_structure["structure"],
                "bias": h1_phase["bias"],
                "strength": h1_structure["trend_strength"],
                "phase": h1_phase["phase"],
                "nearest_support": h1_sr["nearest_support"],
                "nearest_resistance": h1_sr["nearest_resistance"],
            }

        # Confluence check
        h4_bias = result["h4"]["bias"]
        h1_bias = result["h1"]["bias"]

        if h4_bias == h1_bias and h4_bias != "NEUTRAL":
            result["confluence"] = True
            if h4_bias == "BULLISH":
                result["recommendation"] = (
                    f"✅ CONFLUENCE BULLISH — H4 {result['h4']['phase']} + H1 {result['h1']['phase']}. "
                    f"Tìm Long entry tại H1 support ${result['h1'].get('nearest_support', 'N/A'):,.0f}"
                    if result['h1'].get('nearest_support') else
                    f"✅ CONFLUENCE BULLISH — H4 + H1 đều bullish. Tìm pullback để Long."
                )
            else:
                result["recommendation"] = (
                    f"✅ CONFLUENCE BEARISH — H4 {result['h4']['phase']} + H1 {result['h1']['phase']}. "
                    f"Tìm Short entry tại H1 resistance ${result['h1'].get('nearest_resistance', 'N/A'):,.0f}"
                    if result['h1'].get('nearest_resistance') else
                    f"✅ CONFLUENCE BEARISH — H4 + H1 đều bearish. Tìm rally để Short."
                )
        elif h4_bias != h1_bias and h4_bias != "NEUTRAL" and h1_bias != "NEUTRAL":
            result["confluence"] = False
            result["recommendation"] = (
                f"⚠️ CONFLICT — H4 {h4_bias} nhưng H1 {h1_bias}. "
                f"Ưu tiên theo H4 (timeframe lớn hơn). Chờ H1 align."
            )
        else:
            result["recommendation"] = "⏸️ Chưa có confluence rõ ràng. Chờ cả 2 TF cùng hướng."

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                abs(high[1:] - close[:-1]),
                abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-period:])
        return float(atr)

    def _suggest_position_size(self, risk_pct: float) -> str:
        """Đề xuất position size dựa trên risk %."""
        if risk_pct < 1:
            return "Aggressive (3-5% portfolio) — Risk thấp"
        elif risk_pct < 2:
            return "Normal (2-3% portfolio) — Risk vừa phải"
        elif risk_pct < 3:
            return "Conservative (1-2% portfolio) — Risk cao"
        else:
            return "Minimal (0.5-1% portfolio) — Risk rất cao, cẩn thận"

    def generate_expert_commentary(self, df: pd.DataFrame, ta_result: dict, ml_result: dict, news_result: dict) -> dict:
        """
        Tổng hợp tất cả phân tích thành nhận định chuyên gia.
        Output giống như một trader pro viết analysis.
        """
        # Gather all analyses
        sr_levels = self.find_support_resistance(df)
        market_structure = self.detect_market_structure(df)
        divergence = self.detect_divergence(df)
        trend_phase = self.calculate_trend_phase(df)

        current_price = float(df["close"].iloc[-1])

        # Build expert narrative
        narratives = []

        # 1. Market Structure
        struct = market_structure["structure"]
        if struct == "UPTREND":
            narratives.append(f"📈 BTC đang trong UPTREND với {market_structure['higher_highs']} Higher Highs. Xu hướng tăng vẫn intact.")
        elif struct == "DOWNTREND":
            narratives.append(f"📉 BTC đang trong DOWNTREND với {market_structure['lower_lows']} Lower Lows. Bears đang kiểm soát.")
        else:
            narratives.append("↔️ BTC đang SIDEWAY/RANGING. Chờ breakout để xác nhận hướng đi.")

        # 2. Trend Phase
        narratives.append(f"🔄 Phase: {trend_phase['phase']} — {trend_phase['description']}")

        # 3. Key Levels
        if sr_levels["nearest_resistance"]:
            narratives.append(f"🔺 Resistance gần nhất: ${sr_levels['nearest_resistance']:,.0f}")
        if sr_levels["nearest_support"]:
            narratives.append(f"🔻 Support gần nhất: ${sr_levels['nearest_support']:,.0f}")

        # 4. Divergence
        if divergence["detected"]:
            narratives.append(f"⚡ {divergence['type']} DIVERGENCE detected! {divergence['description']}")

        # 5. BOS
        if market_structure.get("break_of_structure"):
            narratives.append(f"💥 {market_structure['break_of_structure']}")

        # 6. TA Summary
        ta_score = ta_result.get("score", 0)
        if ta_score > 0.3:
            narratives.append("✅ Technical indicators: Bullish confluence — RSI + MACD + EMA đều positive.")
        elif ta_score < -0.3:
            narratives.append("❌ Technical indicators: Bearish confluence — Indicators đều negative.")
        else:
            narratives.append("⚖️ Technical indicators: Mixed signals — Không có confluence rõ ràng.")

        # 7. Sentiment
        news_score = news_result.get("score", 0)
        articles = news_result.get("articles_analyzed", 0)
        if news_score > 0.15:
            narratives.append(f"📰 Sentiment tích cực ({articles} articles analyzed) — Tin tức ủng hộ tăng giá.")
        elif news_score < -0.15:
            narratives.append(f"📰 Sentiment tiêu cực ({articles} articles) — FUD đang lan rộng.")
        else:
            narratives.append(f"📰 Sentiment trung tính ({articles} articles) — Không có catalyst rõ ràng.")

        # 8. ML Prediction
        ml_prediction = ml_result.get("prediction", "N/A")
        ml_confidence = ml_result.get("confidence", 0)
        if ml_prediction == "UP" and ml_confidence > 0.5:
            narratives.append(f"🤖 AI dự báo: TĂNG (confidence {ml_confidence:.0%}) — Model detect pattern bullish.")
        elif ml_prediction == "DOWN" and ml_confidence > 0.5:
            narratives.append(f"🤖 AI dự báo: GIẢM (confidence {ml_confidence:.0%}) — Model detect pattern bearish.")
        else:
            narratives.append(f"🤖 AI dự báo: Chưa chắc chắn (confidence {ml_confidence:.0%}) — Cần thêm data.")

        # Final verdict
        overall_bias = self._determine_bias(ta_score, news_score, ml_result, market_structure, trend_phase)

        return {
            "current_price": current_price,
            "market_structure": market_structure,
            "trend_phase": trend_phase,
            "support_resistance": sr_levels,
            "divergence": divergence,
            "narratives": narratives,
            "overall_bias": overall_bias,
            "risk_level": self._assess_risk_level(df, market_structure, divergence),
        }

    def _determine_bias(self, ta_score, news_score, ml_result, structure, phase) -> dict:
        """Xác định bias tổng thể từ tất cả sources."""
        bullish_points = 0
        bearish_points = 0
        reasons = []

        if ta_score > 0.2:
            bullish_points += 2
            reasons.append("TA bullish")
        elif ta_score < -0.2:
            bearish_points += 2
            reasons.append("TA bearish")

        if news_score > 0.1:
            bullish_points += 1
            reasons.append("News positive")
        elif news_score < -0.1:
            bearish_points += 1
            reasons.append("News negative")

        if ml_result.get("prediction") == "UP":
            bullish_points += 2
            reasons.append("ML predicts UP")
        elif ml_result.get("prediction") == "DOWN":
            bearish_points += 2
            reasons.append("ML predicts DOWN")

        if structure["structure"] == "UPTREND":
            bullish_points += 2
            reasons.append("Uptrend structure")
        elif structure["structure"] == "DOWNTREND":
            bearish_points += 2
            reasons.append("Downtrend structure")

        if phase["bias"] == "BULLISH":
            bullish_points += 1
            reasons.append(f"{phase['phase']} phase")
        elif phase["bias"] == "BEARISH":
            bearish_points += 1
            reasons.append(f"{phase['phase']} phase")

        total = bullish_points + bearish_points
        if total == 0:
            return {"direction": "NEUTRAL", "strength": 0, "reasons": ["No clear signals"]}

        if bullish_points > bearish_points:
            strength = bullish_points / (total) * 100
            return {"direction": "BULLISH", "strength": round(strength), "reasons": reasons}
        elif bearish_points > bullish_points:
            strength = bearish_points / (total) * 100
            return {"direction": "BEARISH", "strength": round(strength), "reasons": reasons}
        else:
            return {"direction": "NEUTRAL", "strength": 50, "reasons": reasons}

    def _assess_risk_level(self, df: pd.DataFrame, structure: dict, divergence: dict) -> dict:
        """Đánh giá mức độ rủi ro hiện tại."""
        risk_factors = []
        risk_score = 0

        # High volatility = higher risk
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * 100
        if volatility > 3:
            risk_factors.append(f"Volatility cao ({volatility:.1f}%)")
            risk_score += 2
        elif volatility > 2:
            risk_factors.append(f"Volatility vừa ({volatility:.1f}%)")
            risk_score += 1

        # Divergence = potential reversal risk
        if divergence.get("detected"):
            risk_factors.append(f"{divergence['type']} divergence — risk đảo chiều")
            risk_score += 2

        # Ranging market = choppy, hard to trade
        if structure["structure"] == "RANGING":
            risk_factors.append("Market sideway — dễ bị whipsaw")
            risk_score += 1

        # Near ATH/ATL
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if not np.isnan(rsi):
                if rsi > 80:
                    risk_factors.append(f"RSI extreme overbought ({rsi:.0f})")
                    risk_score += 2
                elif rsi < 20:
                    risk_factors.append(f"RSI extreme oversold ({rsi:.0f})")
                    risk_score += 2

        if risk_score >= 5:
            level = "HIGH ⚠️"
        elif risk_score >= 3:
            level = "MEDIUM ⚡"
        else:
            level = "LOW ✅"

        return {
            "level": level,
            "score": risk_score,
            "factors": risk_factors,
            "recommendation": self._risk_recommendation(risk_score),
        }

    def _risk_recommendation(self, risk_score: int) -> str:
        if risk_score >= 5:
            return "Giảm position size. Chỉ trade với SL chặt. Hoặc đứng ngoài chờ cơ hội tốt hơn."
        elif risk_score >= 3:
            return "Trade với position size vừa phải. Set SL rõ ràng. Không FOMO."
        else:
            return "Điều kiện thuận lợi. Có thể trade với confidence cao hơn."
