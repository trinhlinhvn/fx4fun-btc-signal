"""
Signal Engine
Combines Technical Analysis + News Sentiment + ML Prediction + Expert Analysis
to generate final trading signals like a professional trader.
"""
from datetime import datetime
from config import SIGNAL_WEIGHTS, SIGNAL_THRESHOLDS, ML_CONFIG
from data_fetcher import BTCDataFetcher
from technical_analysis import TechnicalAnalyzer
from news_sentiment import NewsSentimentAnalyzer
from ml_predictor import EnsemblePredictor
from expert_analysis import ExpertAnalyzer
from smc_ict_analysis import SMCICTAnalyzer
from fund_manager import FundManager
from kol_monitor import KOLMonitor
from onchain_monitor import OnChainMonitor
from multi_coin_scanner import MultiCoinScanner
from pro_trading import ProTradingSystem
from money_flow import MoneyFlowAnalyzer
from fear_greed import FearGreedIndex, LiquidationHeatmap
from backtester import Backtester
from volume_profile import VolumeOrderFlowEngine


def log(msg: str):
    """Print with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


class SignalEngine:
    """
    Core engine that combines all analysis sources into a unified signal.

    Signal scoring:
    - Technical Analysis: RSI, MACD, EMA, Bollinger Bands
    - News Sentiment: NLP analysis of crypto news
    - ML Prediction: XGBoost + LSTM ensemble
    - Expert Analysis: Market structure, S/R, divergence, risk management

    Final score range: -1.0 (Strong Sell) to +1.0 (Strong Buy)
    """

    def __init__(self):
        self.data_fetcher = BTCDataFetcher()
        self.ta_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.ml_predictor = EnsemblePredictor()
        self.expert = ExpertAnalyzer()
        self.smc = SMCICTAnalyzer()
        self.fund = FundManager(capital_usd=1000.0)
        self.kol = KOLMonitor()
        self.onchain = OnChainMonitor()
        self.multi_coin = MultiCoinScanner()
        self.pro = ProTradingSystem()
        self.money_flow = MoneyFlowAnalyzer()
        self.fear_greed = FearGreedIndex()
        self.liq_heatmap = LiquidationHeatmap()
        self.vol_flow = VolumeOrderFlowEngine()
        self.weights = SIGNAL_WEIGHTS
        self.thresholds = SIGNAL_THRESHOLDS
        self.is_ml_trained = False
        self._last_signal = None
        self._signal_hold_count = 0
        # Auto-train ML if model not saved
        self._auto_train_ml()

    def train_ml_models(self) -> dict:
        """Train ML models on historical data."""
        log("Fetching training data...")
        df = self.data_fetcher.get_klines(interval="4h", limit=1500)

        if df.empty or len(df) < 100:
            return {"error": "Not enough historical data for training"}

        df = self.ta_analyzer.calculate_indicators(df)

        log("Training ML ensemble...")
        result = self.ml_predictor.train(df)
        self.is_ml_trained = True
        return result

    def _auto_train_ml(self):
        """Auto-train ML model if not already trained (check saved model)."""
        import os
        if os.path.exists("models/xgboost_btc.joblib"):
            self.is_ml_trained = True
            log("ML model loaded from disk.")
        else:
            log("No ML model found. Will train on first signal generation.")

    def generate_signal(self) -> dict:
        """
        Generate comprehensive trading signal with expert-level analysis.
        Returns detailed breakdown of all analysis components.
        """
        timestamp = datetime.now().isoformat()

        # 1. Fetch price data (multi-timeframe from Binance Futures)
        log("Fetching price data...")
        current_price = self.data_fetcher.get_current_price()
        # H4 data: trend direction (200 candles = ~33 days)
        historical_df = self.data_fetcher.get_h4_data(limit=200)
        # M15 data: entry timing (100 candles = ~25 hours)
        historical_m15 = self.data_fetcher.get_klines(interval="15m", limit=100)

        if historical_df.empty:
            return {"error": "Cannot fetch price data", "timestamp": timestamp}

        # 2. Technical Analysis
        log("Running technical analysis...")
        ta_result = self.ta_analyzer.analyze(historical_df)
        ta_score = ta_result.get("score", 0)

        # 3. News Sentiment
        log("Analyzing news sentiment...")
        news_result = self.news_analyzer.analyze()
        news_score = news_result.get("score", 0)

        # 4. ML Prediction
        log("Running ML prediction...")
        historical_with_ta = self.ta_analyzer.calculate_indicators(historical_df.copy())
        ml_result = self.ml_predictor.predict(historical_with_ta)
        ml_score = ml_result.get("score", 0)

        # 5. Expert Analysis
        log("Generating expert analysis...")
        expert_commentary = self.expert.generate_expert_commentary(
            historical_with_ta, ta_result, ml_result, news_result
        )

        # 5b. Multi-timeframe analysis (H4 trend + M15 entry)
        log("Multi-timeframe analysis (H4 + M15)...")
        m15_with_ta = self.ta_analyzer.calculate_indicators(historical_m15.copy()) if not historical_m15.empty else historical_m15
        mtf_result = self.expert.analyze_multi_timeframe(m15_with_ta, historical_with_ta)

        # 5c. SMC / ICT Analysis
        log("SMC/ICT analysis...")
        smc_result = self.smc.generate_smc_analysis(historical_with_ta)
        smc_score = smc_result.get("score", 0)

        # 5d. KOL / Politician Twitter Monitor (non-blocking, use cached if slow)
        log("Scanning KOL tweets...")
        try:
            kol_result = self.kol.scan_all_kols(hours=6)
        except Exception:
            kol_result = {"score": 0, "total_tweets": 0, "tier1_active": False}
        kol_score = kol_result.get("score", 0)

        # 5e. On-chain & Exchange Flow Monitor
        log("Scanning on-chain & exchange flows...")
        try:
            onchain_result = self.onchain.scan_all(symbol="BTCUSDT")
        except Exception:
            onchain_result = {"score": 0}
        onchain_score = onchain_result.get("score", 0)

        # 5f. Money Flow Analysis (Lux Algo style)
        log("Money Flow analysis...")
        mf_result = self.money_flow.analyze(historical_with_ta)
        mf_score = mf_result.get("score", 0)

        # 5g. Volume Profile + Order Flow
        log("Volume Profile + Order Flow...")
        vof_result = self.vol_flow.analyze(historical_with_ta)
        vof_score = vof_result.get("score", 0)

        # 5h. Fear & Greed Index (contrarian)
        log("Fear & Greed Index...")
        fg_result = self.fear_greed.get_current()
        fg_score = fg_result.get("score", 0)

        # 5h. Liquidation Heatmap
        log("Liquidation Heatmap...")
        liq_result = self.liq_heatmap.estimate_liquidation_clusters(
            current_price.get("price", 0) if current_price else 0
        )

        # 6. CONFLUENCE VOTING SYSTEM (thay thế weighted average cũ)
        # =========================================================
        # Thay vì average (bị dilute), dùng voting:
        # - Mỗi source VOTE: BULLISH (+1), BEARISH (-1), hoặc NEUTRAL (0)
        # - Score = (bullish_votes - bearish_votes) / total_votes × amplifier
        # - Amplifier tăng khi confluence cao (3+ sources cùng hướng)
        # =========================================================
        
        has_ml = not ("error" in ml_result or not self.is_ml_trained)
        ml_note = None if has_ml else "ML not available."

        # Normalize each score to vote: > threshold = bullish, < -threshold = bearish
        vote_threshold = 0.05  # Hạ threshold để TA dễ vote hơn (trước 0.08)
        
        votes = []
        vote_details = {}

        # Core sources (always present) — ƯU TIÊN TA
        sources = [
            ("TA", ta_score, 2.5),           # TA ưu tiên cao nhất
            ("SMC", smc_score, 2.0),         # SMC vẫn quan trọng
            ("Money_Flow", mf_score, 1.8),   # Volume-based = reliable
            ("Vol_Profile", vof_score, 2.2), # Volume Profile + Order Flow
            ("Fear_Greed", fg_score, 1.0),   # Contrarian signal
        ]
        
        # Optional sources
        if has_ml and ml_score != 0:
            sources.append(("ML", ml_score, 1.5))
        if news_score != 0:
            sources.append(("News", news_score, 0.8))
        if onchain_score != 0:
            sources.append(("Onchain", onchain_score, 1.5))
        if kol_score != 0:
            sources.append(("KOL", kol_score, 1.2 if not kol_result.get("tier1_active") else 2.5))

        total_weight = 0
        weighted_vote_sum = 0
        bullish_count = 0
        bearish_count = 0

        for name, score, weight in sources:
            if score > vote_threshold:
                vote = 1
                bullish_count += 1
            elif score < -vote_threshold:
                vote = -1
                bearish_count += 1
            else:
                vote = 0

            votes.append({"source": name, "score": round(score, 4), "vote": vote, "weight": weight})
            weighted_vote_sum += vote * weight * abs(score)  # Stronger score = more influence
            total_weight += weight

        # Confluence amplifier
        total_voters = bullish_count + bearish_count
        if total_voters == 0:
            confluence_ratio = 0
            confluence_amp = 0.5
        else:
            # How aligned are the votes? (0 = split, 1 = unanimous)
            majority = max(bullish_count, bearish_count)
            confluence_ratio = majority / total_voters
            # Amplify when 75%+ agree (non-linear boost)
            if confluence_ratio >= 0.85:
                confluence_amp = 2.5  # Almost unanimous
            elif confluence_ratio >= 0.75:
                confluence_amp = 2.0  # Strong majority
            elif confluence_ratio >= 0.6:
                confluence_amp = 1.5  # Simple majority
            else:
                confluence_amp = 0.8  # Split — dampen signal

        # Final score
        if total_weight > 0:
            raw_score = weighted_vote_sum / total_weight
        else:
            raw_score = 0

        final_score = raw_score * confluence_amp
        final_score = max(-1.0, min(1.0, final_score))  # Clamp

        # Store for display
        effective_weights = {s[0].lower(): s[2] for s in sources}
        confluence_info = {
            "bullish_votes": bullish_count,
            "bearish_votes": bearish_count,
            "neutral_votes": len(votes) - bullish_count - bearish_count,
            "total_sources": len(votes),
            "confluence_ratio": round(confluence_ratio, 2),
            "amplifier": confluence_amp,
            "raw_score": round(raw_score, 4),
            "votes": votes,
        }

        # 7. Expert bias adjustment
        # Nếu expert analysis phát hiện divergence hoặc BOS, điều chỉnh score
        final_score = self._apply_expert_adjustment(final_score, expert_commentary)

        # 8. Determine signal (with anti-whipsaw logic)
        signal = self._score_to_signal(final_score)

        # Anti-whipsaw: nhẹ hơn — chỉ block nếu flip với score rất yếu
        if self._last_signal and self._last_signal != signal:
            if "BUY" in self._last_signal and "SELL" in signal:
                if final_score > -0.25:
                    signal = "HOLD 🟡"
            elif "SELL" in self._last_signal and "BUY" in signal:
                if final_score < 0.25:
                    signal = "HOLD 🟡"

        self._last_signal = signal

        # 9. Gợi ý Entry Zone + SL + TP (đơn giản, không tính Futures)
        ta_direction = "BUY" if ta_score > 0.05 else "SELL" if ta_score < -0.05 else signal
        risk_reward_direction = signal if "BUY" in signal or "SELL" in signal else ta_direction
        risk_reward = self.expert.calculate_risk_reward(historical_with_ta, risk_reward_direction)
        
        if risk_reward and risk_reward.get("position_type") not in [None]:
            # Smart entry: dùng OB/FVG nếu có entry tốt hơn
            smc_entry = smc_result.get("entry_suggestion", {}) if smc_result else {}
            if smc_entry.get("entry_zone") and smc_entry.get("action") != "WAIT":
                smc_entry_price = smc_entry["entry_zone"]
                current = current_price.get("price", 0) if current_price else 0
                if "LONG" in risk_reward.get("position_type", "") and smc_entry_price < current:
                    risk_reward["entry"] = smc_entry_price
                    risk_reward["entry_zone"]["low"] = round(smc_entry_price * 0.998, 2)
                    risk_reward["entry_zone"]["high"] = round(smc_entry_price * 1.002, 2)
                    risk_reward["entry_zone"]["mid"] = smc_entry_price
                    risk_reward["entry_type"] = "LIMIT (OB/FVG)"
                elif "SHORT" in risk_reward.get("position_type", "") and smc_entry_price > current:
                    risk_reward["entry"] = smc_entry_price
                    risk_reward["entry_zone"]["low"] = round(smc_entry_price * 0.998, 2)
                    risk_reward["entry_zone"]["high"] = round(smc_entry_price * 1.002, 2)
                    risk_reward["entry_zone"]["mid"] = smc_entry_price
                    risk_reward["entry_type"] = "LIMIT (OB/FVG)"
                else:
                    risk_reward["entry_type"] = "MARKET"
            else:
                risk_reward["entry_type"] = "MARKET"

            # Note
            risk_reward["note"] = "Gợi ý vùng giá — phân tích lại mỗi 5 phút"

        # 10. Calculate confidence from confluence voting
        if confluence_ratio >= 0.85:
            agreement = f"HIGH ({bullish_count+bearish_count}/{len(votes)} sources, {int(confluence_ratio*100)}% aligned)"
        elif confluence_ratio >= 0.65:
            agreement = f"MEDIUM ({max(bullish_count,bearish_count)}/{len(votes)} majority)"
        else:
            agreement = f"LOW ({bullish_count}↑ {bearish_count}↓ — sources split)"

        # 11. Generate top 3 trade reasons (pro trader explanation)
        trade_reasons = self.generate_trade_reasons(
            ta_result, news_result, ml_result, smc_result,
            expert_commentary, mtf_result, signal, kol_result
        )

        # 12. Fund Manager — Position sizing, market regime, signal quality
        market_regime = self.fund.detect_market_regime(historical_with_ta)
        trend_template = self.fund.check_trend_template(historical_with_ta)
        is_strong, strong_reason = self.fund.is_strong_signal({
            "final_score": final_score,
            "confidence": agreement,
            "signal": signal,
            "risk_reward": risk_reward,
            "multi_timeframe": mtf_result,
            "trade_reasons": trade_reasons,
        })
        risk_pct_for_sizing = risk_reward.get("risk_percent", 2.0) if isinstance(risk_reward, dict) else 2.0
        position_sizing = self.fund.calculate_position_size(
            final_score, agreement, risk_pct_for_sizing, leverage=10
        )

        # 13. Pro Trading System — 7 Wall Street filters
        log("Pro Trading evaluation...")
        pro_evaluation = self.pro.evaluate_trade(
            {
                "signal": signal,
                "final_score": final_score,
                "confidence": agreement,
                "risk_reward": risk_reward,
                "trade_reasons": trade_reasons,
                "fund_management": {"position_sizing": position_sizing, "market_regime": market_regime, "is_strong_signal": is_strong},
                "multi_timeframe": mtf_result,
                "smc_ict": smc_result,
                "components": {"technical_analysis": ta_result, "news_sentiment": news_result},
                "current_price": current_price,
            },
            df=historical_with_ta,
        )

        return {
            "timestamp": timestamp,
            "signal": signal,
            "final_score": round(final_score, 4),
            "confidence": agreement,
            "confluence": confluence_info,
            "timeframe": {
                "analysis": "H4 (4-hour trend) + M15 (15-min entry)",
                "forecast_horizon": "4-12 giờ tới",
                "entry_tf": "M15 (15-min) — timing entry chính xác",
                "trend_tf": "H4 (4-hour) — xác định trend direction",
                "note": "Phân tích mỗi 5 phút. Gửi Telegram khi có STRONG signal.",
            },
            "current_price": current_price,
            "components": {
                "technical_analysis": {
                    "score": ta_score,
                    "weight": effective_weights.get("ta", 0),
                    "weighted_score": round(ta_score * effective_weights.get("ta", 0), 4),
                    "details": ta_result.get("signals", {}),
                },
                "news_sentiment": {
                    "score": news_score,
                    "weight": effective_weights.get("news", 0),
                    "weighted_score": round(news_score * effective_weights.get("news", 0), 4),
                    "articles_analyzed": news_result.get("articles_analyzed", 0),
                    "source": news_result.get("source", "none"),
                    "top_articles": news_result.get("details", [])[:5],
                },
                "ml_prediction": {
                    "score": ml_score,
                    "weight": effective_weights.get("ml", 0),
                    "weighted_score": round(ml_score * effective_weights.get("ml", 0), 4),
                    "prediction": ml_result.get("prediction", "N/A"),
                    "ml_confidence": ml_result.get("confidence", 0),
                    "models_used": ml_result.get("models_used", []),
                    "note": ml_note,
                },
            },
            "expert_analysis": expert_commentary,
            "multi_timeframe": mtf_result,
            "smc_ict": smc_result,
            "kol_monitor": kol_result,
            "onchain": onchain_result,
            "money_flow": mf_result,
            "volume_order_flow": vof_result,
            "fear_greed": fg_result,
            "liquidation_heatmap": liq_result,
            "risk_reward": risk_reward,
            "trade_reasons": trade_reasons,
            "fund_management": {
                "is_strong_signal": is_strong,
                "strong_signal_reason": strong_reason,
                "market_regime": market_regime,
                "trend_template": trend_template,
                "position_sizing": position_sizing,
                "performance": self.fund.get_performance_summary(),
            },
            "pro_trading": pro_evaluation,
            "market_context": self._safe_get_market_data(),
        }

    def _safe_get_market_data(self) -> dict:
        """Get market data without blocking if rate limited."""
        try:
            return self.data_fetcher.get_market_data()
        except Exception:
            return {}

    def _apply_expert_adjustment(self, score: float, expert: dict) -> float:
        """
        Điều chỉnh score dựa trên expert analysis.
        Divergence và Break of Structure là tín hiệu mạnh.
        """
        adjustment = 0.0

        # Divergence boost
        divergence = expert.get("divergence", {})
        if divergence.get("detected"):
            if divergence["type"] == "BULLISH" and divergence.get("strength") == "STRONG":
                adjustment += 0.15
            elif divergence["type"] == "BEARISH" and divergence.get("strength") == "STRONG":
                adjustment -= 0.15
            elif divergence["type"] == "BULLISH":
                adjustment += 0.08
            elif divergence["type"] == "BEARISH":
                adjustment -= 0.08

        # Break of Structure boost
        bos = expert.get("market_structure", {}).get("break_of_structure")
        if bos:
            if "BULLISH" in bos:
                adjustment += 0.1
            elif "BEARISH" in bos:
                adjustment -= 0.1

        # Clamp to valid range
        return max(-1.0, min(1.0, score + adjustment))

    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to trading signal."""
        if score >= self.thresholds["strong_buy"]:
            return "STRONG BUY 🟢🟢"
        elif score >= self.thresholds["buy"]:
            return "BUY 🟢"
        elif score <= self.thresholds["strong_sell"]:
            return "STRONG SELL 🔴🔴"
        elif score <= self.thresholds["sell"]:
            return "SELL 🔴"
        else:
            return "HOLD 🟡"

    def generate_trade_reasons(self, ta_result, news_result, ml_result, smc_result, expert, mtf, signal_direction, kol_result=None) -> list:
        """
        Trích xuất 3 lý do CỐT LÕI nhất tại sao trade theo direction này.
        Như một pro trader giải thích setup trong 3 câu ngắn gọn.
        """
        is_long = "BUY" in signal_direction or "UP" in signal_direction
        is_short = "SELL" in signal_direction or "DOWN" in signal_direction
        reasons = []

        # === REASON 0: KOL Tier-1 emergency (highest priority if active) ===
        if kol_result and kol_result.get("tier1_active") and kol_result.get("emergency_alerts"):
            for alert in kol_result["emergency_alerts"]:
                if (is_long and alert["sentiment"] > 0) or (is_short and alert["sentiment"] < 0):
                    reasons.append(f"🚨 [{alert['kol']}] tweet {alert['direction']} — '{alert['text'][:80]}...'")
                    break

        # === REASON 1: Market Structure / SMC (cao priority nhất) ===
        smc_struct = smc_result.get("structure", {}) if smc_result else {}
        smc_obs = smc_result.get("order_blocks", {}) if smc_result else {}
        smc_pd = smc_result.get("premium_discount", {}) if smc_result else {}
        smc_sweep = smc_result.get("sweep", {}) if smc_result else {}

        if is_long:
            if smc_sweep.get("detected") and smc_sweep.get("signal") == "BUY":
                reasons.append(f"🔄 Liquidity Sweep tại ${smc_sweep['level_swept']:,.0f} → smart money đã mua, giá reverse lên")
            elif smc_pd.get("zone") in ["DISCOUNT", "SLIGHT_DISCOUNT"]:
                reasons.append(f"💰 Giá ở vùng {smc_pd['zone']} ({smc_pd['position_pct']:.0f}% range) → entry rẻ, R:R cao")
            elif smc_struct.get("bos") and "BULLISH" in smc_struct["bos"].get("type", ""):
                reasons.append(f"💥 Bullish BOS confirmed — break above ${smc_struct['bos']['level']:,.0f}, trend continuation")
            elif smc_obs.get("nearest_demand"):
                ob = smc_obs["nearest_demand"]
                reasons.append(f"🟢 Bullish Order Block tại ${ob['midpoint']:,.0f} ({ob['strength']}) — institutional demand zone")
        elif is_short:
            if smc_sweep.get("detected") and smc_sweep.get("signal") == "SELL":
                reasons.append(f"🔄 Liquidity Sweep tại ${smc_sweep['level_swept']:,.0f} → smart money đã bán, giá reverse xuống")
            elif smc_pd.get("zone") in ["PREMIUM", "SLIGHT_PREMIUM"]:
                reasons.append(f"💸 Giá ở vùng {smc_pd['zone']} ({smc_pd['position_pct']:.0f}% range) → entry đắt, áp lực bán")
            elif smc_struct.get("bos") and "BEARISH" in smc_struct["bos"].get("type", ""):
                reasons.append(f"💥 Bearish BOS confirmed — break below ${smc_struct['bos']['level']:,.0f}, trend continuation")
            elif smc_obs.get("nearest_supply"):
                ob = smc_obs["nearest_supply"]
                reasons.append(f"🔴 Bearish Order Block tại ${ob['midpoint']:,.0f} ({ob['strength']}) — institutional supply zone")

        # === REASON 2: TA Confluence ===
        ta_signals = ta_result.get("signals", {}) if ta_result else {}
        if is_long:
            ta_bull = []
            if ta_signals.get("rsi", {}).get("score", 0) > 0.3:
                ta_bull.append(f"RSI {ta_signals['rsi']['value']:.0f} oversold")
            if ta_signals.get("macd", {}).get("score", 0) > 0:
                ta_bull.append("MACD bullish crossover")
            if ta_signals.get("ema_cross", {}).get("score", 0) > 0:
                ta_bull.append("EMA 21>50 (uptrend)")
            if ta_signals.get("bollinger", {}).get("score", 0) > 0.3:
                ta_bull.append("price tại BB lower (oversold)")

            if len(ta_bull) >= 2:
                reasons.append(f"📈 TA Confluence: {' + '.join(ta_bull[:3])}")

        elif is_short:
            ta_bear = []
            if ta_signals.get("rsi", {}).get("score", 0) < -0.3:
                ta_bear.append(f"RSI {ta_signals['rsi']['value']:.0f} overbought")
            if ta_signals.get("macd", {}).get("score", 0) < 0:
                ta_bear.append("MACD bearish crossover")
            if ta_signals.get("ema_cross", {}).get("score", 0) < 0:
                ta_bear.append("EMA 21<50 (downtrend)")
            if ta_signals.get("bollinger", {}).get("score", 0) < -0.3:
                ta_bear.append("price tại BB upper (overbought)")

            if len(ta_bear) >= 2:
                reasons.append(f"📉 TA Confluence: {' + '.join(ta_bear[:3])}")

        # === REASON 3: Multi-timeframe + News + Divergence ===
        mtf_conf = mtf.get("confluence") if mtf else False
        h4_bias = mtf.get("h4", {}).get("bias", "") if mtf else ""
        divergence = expert.get("divergence", {}) if expert else {}
        news_score = news_result.get("score", 0) if news_result else 0

        if is_long:
            if mtf_conf and h4_bias == "BULLISH":
                reasons.append(f"📐 H4 + H1 đồng thuận BULLISH — multi-timeframe confluence")
            elif divergence.get("detected") and divergence.get("type") == "BULLISH":
                reasons.append(f"⚡ Bullish Divergence ({divergence.get('strength')}) — RSI HL trong khi price LL → reversal signal")
            elif news_score > 0.15:
                reasons.append(f"📰 News sentiment +{news_score:.2f} ({news_result.get('articles_analyzed', 0)} bài) — tin tức ủng hộ tăng")
            elif ml_result.get("prediction") == "UP" and ml_result.get("confidence", 0) > 0.5:
                reasons.append(f"🤖 ML model dự báo UP (confidence {ml_result['confidence']:.0%})")

        elif is_short:
            if mtf_conf and h4_bias == "BEARISH":
                reasons.append(f"📐 H4 + H1 đồng thuận BEARISH — multi-timeframe confluence")
            elif divergence.get("detected") and divergence.get("type") == "BEARISH":
                reasons.append(f"⚡ Bearish Divergence ({divergence.get('strength')}) — RSI LH trong khi price HH → reversal signal")
            elif news_score < -0.15:
                reasons.append(f"📰 News sentiment {news_score:.2f} ({news_result.get('articles_analyzed', 0)} bài) — FUD đang lan rộng")
            elif ml_result.get("prediction") == "DOWN" and ml_result.get("confidence", 0) > 0.5:
                reasons.append(f"🤖 ML model dự báo DOWN (confidence {ml_result['confidence']:.0%})")

        # Đảm bảo có ít nhất 3 reasons (fallback nếu không tìm được)
        if len(reasons) < 3:
            ta_score = ta_result.get("score", 0) if ta_result else 0
            smc_score = smc_result.get("score", 0) if smc_result else 0
            phase = expert.get("trend_phase", {}) if expert else {}

            if len(reasons) < 3:
                if is_long and smc_score > 0.1:
                    reasons.append(f"💎 SMC score +{smc_score:.2f} — institutional bias BULLISH")
                elif is_short and smc_score < -0.1:
                    reasons.append(f"💎 SMC score {smc_score:.2f} — institutional bias BEARISH")

            if len(reasons) < 3 and phase.get("phase"):
                reasons.append(f"🔄 Wyckoff phase: {phase['phase']} — {phase.get('description', '')[:80]}")

            if len(reasons) < 3:
                bias = expert.get("overall_bias", {}) if expert else {}
                if bias.get("direction") != "NEUTRAL":
                    reasons.append(f"🎯 Overall bias {bias.get('direction')} với {bias.get('strength', 0)}% confidence")

        return reasons[:3]

    def scan_top_coins(self) -> dict:
        """Scan top 10 coins, return top 3 with strongest signals."""
        return self.multi_coin.scan_top_coins()

    def _calculate_agreement(self, ta: float, news: float, ml: float, smc: float = 0) -> str:
        """Calculate confidence based on all source agreement."""
        scores = [s for s in [ta, news, ml, smc] if s != 0]
        if not scores:
            return "LOW (no data)"
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        total = len(scores)
        if positive == total or negative == total:
            return f"HIGH ({positive}/{total} sources agree)"
        elif positive >= total * 0.6 or negative >= total * 0.6:
            return f"MEDIUM ({max(positive,negative)}/{total} sources agree)"
        else:
            return f"LOW ({max(positive,negative)}/{total} — sources diverge)"
        """Calculate confidence based on all source agreement."""
        scores = [s for s in [ta, news, ml, smc] if s != 0]  # Exclude zero (no data)
        if not scores:
            return "LOW (no data)"
        
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        total = len(scores)

        # Numeric confidence
        if positive == total or negative == total:
            return f"HIGH ({positive}/{total} sources agree)"
        elif positive >= total * 0.6 or negative >= total * 0.6:
            return f"MEDIUM ({max(positive,negative)}/{total} sources agree)"
        else:
            return f"LOW ({max(positive,negative)}/{total} — sources diverge)"
