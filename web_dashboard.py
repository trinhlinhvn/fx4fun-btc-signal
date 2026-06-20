"""
Web Dashboard
Flask-based web interface for BTC Trading Signal Bot.
Features:
- Real-time signal display
- Price chart with indicators
- Expert analysis panel
- Signal history
- Auto-refresh via WebSocket
"""
import json
import threading
import time
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from signal_engine import SignalEngine

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "btc-signal-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
engine = SignalEngine()
signal_history = []
latest_signal = None


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/charts")
def charts():
    """Charts page with Bollinger Bands and MACD."""
    return render_template("charts.html")


@app.route("/about")
def about():
    """Architecture & documentation page."""
    return render_template("about.html")


@app.route("/api/signal")
def get_signal():
    """
    API endpoint to get current signal.
    Returns cached signal immediately. If no cache, triggers background generation.
    NEVER blocks — always returns fast.
    """
    global latest_signal

    # Always return immediately with whatever we have
    if latest_signal:
        return jsonify(latest_signal)

    # No signal yet — trigger generation in background and return loading state
    def gen():
        global latest_signal
        try:
            result = engine.generate_signal()
            latest_signal = result
            signal_history.append({
                "timestamp": result.get("timestamp"),
                "signal": result.get("signal"),
                "score": result.get("final_score"),
                "price": result.get("current_price", {}).get("price"),
            })
        except Exception as e:
            print(f"[SIGNAL] Generation error: {e}")

    threading.Thread(target=gen, daemon=True).start()

    return jsonify({
        "signal": "LOADING...",
        "final_score": 0,
        "confidence": "Loading...",
        "timestamp": datetime.now().isoformat(),
        "current_price": {"price": 0, "change_24h": 0},
        "components": {"technical_analysis": {"score": 0, "weight": 0, "weighted_score": 0, "details": {}},
                       "news_sentiment": {"score": 0, "weight": 0, "weighted_score": 0, "articles_analyzed": 0, "source": "loading", "top_articles": []},
                       "ml_prediction": {"score": 0, "weight": 0, "weighted_score": 0, "prediction": "N/A", "ml_confidence": 0, "models_used": [], "note": "Loading..."}},
        "expert_analysis": {"narratives": ["⏳ Loading analysis..."], "overall_bias": {"direction": "NEUTRAL", "strength": 0, "reasons": []}},
        "smc_ict": {"narratives": ["⏳ Loading SMC data..."], "score": 0},
        "risk_reward": {"position_type": "NO TRADE", "reason": "Loading...", "suggestion": "Đang phân tích..."},
        "market_context": {},
    })


@app.route("/api/history")
def get_history():
    """Get signal history."""
    return jsonify(signal_history[-50:])


@app.route("/api/train")
def train_models():
    """Train ML models."""
    result = engine.train_ml_models()
    return jsonify(result)


@app.route("/api/status")
def get_status():
    """Get bot status."""
    return jsonify({
        "status": "running",
        "ml_trained": engine.is_ml_trained,
        "signals_generated": len(signal_history),
        "last_update": latest_signal.get("timestamp") if latest_signal else None,
        "version": "3.0 Expert Edition",
    })


@app.route("/api/scan-coins")
def scan_coins_endpoint():
    """Scan top 10 coins, return top 3 strongest signals."""
    try:
        result = engine.scan_top_coins()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/backtest")
def backtest_endpoint():
    """Run backtest on historical data."""
    from backtester import Backtester
    try:
        bt = Backtester()
        result = bt.run_backtest(days=90, sl_pct=2.5, tp_pct=3.5, leverage=10)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/fear-greed")
def fear_greed_endpoint():
    """Get Fear & Greed Index."""
    try:
        return jsonify(engine.fear_greed.get_current())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/chart/<timeframe>")
def get_chart_data(timeframe):
    """
    Get OHLCV + BB + SMC zones for charts.
    Returns price, bollinger bands, order blocks, FVG, S/R levels.
    """
    from technical_analysis import TechnicalAnalyzer
    from data_fetcher import BTCDataFetcher
    from smc_ict_analysis import SMCICTAnalyzer
    from expert_analysis import ExpertAnalyzer

    ta = TechnicalAnalyzer()
    fetcher = BTCDataFetcher()
    smc = SMCICTAnalyzer()
    expert = ExpertAnalyzer()

    if timeframe == "h1":
        df = fetcher.get_h1_data(limit=100)
    elif timeframe == "m15":
        df = fetcher.get_klines(interval="15m", limit=100)
    else:
        df = fetcher.get_h4_data(limit=200)

    if df.empty:
        return jsonify({"error": "No data available", "timeframe": timeframe})

    df = ta.calculate_indicators(df)

    # SMC zones
    smc_data = smc.generate_smc_analysis(df)
    obs = smc_data.get("order_blocks", {})
    fvgs = smc_data.get("fair_value_gaps", {})
    pd_zone = smc_data.get("premium_discount", {})

    # S/R levels
    sr = expert.find_support_resistance(df, window=4 if timeframe == "h1" else 5)

    # Price data
    records = []
    for idx, row in df.iterrows():
        record = {
            "time": idx.isoformat(),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": round(float(row["volume"]), 2) if "volume" in row else 0,
        }
        if "bb_upper" in row and not pd.isna(row["bb_upper"]):
            record["bb_upper"] = round(float(row["bb_upper"]), 2)
            record["bb_middle"] = round(float(row["bb_middle"]), 2)
            record["bb_lower"] = round(float(row["bb_lower"]), 2)
        records.append(record)

    # SMC zones for chart overlay
    zones = {
        "bullish_obs": [{"high": ob["high"], "low": ob["low"], "strength": ob["strength"]} for ob in obs.get("bullish_obs", [])],
        "bearish_obs": [{"high": ob["high"], "low": ob["low"], "strength": ob["strength"]} for ob in obs.get("bearish_obs", [])],
        "bullish_fvgs": [{"top": f["top"], "bottom": f["bottom"]} for f in fvgs.get("bullish_fvgs", [])],
        "bearish_fvgs": [{"top": f["top"], "bottom": f["bottom"]} for f in fvgs.get("bearish_fvgs", [])],
        "support": [s["price"] for s in sr.get("support", [])[:3]],
        "resistance": [r["price"] for r in sr.get("resistance", [])[:3]],
        "equilibrium": pd_zone.get("equilibrium", 0),
        "premium_discount_zone": pd_zone.get("zone", "UNKNOWN"),
    }

    return jsonify({
        "timeframe": timeframe.upper(),
        "candles": len(records),
        "data": records,
        "smc_zones": zones,
    })


def background_updater():
    """Background thread to update signals periodically."""
    global latest_signal
    # Wait 60s before first background update (let chart load first)
    time.sleep(60)
    while True:
        try:
            result = engine.generate_signal()
            latest_signal = result
            signal_history.append({
                "timestamp": result.get("timestamp"),
                "signal": result.get("signal"),
                "score": result.get("final_score"),
                "price": result.get("current_price", {}).get("price"),
            })
            if len(signal_history) > 100:
                signal_history.pop(0)
            socketio.emit("signal_update", result)
        except Exception as e:
            print(f"[DASHBOARD] Background update error: {e}")
        from config import SCAN_INTERVAL_SECONDS
        time.sleep(SCAN_INTERVAL_SECONDS)


def run_dashboard(host="0.0.0.0", port=5000, debug=False):
    """Start the web dashboard."""
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()

    print(f"[DASHBOARD] Starting web dashboard at http://localhost:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))

    # Preload chart data FIRST (fast, cache for chart endpoint)
    print("[DASHBOARD] Preloading chart cache...")
    from data_fetcher import BTCDataFetcher
    _preloader = BTCDataFetcher()
    _preloader.get_h4_data(limit=200)
    _preloader.get_klines(interval="15m", limit=100)
    print("[DASHBOARD] Chart cache ready (H4 + M15).")

    # Start background signal updater (delayed 60s so charts work immediately)
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()

    print(f"[DASHBOARD] Server ready at http://localhost:{port}")
    print(f"[DASHBOARD] Charts load immediately. Signal loads in ~60s.")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
