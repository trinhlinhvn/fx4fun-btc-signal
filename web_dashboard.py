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


@app.route("/api/signal")
def get_signal():
    """API endpoint to get current signal. Uses cache to avoid rate limits."""
    global latest_signal
    # Return cached signal if less than 60 seconds old
    if latest_signal and latest_signal.get("timestamp"):
        from datetime import datetime as dt
        try:
            last_time = dt.fromisoformat(latest_signal["timestamp"])
            if (dt.now() - last_time).total_seconds() < 60:
                return jsonify(latest_signal)
        except (ValueError, TypeError):
            pass

    result = engine.generate_signal()
    latest_signal = result
    signal_history.append({
        "timestamp": result.get("timestamp"),
        "signal": result.get("signal"),
        "score": result.get("final_score"),
        "price": result.get("current_price", {}).get("price"),
    })
    # Keep only last 100 signals
    if len(signal_history) > 100:
        signal_history.pop(0)
    return jsonify(result)


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
        "version": "2.0 Expert Edition",
    })


@app.route("/api/chart/<timeframe>")
def get_chart_data(timeframe):
    """
    Get OHLCV + indicators data for charts.
    timeframe: 'h1' or 'h4'
    """
    from technical_analysis import TechnicalAnalyzer
    from data_fetcher import BTCDataFetcher

    fetcher = BTCDataFetcher()
    ta = TechnicalAnalyzer()

    if timeframe == "h1":
        df = fetcher.get_h1_data(limit=100)
    else:
        df = fetcher.get_h4_data(limit=200)

    if df.empty:
        return jsonify({"error": "No data available"})

    df = ta.calculate_indicators(df)

    # Convert to JSON-serializable format
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
        # Bollinger Bands
        if "bb_upper" in row and not pd.isna(row["bb_upper"]):
            record["bb_upper"] = round(float(row["bb_upper"]), 2)
            record["bb_middle"] = round(float(row["bb_middle"]), 2)
            record["bb_lower"] = round(float(row["bb_lower"]), 2)
        # MACD
        if "macd" in row and not pd.isna(row["macd"]):
            record["macd"] = round(float(row["macd"]), 2)
            record["macd_signal"] = round(float(row["macd_signal"]), 2)
            record["macd_histogram"] = round(float(row["macd_histogram"]), 2)

        records.append(record)

    return jsonify({
        "timeframe": timeframe.upper(),
        "candles": len(records),
        "data": records,
    })


def background_updater():
    """Background thread to update signals periodically."""
    global latest_signal
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
            # Emit to connected clients
            socketio.emit("signal_update", result)
        except Exception as e:
            print(f"[DASHBOARD] Background update error: {e}")
        time.sleep(300)  # Update every 5 minutes


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
    run_dashboard(port=port, debug=True)
