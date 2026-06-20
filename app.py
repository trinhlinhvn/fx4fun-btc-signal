"""
Entry point for production deployment (Render/Gunicorn).
"""
import threading
from web_dashboard import app, socketio, background_updater, engine, latest_signal
import web_dashboard

# Preload first signal on startup so frontend doesn't wait
def preload_signal():
    try:
        print("[APP] Preloading first signal...")
        result = engine.generate_signal()
        web_dashboard.latest_signal = result
        web_dashboard.signal_history.append({
            "timestamp": result.get("timestamp"),
            "signal": result.get("signal"),
            "score": result.get("final_score"),
            "price": result.get("current_price", {}).get("price"),
        })
        print(f"[APP] Preload done: {result.get('signal')}")
    except Exception as e:
        print(f"[APP] Preload error: {e}")

# Start background updater
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# Preload in background (don't block gunicorn startup)
preload_thread = threading.Thread(target=preload_signal, daemon=True)
preload_thread.start()
