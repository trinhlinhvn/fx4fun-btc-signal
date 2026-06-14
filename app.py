"""
Entry point for production deployment (Render/Gunicorn).
Render looks for app:app by default.
"""
from web_dashboard import app, socketio, background_updater
import threading

# Start background updater
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# Export 'app' for gunicorn
# Run with: gunicorn --worker-class eventlet -w 1 app:app
