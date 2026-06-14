"""
ML/AI Prediction Module
Uses LSTM + XGBoost ensemble to predict BTC price trend.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime


class FeatureEngineer:
    """Creates ML features from price data and technical indicators."""

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for ML model from OHLC + indicators.
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features["return_1"] = df["close"].pct_change(1)
        features["return_3"] = df["close"].pct_change(3)
        features["return_5"] = df["close"].pct_change(5)
        features["return_10"] = df["close"].pct_change(10)

        # Volatility
        features["volatility_5"] = df["close"].rolling(5).std() / df["close"].rolling(5).mean()
        features["volatility_10"] = df["close"].rolling(10).std() / df["close"].rolling(10).mean()

        # Price position relative to range
        features["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        features["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]

        # Volume proxy (using price range as volume substitute when volume unavailable)
        features["range_pct"] = (df["high"] - df["low"]) / df["low"]

        # Moving averages ratios
        features["ma5_ratio"] = df["close"] / df["close"].rolling(5).mean()
        features["ma10_ratio"] = df["close"] / df["close"].rolling(10).mean()
        features["ma20_ratio"] = df["close"] / df["close"].rolling(20).mean()

        # Momentum
        features["momentum_3"] = df["close"] - df["close"].shift(3)
        features["momentum_5"] = df["close"] - df["close"].shift(5)
        features["momentum_10"] = df["close"] - df["close"].shift(10)

        # RSI (if available)
        if "rsi" in df.columns:
            features["rsi"] = df["rsi"]
            features["rsi_change"] = df["rsi"].diff()

        # MACD (if available)
        if "macd" in df.columns:
            features["macd"] = df["macd"]
            features["macd_signal"] = df["macd_signal"]
            features["macd_hist"] = df["macd_histogram"]

        # EMA (if available)
        if "ema_short" in df.columns:
            features["ema_diff"] = (df["ema_short"] - df["ema_long"]) / df["close"]

        # Bollinger Bands (if available)
        if "bb_upper" in df.columns:
            features["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            features["bb_width"] = df["bb_width"]

        # Lag features (past values as features)
        for lag in [1, 2, 3, 5]:
            features[f"return_lag_{lag}"] = features["return_1"].shift(lag)

        return features

    @staticmethod
    def create_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.01) -> pd.Series:
        """
        Create target variable: future price direction.
        - 1 = price goes up > threshold
        - 0 = price stays flat
        - -1 = price goes down > threshold
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1

        target = pd.Series(0, index=df.index)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1

        return target


class XGBoostPredictor:
    """XGBoost-based trend predictor - fast and effective for tabular data."""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.model_path = "models/xgboost_btc.joblib"

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """Train XGBoost model with time-series cross-validation."""
        from xgboost import XGBClassifier

        # Clean data
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        # Remove last rows where target is unknown (future)
        mask = y != 0  # Keep only clear signals for training
        # Actually keep all, including 0 (flat) as a class
        X = X.values
        y_encoded = y.values + 1  # Convert -1,0,1 to 0,1,2 for classifier

        self.feature_names = features.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = accuracy_score(y_val, model.predict(X_val))
            scores.append(score)

        # Train final model on all data
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_scaled, y_encoded)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_path)

        return {
            "avg_accuracy": round(np.mean(scores), 4),
            "scores": [round(s, 4) for s in scores],
            "n_samples": len(X),
            "n_features": X.shape[1],
        }

    def predict(self, features: pd.DataFrame) -> dict:
        """Predict trend direction for the latest data point."""
        if self.model is None:
            if os.path.exists(self.model_path):
                saved = joblib.load(self.model_path)
                self.model = saved["model"]
                self.scaler = saved["scaler"]
            else:
                return {"prediction": 0, "confidence": 0, "error": "Model not trained"}

        # Get latest feature row
        latest = features.iloc[[-1]].dropna(axis=1)

        # Align features with training features
        if self.feature_names:
            missing = set(self.feature_names) - set(latest.columns)
            for col in missing:
                latest[col] = 0
            latest = latest[self.feature_names]

        X_scaled = self.scaler.transform(latest.values)

        # Get prediction and probability
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]

        # Convert back from 0,1,2 to -1,0,1
        direction = int(pred) - 1
        confidence = float(max(proba))

        return {
            "prediction": direction,  # -1=DOWN, 0=FLAT, 1=UP
            "confidence": round(confidence, 4),
            "probabilities": {
                "down": round(float(proba[0]), 4),
                "flat": round(float(proba[1]), 4),
                "up": round(float(proba[2]), 4),
            },
        }

    def get_feature_importance(self) -> dict:
        """Get top features by importance."""
        if self.model is None or self.feature_names is None:
            return {}

        importance = self.model.feature_importances_
        top_idx = np.argsort(importance)[-10:][::-1]

        return {
            self.feature_names[i]: round(float(importance[i]), 4)
            for i in top_idx
        }


class LSTMPredictor:
    """LSTM-based predictor for capturing sequential patterns."""

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = "models/lstm_btc.keras"
        self.tf_available = self._check_tensorflow()

    @staticmethod
    def _check_tensorflow():
        try:
            import tensorflow
            return True
        except ImportError:
            return False

    def _build_model(self, input_shape: tuple):
        """Build LSTM model architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(16, activation="relu"),
            Dropout(0.1),
            Dense(3, activation="softmax"),  # 3 classes: down, flat, up
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """Train LSTM model."""
        if not self.tf_available:
            return {"error": "TensorFlow not installed. Install with: pip install tensorflow"}

        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        # Clean data
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx].values
        y = target.loc[valid_idx].values + 1  # Convert -1,0,1 to 0,1,2

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) < 20:
            return {"error": "Not enough data for LSTM training", "n_samples": len(X_seq)}

        # Split (time-based, no shuffle)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Build and train
        self.model = self._build_model((self.sequence_length, X_train.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,
            verbose=0,
        )

        # Save
        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, "models/lstm_scaler.joblib")

        val_acc = history.history.get("val_accuracy", [0])[-1]

        return {
            "val_accuracy": round(float(val_acc), 4),
            "train_accuracy": round(float(history.history["accuracy"][-1]), 4),
            "epochs": 50,
            "n_samples": len(X_seq),
            "sequence_length": self.sequence_length,
        }

    def predict(self, features: pd.DataFrame) -> dict:
        """Predict using LSTM model."""
        if not self.tf_available:
            return {"prediction": 0, "confidence": 0, "error": "TensorFlow not installed"}

        if self.model is None:
            if os.path.exists(self.model_path):
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
                self.scaler = joblib.load("models/lstm_scaler.joblib")
            else:
                return {"prediction": 0, "confidence": 0, "error": "Model not trained"}

        # Get last N rows for sequence
        clean_features = features.dropna()
        if len(clean_features) < self.sequence_length:
            return {"prediction": 0, "confidence": 0, "error": "Not enough data for sequence"}

        last_rows = clean_features.iloc[-self.sequence_length:].values
        X_scaled = self.scaler.transform(last_rows)
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)

        # Predict
        proba = self.model.predict(X_seq, verbose=0)[0]
        pred = int(np.argmax(proba))

        direction = pred - 1  # Convert 0,1,2 back to -1,0,1
        confidence = float(max(proba))

        return {
            "prediction": direction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "down": round(float(proba[0]), 4),
                "flat": round(float(proba[1]), 4),
                "up": round(float(proba[2]), 4),
            },
        }


class EnsemblePredictor:
    """
    Combines XGBoost + LSTM predictions using weighted voting.
    This ensemble approach is more robust than any single model.
    """

    def __init__(self):
        self.xgb = XGBoostPredictor()
        self.lstm = LSTMPredictor(sequence_length=10)
        self.feature_engineer = FeatureEngineer()
        self.weights = {"xgboost": 0.55, "lstm": 0.45}  # XGB slightly more reliable

    def train(self, df: pd.DataFrame) -> dict:
        """Train both models on the same dataset."""
        features = self.feature_engineer.create_features(df)
        target = self.feature_engineer.create_target(df, horizon=3, threshold=0.005)

        print("[ML] Training XGBoost model...")
        xgb_result = self.xgb.train(features, target)

        print("[ML] Training LSTM model...")
        lstm_result = self.lstm.train(features, target)

        return {
            "xgboost": xgb_result,
            "lstm": lstm_result,
            "status": "trained",
            "timestamp": datetime.now().isoformat(),
        }

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Get ensemble prediction by combining both models.
        Returns a score from -1 (strong sell) to 1 (strong buy).
        """
        features = self.feature_engineer.create_features(df)

        xgb_pred = self.xgb.predict(features)
        lstm_pred = self.lstm.predict(features)

        # If one model has an error, use only the other
        if "error" in xgb_pred and "error" in lstm_pred:
            return {
                "score": 0,
                "prediction": "UNKNOWN",
                "confidence": 0,
                "error": "Both models failed",
            }

        if "error" in lstm_pred:
            ensemble_score = xgb_pred["prediction"] * xgb_pred["confidence"]
            confidence = xgb_pred["confidence"]
            models_used = ["xgboost"]
        elif "error" in xgb_pred:
            ensemble_score = lstm_pred["prediction"] * lstm_pred["confidence"]
            confidence = lstm_pred["confidence"]
            models_used = ["lstm"]
        else:
            # Weighted ensemble
            xgb_score = xgb_pred["prediction"] * xgb_pred["confidence"]
            lstm_score = lstm_pred["prediction"] * lstm_pred["confidence"]
            ensemble_score = (
                xgb_score * self.weights["xgboost"] +
                lstm_score * self.weights["lstm"]
            )
            confidence = (
                xgb_pred["confidence"] * self.weights["xgboost"] +
                lstm_pred["confidence"] * self.weights["lstm"]
            )
            models_used = ["xgboost", "lstm"]

        # Determine direction
        if ensemble_score > 0.15:
            prediction = "UP"
        elif ensemble_score < -0.15:
            prediction = "DOWN"
        else:
            prediction = "FLAT"

        return {
            "score": round(ensemble_score, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "models_used": models_used,
            "xgboost_result": xgb_pred,
            "lstm_result": lstm_pred,
            "feature_importance": self.xgb.get_feature_importance(),
        }
