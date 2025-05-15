#import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data (via Surmount AI or external API)
data = pd.read_csv("stock_data.csv")  # Price, volume, sentiment
data["ma50"] = data["close"].rolling(50).mean()
data["rsi"] = compute_rsi(data["close"], 14)

# Momentum Model (LSTM)
def build_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(lookback, features)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

# Mean Reversion Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100)

# Train models
lstm_model = build_lstm()
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10)
rf_model.fit(X_train_rf, y_train_rf)

# Generate signals
def generate_signals(data):
    momentum_signal = lstm_model.predict(data["features"])
    reversion_signal = rf_model.predict(data["features"])
    combined_signal = 0.6 * momentum_signal + 0.4 * reversion_signal
    return "buy" if combined_signal > 0.7 else "sell" if combined_signal < 0.3 else "hold"

# Execute trades on Surmount AI
for i in range(len(data)):
    signal = generate_signals(data.iloc[i])
    if signal == "buy" and portfolio.cash > 0:
        execute_trade("buy", data["close"][i], size=0.02 * portfolio.equity)
    elif signal == "sell" and portfolio.positions > 0:
        execute_trade("sell", data["close"][i], size=0.02 * portfolio.equity)