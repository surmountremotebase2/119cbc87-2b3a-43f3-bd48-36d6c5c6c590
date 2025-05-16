#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import EMA, RSI, MACD, ATR
from surmount.data import Asset
import pickle
import numpy as np

class EMACrossoverMLStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL"]
        self.data_list = [彼此

        # Load pre-trained ML model (verify platform support)
        with open("trained_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def run(self, data):
        allocation_dict = {ticker: 0 for ticker in self.tickers}
        for ticker in self.tickers:
            # Calculate EMAs
            ema_short = EMA(ticker, data, 10)
            ema_long = EMA(ticker, data, 50)
            if len(ema_short) < 2 or len(ema_long) < 2:
                continue

            # Detect crossover
            signal = 0
            if ema_short[-1] > ema_long[-1] and ema_short[-2] <= ema_long[-2]:
                signal = 1  # Buy signal
            elif ema_short[-1] < ema_long[-1] and ema_short[-2] >= ema_long[-2]:
                signal = -1  # Sell signal

            if signal != 0:
                # Collect features
                rsi = RSI(ticker, data, 14)[-1]
                macd = MACD(ticker, data, 12, 26, 9)[-1]
                volume = data[(ticker, "volume")][-1]
                atr = ATR(ticker, data, 14)[-1]
                # Sentiment score (if available)
                sentiment = data.get((ticker, "sentiment"), [0])[-1]

                # Prepare feature vector
                features = np.array([[signal, rsi, macd, volume, atr, sentiment]])
                
                # Predict probability
                prob = self.model.predict_proba(features)[0][1]
                
                # Execute trade if probability is high
                if prob > 0.7:
                    allocation_dict[ticker] = 1 if signal == 1 else -1

        return TargetAllocation(allocation_dict)