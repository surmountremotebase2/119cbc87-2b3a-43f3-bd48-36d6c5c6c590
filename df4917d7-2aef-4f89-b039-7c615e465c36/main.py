#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import BB, RSI
from surmount.data import InsiderTrading
from datetime import datetime, timedelta

class MeanReversionStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL", "GOOG", "MSFT", "AMZN"]
        self.data_list = [
            BB(ticker, period=20, std=2) for ticker in self.tickers
        ] + [
            RSI(ticker, period=14) for ticker in self.tickers
        ] + [
            InsiderTrading(ticker) for ticker in self.tickers
        ]

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
        target_allocation = TargetAllocation()
        for ticker in self.tickers:
            bb_data = data.get((f"bb_{ticker}", ticker))
            rsi_data = data.get((f"rsi_{ticker}", ticker))
            insider_data = data.get((f"insider_{ticker}", ticker))

            if bb_data and rsi_data and insider_data:
                current_price = data.get(("ohlcv", ticker))["close"][-1]
                lower_band = bb_data["lower_band"][-1]
                upper_band = bb_data["upper_band"][-1]
                rsi = rsi_data["rsi"][-1]
                recent_insider_buys = sum(1 for trade in insider_data["transactions"] if trade["transactionType"] == "Buy" and trade["date"] > (datetime.now() - timedelta(days=30)))

                if current_price <= lower_band and rsi < 30 and recent_insider_buys > 0:
                    target_allocation[ticker] = 0.25  # Equal weight

        return target_allocationType code here