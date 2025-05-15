#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import BB, RSI
from surmount.data import InsiderTrading
from datetime import datetime, timedelta

class TradingStrategy(Strategy):
    def __init__(self):
        # Define tickers to trade
        self.tickers = ["AAPL", "GOOG", "MSFT", "AMZN"]
        # Initialize data sources: Bollinger Bands, RSI, and Insider Trading
        self.data_list = [
            BB(ticker, period=20, std=2) for ticker in self.tickers
        ] + [
            RSI(ticker, period=14) for ticker in self.tickers
        ] + [
            InsiderTrading(ticker) for ticker in self.tickers
        ]

    @property
    def interval(self):
        # Daily data for backtesting
        return "1day"

    @property
    def assets(self):
        # Return list of tickers
        return self.tickers

    @property
    def data(self):
        # Return data sources
        return self.data_list

    def run(self, data):
        # Initialize TargetAllocation object
        target_allocation = TargetAllocation()
        
        for ticker in self.tickers:
            # Access data for Bollinger Bands, RSI, and Insider Trading
            bb_data = data.get((f"bb_{ticker}", ticker))
            rsi_data = data.get((f"rsi_{ticker}", ticker))
            insider_data = data.get((f"insider_{ticker}", ticker))
            ohlcv_data = data.get(("ohlcv", ticker))

            # Ensure all required data is available
            if bb_data and rsi_data and insider_data and ohlcv_data:
                current_price = ohlcv_data["close"][-1]
                lower_band = bb_data["lower_band"][-1]
                upper_band = bb_data["upper_band"][-1]
                rsi = rsi_data["rsi"][-1]
                
                # Count recent insider buys (within 30 days)
                # Note: Temporarily remove insider trading condition for 1993-1995 backtest
                # recent_insider_buys = sum(
                #     1 for trade in insider_data["transactions"]
                #     if trade["transactionType"] == "Buy" and
                #     trade["date"] > (datetime.now() - timedelta(days=30))
                # )

                # Buy signal: price near lower band, RSI oversold
                # Modified to exclude insider trading due to date incompatibility
                if current_price <= lower_band and rsi < 30:
                    target_allocation[ticker] = 0.25  # Equal weight for simplicity

        # Return the computed allocations
        return target_allocation