from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import BB

class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]
        self.data_list = []

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
        buy_list = []
        for ticker in self.tickers:
            bb = BB(ticker, data['ohlcv'], length=20, std=2)
            if bb['lower'] and len(bb['lower']) > 0:
                current_close = data['ohlcv'][-1][ticker]['close']
                current_lower = bb['lower'][-1]
                if current_close < current_lower:
                    buy_list.append(ticker)
        if buy_list:
            N = len(buy_list)
            for ticker in buy_list:
                allocation_dict[ticker] = 1 / N
        return TargetAllocation(allocation_dict)#