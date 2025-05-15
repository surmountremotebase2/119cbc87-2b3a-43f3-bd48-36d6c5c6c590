#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import BB, RSI, ATR, SMA

class CombinedStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]
        self.data_list = []
        self.positions = {}  # Track mean reversion positions

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
        signals = {ticker: 0 for ticker in self.tickers}

        # Mean Reversion (with RSI and exit)
        for ticker in self.tickers:
            bb = BB(ticker, data['ohlcv'], length=20, std=2)
            rsi = RSI(ticker, data['ohlcv'], length=14)
            if bb['lower'] and rsi and len(bb['lower']) > 0 and len(rsi) > 0:
                current_close = data['ohlcv'][-1][ticker]['close']
                current_lower = bb['lower'][-1]
                current_rsi = rsi[-1]
                if current_close < current_lower and current_rsi < 30 and ticker not in self.positions:
                    signals[ticker] += 1
                    self.positions[ticker] = current_close
            if ticker in self.positions and bb['middle'] and len(bb['middle']) > 0:
                if current_close > bb['middle'][-1]:
                    del self.positions[ticker]

        # Momentum (SMA Crossover)
        for ticker in self.tickers:
            sma_short = SMA(ticker, data['ohlcv'], length=50)
            sma_long = SMA(ticker, data['ohlcv'], length=200)
            if sma_short and sma_long and len(sma_short) > 1 and len(sma_long) > 1:
                FFif sma_short[-1] > sma_long[-1] and sma_short[-2] <= sma_long[-2]:
                signals[ticker] += 1

        # Sentiment (Volume Spike using raw OHLCV volume)
        for ticker in self.tickers:
            if len(data['ohlcv']) > 20:
                volumes = [data['ohlcv'][i][ticker]['volume'] for i in range(-20, 0)]
                avg_volume = sum(volumes[:-1]) / 19  # Exclude current day
                current_volume = data['ohlcv'][-1][ticker]['volume']
                current_data = data['ohlcv'][-1][ticker]
                if current_volume > 2 * avg_volume and current_data['close'] > current_data['open']:
                    signals[ticker] += 1

        # Allocate based on signal strength
        total_signals = sum(signals.values())
        if total_signals > 0:
            for ticker in self.tickers:
                allocation_dict[ticker] = signals[ticker] / total_signals

        return TargetAllocation(allocation_dict)