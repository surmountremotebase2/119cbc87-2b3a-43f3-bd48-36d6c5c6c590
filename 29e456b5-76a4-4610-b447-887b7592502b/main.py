#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, ATR

class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]
        self.data_list = []
        self.positions = {}  # Track open positions {ticker: entry_price}

    @property
    def interval(self):
        return "1hour"  # Intraday trading

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def run(self, data):
        allocation_dict = {ticker: 0 for ticker in self.tickers}
        signals = {ticker: 0 for ticker in self.tickers}

        for ticker in self.tickers:
            if len(data['ohlcv']) < 21:  # Need at least 20 periods + 1 for prev day
                continue

            # Get previous day's high (assuming 6.5-hour trading day)
            prev_day_high = max(data['ohlcv'][i][ticker]['high'] for i in range(-13, -7))  # Last 6 hours of prev day
            current_data = data['ohlcv'][-1][ticker]
            current_close = current_data['close']
            current_volume = current_data['volume']

            # Volume confirmation (20-period average)
            volumes = [data['ohlcv'][i][ticker]['volume'] for i in range(-20, 0)]
            avg_volume = sum(volumes[:-1]) / 19  # Exclude current period
            volume_spike = current_volume > 2 * avg_volume

            # RSI confirmation
            rsi = RSI(ticker, data['ohlcv'], length=14)
            rsi_valid = rsi and len(rsi) > 0 and rsi[-1] > 60

            # ATR for profit target and stop-loss
            atr = ATR(ticker, data['ohlcv'], length=14)
            atr_valid = atr and len(atr) > 0
            atr_value = atr[-1] if atr_valid else 1.0  # Fallback to avoid division by zero

            # Breakout signal
            if (current_close > prev_day_high and volume_spike and rsi_valid and
                ticker not in self.positions):
                signals[ticker] += 1
                self.positions[ticker] = current_close

            # Exit conditions: profit target, stop-loss, or end of day
            if ticker in self.positions:
                entry_price = self.positions[ticker]
                profit_target = entry_price + 2 * atr_value
                stop_loss = entry_price - atr_value
                # Approximate end of day: last hour of trading (e.g., after 3 PM)
                current_hour = int(data['ohlcv'][-1][ticker]['date'].split('T')[1][:2])
                is_end_of_day = current_hour >= 15  # Adjust based on market hours
                if (current_close >= profit_target or current_close <= stop_loss or
                    is_end_of_day):
                    del self.positions[ticker]

        # Allocate based on signal strength
        total_signals = sum(signals.values())
        if total_signals > 0:
            for ticker in self.tickers:
                allocation_dict[ticker] = min(signals[ticker] / total_signals, 0.2)  # Cap at 20%

        return TargetAllocation(allocation_dict)