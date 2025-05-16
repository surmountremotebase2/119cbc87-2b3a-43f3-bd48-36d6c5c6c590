#from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, ATR, SMA

class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]
        self.data_list = []
        self.positions = {}  # Track open positions {ticker: entry_price}
        self.bars_per_day = 7  # Assuming 7 bars per day for 1-hour intervals

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

            # Get previous day's high (full day, 7 bars)
            prev_day_start = len(data['ohlcv']) - (self.bars_per_day * 2)
            prev_day_end = len(data['ohlcv']) - self.bars_per_day
            prev_day_high = max(data['ohlcv'][i][ticker]['high'] for i in range(prev_day_start, prev_day_end))

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

            # Trend filter using 50-period SMA
            sma = SMA(ticker, data['ohlcv'], length=50)
            trend_valid = sma and len(sma) > 0 and current_close > sma[-1]

            # ATR for profit target and stop-loss
            atr = ATR(ticker, data['ohlcv'], length=14)
            atr_valid = atr and len(atr) > 0
            atr_value = atr[-1] if atr_valid else 1.0  # Fallback to avoid division by zero

            # Time filter for entries (9:00-11:00 AM, 2:00-4:00 PM)
            try:
                date_str = data['ohlcv'][-1][ticker]['date']
                time_part = date_str.split(' ')[1] if ' ' in date_str else date_str.split('T')[1]
                current_hour = int(time_part[:2])
                allow_entry = current_hour in [9, 10, 14, 15]
            except (IndexError, ValueError):
                allow_entry = False

            # Breakout signal
            if (allow_entry and current_close > prev_day_high and volume_spike and
                rsi_valid and trend_valid and ticker not in self.positions):
                signals[ticker] += 1
                self.positions[ticker] = current_close

            # Exit conditions: profit target, stop-loss, or end of day
            if ticker in self.positions:
                entry_price = self.positions[ticker]
                profit_target = entry_price + 1.5 * atr_value
                stop_loss = entry_price - 0.75 * atr_value
                try:
                    date_str = data['ohlcv'][-1][ticker]['date']
                    time_part = date_str.split(' ')[1] if ' ' in date_str else date_str.split('T')[1]
                    current_hour = int(time_part[:2])
                    is_end_of_day = current_hour >= 15  # Close positions at 3 PM or later
                except (IndexError, ValueError):
                    is_end_of_day = False
                if (current_close >= profit_target or current_close <= stop_loss or
                    is_end_of_day):
                    del self.positions[ticker]

        # Allocate based on signal strength
        total_signals = sum(signals.values())
        if total_signals > 0:
            for ticker in self.tickers:
                allocation_dict[ticker] = min(signals[ticker] / total_signals, 0.2)  # Cap at 20%

        return TargetAllocation(allocation_dict)