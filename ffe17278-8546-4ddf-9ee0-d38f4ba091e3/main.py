#Ty<antArtifact None>
import surmount
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main(data, capital):
    """
    Main function as required by Surmount API
    
    Parameters:
    data - Dictionary of dataframes for each ticker
    capital - Starting capital for the strategy
    
    Returns:
    List of dictionaries containing trades
    """
    # Initialize algorithm
    algorithm = AITradingAlgorithm(capital)
    
    # Add equities to the algorithm
    for ticker in algorithm.universe:
        if ticker in data:
            algorithm.data[ticker] = data[ticker]
    
    # Run algorithm
    trades = algorithm.run()
    
    return trades

class AITradingAlgorithm:
    def __init__(self, capital):
        # Initial capital
        self.capital = capital
        self.portfolio_value = capital
        
        # Algorithm parameters
        self.lookback_window = 30  # Days of historical data to consider
        self.universe = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]  # Stock universe
        self.position_size = 0.1  # Maximum position size per stock (10% of portfolio)
        self.max_positions = 3  # Maximum number of concurrent positions
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        
        # Technical indicators parameters
        self.fast_ma = 5
        self.slow_ma = 20
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Data storage
        self.data = {}
        
        # Trading state
        self.current_positions = {}  # Track current positions {ticker: quantity}
        self.current_cash = capital
        self.trades = []
        
        # Rebalancing
        self.rebalance_frequency = 7  # Trading days between rebalances
        self.last_rebalance_date = None
    
    def process_data(self, ticker):
        """Process data for a single ticker to create features"""
        if ticker not in self.data:
            return None
            
        df = self.data[ticker].copy()
        
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
        df['sma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
        df['ma_crossover'] = df['sma_fast'] - df['sma_slow']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 0.001)  # Avoid division by zero
        
        # Trend features
        df['price_trend'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(5).replace(0, 0.001) - 1
        
        # Mean reversion potential
        df['distance_from_ma'] = (df['close'] / df['sma_slow'].replace(0, 0.001) - 1) * 100
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def generate_features(self, df):
        """Generate features for algorithmic trading decision"""
        features = {}
        
        # Basic price/volume features
        features['close'] = df['close'].iloc[-1]
        features['volume'] = df['volume'].iloc[-1]
        
        # Technical indicators
        features['rsi'] = df['rsi'].iloc[-1]
        features['ma_crossover'] = df['ma_crossover'].iloc[-1]
        features['momentum'] = df['momentum'].iloc[-1]
        features['volatility'] = df['volatility'].iloc[-1]
        features['distance_from_ma'] = df['distance_from_ma'].iloc[-1]
        features['volume_ratio'] = df['volume_ratio'].iloc[-1]
        
        # Simple trend signals
        features['golden_cross'] = 1 if (df['sma_fast'].iloc[-1] > df['sma_slow'].iloc[-1] and 
                                        df['sma_fast'].iloc[-2] <= df['sma_slow'].iloc[-2]) else 0
        features['death_cross'] = 1 if (df['sma_fast'].iloc[-1] < df['sma_slow'].iloc[-1] and 
                                       df['sma_fast'].iloc[-2] >= df['sma_slow'].iloc[-2]) else 0
        
        # Overbought/oversold signals
        features['overbought'] = 1 if features['rsi'] > self.rsi_overbought else 0
        features['oversold'] = 1 if features['rsi'] < self.rsi_oversold else 0
        
        # Trend strength
        features['price_trend'] = df['price_trend'].iloc[-1]
        features['trend_strength'] = abs(df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
        
        return features
    
    def analyze_sentiment(self, features):
        """Analyze technical sentiment (simplified approach)"""
        # In a real implementation, this would fetch news and social media data
        # Here we'll use technical indicators as a proxy for sentiment
        
        sentiment = 0.5  # Neutral starting point
        
        # RSI contribution (higher RSI = more bullish)
        rsi = features['rsi']
        if rsi > 70:
            sentiment += 0.2
        elif rsi < 30:
            sentiment -= 0.2
        else:
            sentiment += (rsi - 50) / 100
        
        # Moving average crossover
        if features['golden_cross']:
            sentiment += 0.15
        elif features['death_cross']:
            sentiment -= 0.15
        
        # Price trend
        sentiment += features['price_trend'] * 0.05
        
        # Momentum
        sentiment += features['momentum'] * 5  # Scale up since momentum is usually small
        
        # Distance from moving average (mean reversion)
        dist = features['distance_from_ma']
        if dist > 10:  # Significantly above MA
            sentiment -= 0.05  # Potential mean reversion (bearish)
        elif dist < -10:  # Significantly below MA
            sentiment += 0.05  # Potential mean reversion (bullish)
        
        # Final normalization to 0-1 range
        sentiment = max(0, min(1, sentiment))
        
        return sentiment
    
    def detect_market_regime(self, features):
        """Detect the current market regime"""
        volatility = features['volatility']
        trend_strength = features['trend_strength']
        
        # Determine regime
        if volatility > 0.02:  # High volatility
            return "volatile"
        elif trend_strength > 0.1:  # Strong trend
            return "trending"
        else:  # Likely mean-reverting
            return "mean_reverting"
    
    def calculate_position_size(self, confidence, volatility, regime):
        """Calculate position size based on prediction confidence and volatility"""
        # Base position size
        position = self.position_size * confidence
        
        # Adjust for volatility (reduce position in high volatility)
        volatility_factor = 1.0 / (1.0 + volatility * 5)
        
        # Adjust for regime
        regime_factor = 1.0
        if regime == "volatile":
            regime_factor = 0.7  # Reduce size in volatile markets
        elif regime == "trending":
            regime_factor = 1.2  # Increase size in trending markets
        
        # Calculate final position size
        final_position = position * volatility_factor * regime_factor
        
        # Cap at maximum position size
        return min(final_position, self.position_size)
    
    def generate_signals(self):
        """Generate trading signals for all tickers in universe"""
        signals = {}
        
        for ticker in self.universe:
            if ticker not in self.data:
                continue
                
            # Process data for features
            processed_data = self.process_data(ticker)
            if processed_data is None or len(processed_data) < self.lookback_window:
                continue
                
            # Extract features
            features = self.generate_features(processed_data)
            
            # Generate simple predictive signal (combination of indicators)
            signal_value = 0
            
            # RSI component
            if features['rsi'] < 30:  # Oversold
                signal_value += 0.3
            elif features['rsi'] > 70:  # Overbought
                signal_value -= 0.3
            
            # Moving average component
            if features['ma_crossover'] > 0:  # Fast MA above slow MA
                signal_value += 0.2
            else:  # Fast MA below slow MA
                signal_value -= 0.2
            
            # Volume component
            if features['volume_ratio'] > 1.5:  # High volume
                signal_value += 0.1 * np.sign(features['momentum'])  # Amplify current momentum
            
            # Momentum component
            signal_value += features['momentum'] * 3
            
            # Mean reversion component
            if abs(features['distance_from_ma']) > 15:
                signal_value -= np.sign(features['distance_from_ma']) * 0.2  # Counteract extreme deviations
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(features)
            
            # Detect market regime
            regime = self.detect_market_regime(features)
            
            # Adjust signal based on sentiment
            prediction_weight = 0.7
            sentiment_weight = 0.3
            
            # Convert sentiment from 0-1 to -1 to 1
            norm_sentiment = (sentiment - 0.5) * 2
            
            # Combine for final signal
            final_signal = (prediction_weight * signal_value + 
                           sentiment_weight * norm_sentiment)
            
            # Calculate confidence (absolute value of signal)
            confidence = min(1.0, abs(final_signal))
            
            # Determine direction
            direction = 1 if final_signal > 0 else -1
            
            # Calculate position size
            position_size = self.calculate_position_size(
                confidence, 
                features['volatility'],
                regime
            )
            
            signals[ticker] = {
                'direction': direction,
                'confidence': confidence,
                'position_size': position_size,
                'regime': regime,
                'price': features['close'],
                'predicted_signal': signal_value,
                'sentiment': sentiment
            }
        
        return signals
    
    def should_rebalance(self, current_date):
        """Determine if we should rebalance portfolio"""
        if self.last_rebalance_date is None:
            return True
        
        # Calculate trading days between dates
        # This is a simplified calculation - in real usage you'd count actual trading days
        days_diff = (current_date - self.last_rebalance_date).days
        trading_days = max(1, int(days_diff * 5/7))  # Approximate trading days
        
        return trading_days >= self.rebalance_frequency
    
    def execute_trades(self, signals, current_date):
        """Execute trades based on signals"""
        # Sort signals by confidence
        sorted_signals = sorted(
            [(ticker, details) for ticker, details in signals.items()],
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        # Close positions that are no longer top signals
        current_tickers = set(self.current_positions.keys())
        new_top_tickers = set([ticker for ticker, _ in sorted_signals[:self.max_positions]])
        
        for ticker in current_tickers - new_top_tickers:
            # Get current position
            position = self.current_positions[ticker]
            current_price = signals[ticker]['price'] if ticker in signals else self.data[ticker]['close'].iloc[-1]
            
            # Add closing trade
            trade = {
                "ticker": ticker,
                "type": "sell" if position > 0 else "buy",  # close a short position with a buy
                "quantity": abs(position),
                "price": current_price,
                "date": current_date
            }
            self.trades.append(trade)
            
            # Update cash
            self.current_cash += position * current_price
            
            # Remove from current positions
            del self.current_positions[ticker]
        
        # Open or adjust positions for top signals
        positions_to_open = min(self.max_positions, len(sorted_signals))
        
        for i in range(positions_to_open):
            if i >= len(sorted_signals):
                break
                
            ticker, details = sorted_signals[i]
            current_price = details['price']
            
            # Calculate target position
            direction = details['direction']
            position_size = details['position_size']
            target_value = self.portfolio_value * position_size * direction
            target_quantity = int(target_value / current_price)  # Round to whole shares
            
            # Get current position
            current_quantity = self.current_positions.get(ticker, 0)
            
            # Only trade if there's a significant difference
            if abs(target_quantity - current_quantity) > 0:
                # Determine trade type and quantity
                if target_quantity > current_quantity:
                    trade_type = "buy"
                    trade_quantity = target_quantity - current_quantity
                else:
                    trade_type = "sell"
                    trade_quantity = current_quantity - target_quantity
                
                # Add trade
                trade = {
                    "ticker": ticker,
                    "type": trade_type,
                    "quantity": trade_quantity,
                    "price": current_price,
                    "date": current_date
                }
                self.trades.append(trade)
                
                # Update cash
                if trade_type == "buy":
                    self.current_cash -= trade_quantity * current_price
                else:
                    self.current_cash += trade_quantity * current_price
                
                # Update positions
                self.current_positions[ticker] = target_quantity
    
    def update_portfolio_value(self):
        """Calculate current portfolio value"""
        # Start with cash
        portfolio_value = self.current_cash
        
        # Add value of all positions
        for ticker, quantity in self.current_positions.items():
            if ticker in self.data:
                current_price = self.data[ticker]['close'].iloc[-1]
                position_value = quantity * current_price
                portfolio_value += position_value
        
        self.portfolio_value = portfolio_value
    
    def run(self):
        """Run the algorithm and return trades"""
        # Find the common dates across all data
        common_dates = None
        for ticker, df in self.data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # Need enough data for lookback
        if len(common_dates) < self.lookback_window:
            return self.trades
        
        # Iterate through trading days
        for i in range(self.lookback_window, len(common_dates)):
            current_date = common_dates[i]
            
            # Check if we need to rebalance
            if not self.should_rebalance(current_date):
                continue
            
            self.last_rebalance_date = current_date
            
            # Generate signals
            signals = self.generate_signals()
            
            # Execute trades
            self.execute_trades(signals, current_date)
            
            # Update portfolio value
            self.update_portfolio_value()
        
        return self.trades

</antArtifact>
pe code here