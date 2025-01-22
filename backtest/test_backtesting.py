import backtrader as bt
import pandas as pd

# Define the Strategy
class CustomStrategy(bt.Strategy):
    params = (
        ("lookback", 15),  # Lookback period for mean and stddev in minutes
        ("std_multiplier", 2),  # Multiplier for standard deviation
        ("macd_fast", 12),  # MACD fast EMA period
        ("macd_slow", 26),  # MACD slow EMA period
        ("macd_signal", 9),  # MACD signal line period
        ("take_profit", 0.03),  # Take profit level (3%)
        ("stop_loss", 0.015),  # Stop loss level (1.5%)
    )

    def __init__(self):
        # Indicators
        self.data_close = self.datas[0].close

        # Rolling Mean and StdDev
        self.rolling_mean = bt.indicators.SimpleMovingAverage(self.data_close, period=self.params.lookback)
        self.rolling_std = bt.indicators.StandardDeviation(self.data_close, period=self.params.lookback)

        # MACD
        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )

        # Track entry price
        self.entry_price = None

    def next(self):
        # Calculate conditions for the buy signal
        price_rise = self.data_close[0] > self.rolling_mean[0] + self.params.std_multiplier * self.rolling_std[0]
        macd_condition = self.macd.macd[0] > self.macd.signal[0]

        # Check if we already have a position
        if not self.position:
            # If conditions are met, enter a buy trade
            if price_rise and macd_condition:
                self.entry_price = self.data_close[0]
                self.buy(size=0.00001)  # Adjust size as per your preference
                print(f"BUY executed: {self.data.datetime.datetime(0)}, Price: {self.data_close[0]}")
        else:
            # Check for take profit or stop loss
            if self.entry_price:
                tp_price = self.entry_price * (1 + self.params.take_profit)
                sl_price = self.entry_price * (1 - self.params.stop_loss)

                if self.data_close[0] >= tp_price:
                    self.sell(size=1)
                    print(f"TAKE PROFIT executed: {self.data.datetime.datetime(0)}, Price: {self.data_close[0]}")
                    self.entry_price = None  # Reset entry price
                elif self.data_close[0] <= sl_price:
                    self.sell(size=1)
                    print(f"STOP LOSS executed: {self.data.datetime.datetime(0)}, Price: {self.data_close[0]}")
                    self.entry_price = None  # Reset entry price

# Load CSV Data
class PandasData(bt.feeds.PandasData):
    params = (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
    )

# Main Function
if __name__ == "__main__":
    # Load the data into a Pandas DataFrame
    data_file = "D:\\Boobogo\\coding\\trade\\data\\us100\\data_engineered\\US_TECH100_1d_indicators.csv"  # Replace with your file path
    df = pd.read_csv(data_file, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # Rename columns to match expected names
    df.rename(columns={
        'open_1d': 'open',
        'high_1d': 'high',
        'low_1d': 'low',
        'close_1d': 'close'
    }, inplace=True)

    # Print the first few rows of the dataframe for debugging
    print(df.head())

    # Initialize the Backtrader system
    cerebro = bt.Cerebro()

    # Add the custom strategy
    cerebro.addstrategy(CustomStrategy)

    # Load the data into Backtrader
    data = PandasData(dataname=df)
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.setcash(10000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0001)

    # Run the strategy
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Plot the results without volume
    cerebro.plot(volume=False)