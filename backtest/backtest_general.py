import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD

TP = 0.1
SL = 0.05

# Load the financial data
data = pd.read_csv('D:\\Boobogo\\coding\\trade\\data\\us100\\data_engineered\\US_TECH100_1d_indicators.csv', parse_dates=True, index_col='datetime')
data.rename(columns={'close_1d': 'close'}, inplace=True)
data = data[['close']].copy()

# Calculate daily returns
data['daily_return'] = data['close'].pct_change()
data.dropna(inplace=True)

# Calculate 5-day volatility
data['volatility'] = data['daily_return'].rolling(window=5).std()
data.dropna(inplace=True)

# Calculate MACD and signal line
macd = MACD(data['close'])
data['macd'] = macd.macd()
data['macd_signal'] = macd.macd_signal()
data.dropna(inplace=True)

# Define trading conditions
buy_condition = (data['daily_return'].shift(1).abs() > 2 * data['volatility'].shift(1)) & (data['macd'].shift(1) > data['macd_signal'].shift(1))

# Initialize list to store trades
trades = []

# Simulate trades
for i in range(1, len(data)):  # Start from 1 to ensure previous day's data is available
    # Check for buy condition
    if buy_condition.iloc[i]:
        trade = {
            'entry_date': data.index[i],
            'entry_price': data['close'].iloc[i],
            'take_profit': data['close'].iloc[i] * (1 + TP),
            'stop_loss': data['close'].iloc[i] * (1 - SL),
            'magic_number': len(trades) + 1,
            'status': 'open'
        }
        trades.append(trade)
    
    # Check for take profit and stop loss conditions for each trade
    for trade in trades:
        if trade['status'] == 'open':
            if data['close'].iloc[i] >= trade['take_profit']:
                trade['exit_date'] = data.index[i]
                trade['exit_price'] = trade['take_profit']
                trade['return'] = TP
                trade['status'] = 'closed'
            elif data['close'].iloc[i] <= trade['stop_loss']:
                trade['exit_date'] = data.index[i]
                trade['exit_price'] = trade['stop_loss']
                trade['return'] = -SL
                trade['status'] = 'closed'

# Calculate cumulative returns
data['trade_return'] = 0.0
for trade in trades:
    if 'return' in trade:
        data.loc[trade['exit_date'], 'trade_return'] += trade['return']

data['cumulative_return'] = (1 + data['trade_return']).cumprod()

# Basic trade summary
total_trades = len(trades)
successful_trades = sum(1 for trade in trades if trade.get('return') == TP)
failed_trades = sum(1 for trade in trades if trade.get('return') == -SL)
success_rate = successful_trades / total_trades * 100 if total_trades > 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Successful Trades: {successful_trades}")
print(f"Failed Trades: {failed_trades}")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Cumulative Return: {data['cumulative_return'].iloc[-1]:.2f}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['close'], label='Close Price')

# Flags to ensure each label is added only once
buy_signal_label_added = False
take_profit_label_added = False
stop_loss_label_added = False

for trade in trades:
    if trade['status'] == 'closed':
        if not buy_signal_label_added:
            plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=10, color='g', label='Buy Signal')
            buy_signal_label_added = True
        else:
            plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=10, color='g')
        
        if trade['return'] == TP:
            if not take_profit_label_added:
                plt.plot(trade['exit_date'], trade['exit_price'], 'o', markersize=10, color='b', label='Take Profit')
                take_profit_label_added = True
            else:
                plt.plot(trade['exit_date'], trade['exit_price'], 'o', markersize=10, color='b')
            plt.plot([trade['entry_date'], trade['exit_date']], [trade['entry_price'], trade['exit_price']], 'g--')
        elif trade['return'] == -SL:
            if not stop_loss_label_added:
                plt.plot(trade['exit_date'], trade['exit_price'], 'x', markersize=10, color='r', label='Stop Loss')
                stop_loss_label_added = True
            else:
                plt.plot(trade['exit_date'], trade['exit_price'], 'x', markersize=10, color='r')
            plt.plot([trade['entry_date'], trade['exit_date']], [trade['entry_price'], trade['exit_price']], 'r--')

plt.title('Backtesting Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()