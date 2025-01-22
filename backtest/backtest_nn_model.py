import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from tensorflow.keras.models import load_model  # type: ignore
# from neural_network.data_combined import get_data
import joblib

UPPER_THRESHOLD = 0.8
LOWER_THRESHOLD = 0.2
TP = 0
SL = 0

# Load the pre-trained model
model_path = 'D:\\boobogo\\coding\\trade\\neural_network\\nn_models\\model.keras'
model = load_model(model_path)

# Load the scaler
scaler_path = 'D:\\boobogo\\coding\\trade\\neural_network\\nn_models\\scaler.pkl'
scaler = joblib.load(scaler_path)

data = pd.read_csv('D:\\boobogo\\coding\\trade\\neural_network\\US100_1d_2d_3d.csv', index_col='datetime', parse_dates=True)
data = data.drop(columns=['target'])
data_values = data.values
data_scaled = scaler.transform(data_values)

# Define trading conditions
buy_condition = model.predict(data_scaled) > UPPER_THRESHOLD

# Initialize list to store trades
trades = []

# Simulate trades
for i in range(1, len(data)):  # Start from 1 to ensure previous day's data is available
    # Check for buy condition
    if buy_condition[i]:
        trade = {
            'entry_date': data.index[i],
            'entry_price': data['close'].iloc[i],
            'exit_date': data.index[min(i + 7, len(data) - 1)],  # close after 7 days or at the end of the data
            'exit_price': data['close'].iloc[min(i + 7, len(data) - 1)],
            'magic_number': len(trades) + 1,
            'status': 'close_1dd'
        }
        trades.append(trade)

# Calculate cumulative returns
data['trade_return'] = 0.0
for trade in trades:
    trade_return = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
    data.loc[trade['exit_date'], 'trade_return'] += trade_return

data['cumulative_return'] = (1 + data['trade_return']).cumprod()

# Basic trade summary
total_trades = len(trades)
successful_trades = sum(1 for trade in trades if trade['exit_price'] > trade['entry_price'])
failed_trades = sum(1 for trade in trades if trade['exit_price'] <= trade['entry_price'])
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

for trade in trades:
    if not buy_signal_label_added:
        plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=10, color='g', label='Buy Signal')
        buy_signal_label_added = True
    else:
        plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=10, color='g')
    
    plt.plot(trade['exit_date'], trade['exit_price'], 'o', markersize=10, color='b', label='Exit' if trade['magic_number'] == 1 else "")
    plt.plot([trade['entry_date'], trade['exit_date']], [trade['entry_price'], trade['exit_price']], 'k--')

plt.title('Backtesting Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()