import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from trade_env import TradingEnv
from stable_baselines3 import PPO, A2C
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('data/data_processed.csv') # index_col='Date', parse_dates=True

# Calculate the split point
split_point = int(len(df) * 0.7)

# Split the data for testing set
df = df.iloc[split_point:]

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

# Create an instance of the trading environment
env = TradingEnv(df)

# Load the trained model
model = PPO.load("models/2024-05-08/PPO/190000")  # replace with the path to your model

# Initialize the environment
obs, _ = env.reset()

# Initialize the total reward
total_reward = 0

# Initialize the best and worst rewards
best_reward = float('-inf')
worst_reward = float('inf')

# Initialize the dates of the best and worst rewards
best_date = worst_date = None

# Initialize lists to store the opening and closing actions
open_actions = []
close_actions = []

for i in range(len(df)):
    action, _states =  model.predict(obs, deterministic=False) #env.action_space.sample()
    obs, reward, terminated, trancuated, info = env.step(action)
    total_reward += reward

    # Update the best and worst rewards and their dates
    if reward > best_reward:
        best_reward = reward
        best_date = df.index[i]
    if reward < worst_reward:
        worst_reward = reward
        worst_date = df.index[i]

    # Store the opening and closing actions from info dictionary
    if info['position_opened']:
        open_actions.append((info['bought_index'], info['bought_price']))
    if info['position_closed']:
        close_actions.append((info['sold_index'], info['sold_price']))

    # print(f"Buy Date: {df.iloc[info['open_index']]['datetime']}, Price: {info['open_price']}, "
    # f"Close Date: {df.iloc[i]['datetime']}, Price: {df.loc[i, 'close']}, Reward: {reward}, multipler: {info['multiplier']}")
    print(f"Reward: {reward}, multipler: {info['multiplier']}")

    if terminated:
        break

print(f"Total reward: {total_reward}")
print(f"Best reward: {best_reward} on date: {df.iloc[best_date]['datetime']}")
print(f"Worst reward: {worst_reward} on date: {df.iloc[worst_date]['datetime']}")

# Plot the actions
plt.figure(figsize=(10, 6))
plt.plot(df['close'], 'k-', label='Close price')

# Plot the opening and closing buy actions with dashed lines connecting them
for (open_date, open_price), (close_date, close_price) in zip(open_actions, close_actions):
    plt.plot([open_date, close_date], [open_price, close_price], 'g--', label='Buy')

plt.title('Trading actions')
plt.show()
    
