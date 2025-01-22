from trade_env import TradingEnv
import pandas as pd
# from gymnasium.utils.env_checker import check_env

# Read the CSV file
data = pd.read_csv('data/data_processed.csv')

env = TradingEnv(data)
episodes = 1

for episode in range(episodes):
	terminated = False
	obs = env.reset()
	while not terminated:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, terminated, trancuated, info = env.step(random_action)
		print('reward',reward)