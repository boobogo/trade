from stable_baselines3.common.env_checker import check_env
from trade_env import TradingEnv
import pandas as pd
# from gymnasium.utils.env_checker import check_env

# Read the CSV file
data = pd.read_csv('data/data_processed.csv')

# Create an instance of the trading environment
env = TradingEnv(data)

check_env(env)  # Check the environment
