from trade_env import TradingEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import os
from datetime import datetime
import pandas as pd

models_dir = f"drl_models/{datetime.today().date().strftime("%Y-%m-%d")}/" 
log_dir = f"drl_logs/{datetime.today().date().strftime("%Y-%m-%d")}/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv('data/data_processed.csv') # index_col='Date', parse_dates=True

# Calculate the split point
split_point = int(len(df) * 0.7)

# Split the data for training
df = df.iloc[:split_point]


# Create an instance of the trading environment
env = TradingEnv(df)

env.reset()

# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/PPO/{TIMESTEPS*iters}")
	# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	# model.save(f"{models_dir}/A2C/{TIMESTEPS*iters}")