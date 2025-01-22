import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()

        # df is a DataFrame with the NASDAQ 100 index df
        self.df = df

        # Action space is a Box with 2x2 elements
        # element (0) is between 0 and 1. 0: do nothing, 1: buy
        # element (1) is between 1 and 10. lot size
        # element (2) is between 1 and 500. stop loss (in percentage)
        # element (3) is between 1 and 500. take profit (in percentage)
        low = np.array([0, 1, 1, 1])
        high = np.array([1, 10, 500, 500])
        self.action_space = Box(low, high, shape=(4,), dtype=np.float32)

        # Observations are the current price and indicators
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,)) #shape=(df.shape[1],)

        self.current_step = None
        self.balance = None
        self.equity = None
        self.used_margin = None
        self.free_margin = None
        self.margin_requirement = None
        self.margin_level = None
        self.initial_balance = 1000
        self.leverage = 200
        self.max_balance = self.initial_balance

        self.position_exist = None
        self.bought_price = None
        self.sold_price = None
        self.bought_index = None
        self.sold_index = None
        self.stop_loss = None
        self.take_profit = None
        self.multiplier = None

        self.position_opened = None
        self.position_closed = None
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.balance
        self.position_exist = False
        self.position_opened = False
        self.position_closed = False
        # self.bought_price = 0
        # self.sold_price = 0
        # self.bought_index = 0
        # self.sold_index = 0
        # self.stop_loss = 0
        # self.take_profit = 0
        # self.multiplier = 0 #0.01 lot size = 1 multiplier
        self.margin_level = np.inf
        self.max_balance = self.initial_balance
    
        return self._get_observation(), {}
        
    def step(self, action):
        reward = 0
        self.position_opened = False
        self.position_closed = False

        #get the current price
        current_price = self.df.loc[self.current_step, 'close']

        #margin requirement for the current price
        current_margin_req = current_price / self.leverage
        self.margin_requirement = current_price / self.leverage

        #if open position, calculate the unrealized pnl and used margin
        if self.position_exist == True:
            unrealized_pnl = (current_price - self.bought_price) * self.multiplier
            
            self.equity = self.balance + unrealized_pnl

            self.used_margin = self.margin_requirement * self.multiplier

            self.margin_level = (self.equity / self.used_margin) * 100

            if self.margin_level <= 50:
                reward -= 20
            elif self.margin_level <= 100:
                reward -= 10

            #check if stop loss or take profit is hit
            if current_price <= self.stop_loss: # if stop loss is hit, close the position
                self.balance += unrealized_pnl
                reward -= 5
                self.sold_price = current_price
                self.sold_index = self.current_step
                self.position_closed = True
                self.position_exist = False
            elif current_price >= self.take_profit: # if take profit is hit, close the position
                self.balance += unrealized_pnl
                reward += 10
                self.sold_price = current_price
                self.sold_index = self.current_step
                self.position_closed = True
                self.position_exist = False

        else:   #if NO open position
            self.equity = self.balance
            if round(action[0]) == 1: #open long position
                self.multiplier = action[1]
                self.bought_price = current_price
                self.bought_index = self.current_step
                self.stop_loss = current_price - (current_price * action[2] / 100)
                self.take_profit = current_price + (current_price * action[3] / 100)
                self.position_exist = True
                self.position_opened = True
        
        # After updating self.balance
        self.max_balance = max(self.max_balance, self.balance)
        drawdown = 1 - (self.balance / self.max_balance)

        # Penalize significant drawdowns
        if drawdown > 0.2:  # Change this threshold to suit your needs
            reward -= 50  # Change this penalty to suit your needs

        #increment the current step for the next observation
        self.current_step += 1

        #check if the episode is terminated
        terminated = self.current_step >= len(self.df)-1
        info = {'bought_price': self.bought_price,
                'bought_index': self.bought_index,
                'sold_price': self.sold_price,
                'sold_index': self.sold_index,
                'position_opened': self.position_opened,
                'position_closed': self.position_closed,
                'reward': reward,
                'balance': self.balance,
                'equity': self.equity,
                'multiplier': self.multiplier,}

        return self._get_observation(), reward, terminated, False, info
    
    def render(self, mode='human'):
        pass

    def _get_observation(self):
        # Return the current price and indicators as the observation
        return self.df.iloc[self.current_step][['open','high','low','close','volume','rsi','macd','signal','bb_bbm','bb_bbh','bb_bbl','stoch','stoch_signal']].values.astype(np.float32)