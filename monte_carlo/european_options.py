import numpy as np
from scipy.stats import norm

# Parameters
S0 = 100        # Initial stock price
K = 100         # Strike price
T = 1           # Time to maturity (1 year)
r = 0.04        # Risk-free rate
sigma = 0.2     # Volatility
n_simulations = 100000  # Number of simulations
n_steps = 252   # Daily time steps

# Time increment
dt = T / n_steps

# Simulate price paths
np.random.seed(42)  # For reproducibility
price_paths = np.zeros((n_simulations, n_steps + 1))
price_paths[:, 0] = S0

for t in range(1, n_steps + 1): 
    z = np.random.standard_normal(n_simulations) # Random numbers from standard normal distribution (mean=0, std=1)
    price_paths[:, t] = price_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z) # Geometric Brownian Motion

# Calculate option payoff
payoffs = np.maximum(price_paths[:, -1] - K, 0)

# Discount payoffs to present value
option_price = np.mean(payoffs) * np.exp(-r * T)



# Black-Scholes formula implementation
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calculate Black-Scholes price
bs_price = black_scholes_call(S0, K, T, r, sigma)

# Output comparison
print(f"Monte Carlo Estimated Call Option Price: {option_price:.2f}")
print(f"Black-Scholes Calculated Call Option Price: {bs_price:.2f}")

