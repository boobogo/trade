import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Pricing Formula
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")

# Example Historical Data
historical_data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'S': np.linspace(100, 110, 10),  # Underlying prices
    'K': 105,  # Strike price
    'T': np.linspace(1, 0.9, 10),  # Time to expiry in years
    'r': 0.02,  # Risk-free rate
    'sigma': 0.2,  # Volatility
    'market_price': np.linspace(5, 6, 10),  # Option market prices
})

# Backtest parameters
option_type = "call"
strategy_threshold = 0.5  # Minimum mispricing to trigger a trade

# Simulate backtest
pnl = []
signals = []  # Record signals for visualization
cumulative_pnl = 0

for _, row in historical_data.iterrows():
    theoretical_price = black_scholes(row['S'], row['K'], row['T'], row['r'], row['sigma'], option_type)
    mispricing = row['market_price'] - theoretical_price

    # Simple trading logic
    if mispricing > strategy_threshold:
        signals.append("Sell")
        cumulative_pnl += mispricing  # Assume instant execution
    elif mispricing < -strategy_threshold:
        signals.append("Buy")
        cumulative_pnl -= mispricing  # Assume instant execution
    else:
        signals.append("Hold")

    pnl.append(cumulative_pnl)

# Add signals and PnL to the dataframe
historical_data['signals'] = signals
historical_data['pnl'] = pnl

# Plotting
plt.figure(figsize=(12, 8))

# Price Comparison
plt.subplot(2, 1, 1)
plt.plot(historical_data['date'], historical_data['market_price'], label="Market Price", marker="o")
plt.plot(historical_data['date'], 
         [black_scholes(row['S'], row['K'], row['T'], row['r'], row['sigma'], option_type) for _, row in historical_data.iterrows()], 
         label="Theoretical Price", linestyle="--")
plt.title("Market Price vs Theoretical Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()

# PnL Over Time
plt.subplot(2, 1, 2)
plt.plot(historical_data['date'], historical_data['pnl'], label="Cumulative PnL", color="green", marker="o")
plt.title("Cumulative PnL Over Time")
plt.xlabel("Date")
plt.ylabel("PnL")
plt.axhline(0, color='red', linestyle='--', label="Break-even")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
