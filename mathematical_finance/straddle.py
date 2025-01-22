import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Black-Scholes model for option pricing
def black_scholes(S, X, T, r, sigma, option_type='call'):
    # S: Spot price
    # X: Strike price
    # T: Time to expiration (in years)
    # r: Risk-free rate
    # sigma: Volatility (annualized)
    # option_type: 'call' or 'put'
    
    # d1 and d2 for the Black-Scholes model
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        option_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price

# Parameters
S = 521  # Spot price of QQQ
X = 522.5  # Strike price
T = 1/12  # 1 month to expiration (in years)
r = 0.05  # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

# Calculate the call and put prices using the Black-Scholes model
call_price = black_scholes(S, X, T, r, sigma, 'call')
put_price = black_scholes(S, X, T, r, sigma, 'put')

# Print the calculated option prices
print(f"Call Option Price (BSM): {call_price:.2f}")
print(f"Put Option Price (BSM): {put_price:.2f}")

# Market Prices (for example)
market_call_price = 10.46  # Market call price
market_put_price = 10.92  # Market put price

# Identify if the options are overpriced or underpriced
if market_call_price > call_price:
    print("Market call option is overpriced, consider selling.")
else:
    print("Market call option is underpriced, consider buying.")

if market_put_price > put_price:
    print("Market put option is overpriced, consider selling.")
else:
    print("Market put option is underpriced, consider buying.")
    
# Straddle strategy (buying both call and put)
straddle_cost = market_call_price + market_put_price
print(f"Total cost of the straddle strategy: {straddle_cost:.2f}")

# Example: Simulate price changes and calculate potential profit for straddle
simulated_prices = np.random.normal(S, sigma, 1000)  # Simulate 1000 potential price paths
call_payoffs = np.maximum(simulated_prices - X, 0)  # Call option payoff
put_payoffs = np.maximum(X - simulated_prices, 0)  # Put option payoff
total_payoffs = call_payoffs + put_payoffs  # Total payoff from the straddle

# Calculate the average profit/loss
average_payoff = np.mean(total_payoffs) - straddle_cost  # Subtract the initial cost
print(f"Average profit/loss from the straddle strategy: {average_payoff:.2f}")

# Plotting the payoff distribution
plt.hist(total_payoffs - straddle_cost, bins=50, edgecolor='black')
plt.title('Straddle Strategy Payoff Distribution')
plt.xlabel('Profit/Loss')
plt.ylabel('Frequency')
plt.show()
