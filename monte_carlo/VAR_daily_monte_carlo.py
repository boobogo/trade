import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_var(portfolio_value, mean_return, volatility, num_simulations, confidence_level):
    """
    Perform Monte Carlo simulation to estimate Value at Risk (VaR).

    Parameters:
    - portfolio_value (float): Initial value of the portfolio.
    - mean_return (float): Expected daily return.
    - volatility (float): Daily return volatility.
    - num_simulations (int): Number of simulation runs.
    - confidence_level (float): Confidence level for VaR.

    Returns:
    - var (float): Estimated Value at Risk.
    - losses (ndarray): Simulated portfolio losses.
    """
    # Simulate daily returns
    simulated_returns = np.random.normal(mean_return, volatility, num_simulations)

    # Calculate portfolio losses for each simulation
    losses = portfolio_value * simulated_returns

    # Determine the percentile corresponding to the VaR
    var = -np.percentile(losses, 100 * (1 - confidence_level))

    return var, losses

def plot_var_distribution(losses, var, confidence_level):
    """
    Plot the distribution of portfolio losses and highlight the VaR.

    Parameters:
    - losses (ndarray): Simulated portfolio losses.
    - var (float): Estimated Value at Risk.
    - confidence_level (float): Confidence level for VaR.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    plt.axvline(-var, color='red', linestyle='dashed', linewidth=2, label=f"{confidence_level * 100}% VaR: ${var:,.2f}")
    plt.title("Monte Carlo Simulation: Portfolio Loss Distribution")
    plt.xlabel("Portfolio Loss ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# Main execution logic
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('D:\\Boobogo\\coding\\trade\\data\\us100\\data_engineered\\US_TECH100_1d_indicators.csv')  # Replace with your data file path
    close_prices = data['close_1d']

    # Calculate historical daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Parameters for the Monte Carlo simulation
    portfolio_value = 1000  # Initial portfolio value ($)
    mean_return = daily_returns.mean()  # Mean of historical daily returns
    volatility = daily_returns.std()  # Standard deviation of historical daily returns
    num_simulations = 1000000  # Number of simulations
    confidence_level = 0.95  # 95% confidence level

    # Calculate VaR using Monte Carlo simulation
    var, losses = monte_carlo_var(portfolio_value, mean_return, volatility, num_simulations, confidence_level)

    # Print results
    print(f"Estimated {confidence_level * 100}% Value at Risk (VaR): ${var:,.2f}")

    print(len(losses))
    # Plot the results
    plot_var_distribution(losses, var, confidence_level)
    