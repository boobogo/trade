import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_historical_var(close_prices, confidence_level):
    """
    Calculate Value at Risk (VaR) using historical data.

    Parameters:
    - close_prices (pd.Series): Series of close prices.
    - confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
    - var (float): Estimated Value at Risk as a percentage.
    - losses (ndarray): Simulated portfolio losses in percentage.
    """
    # Calculate daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Determine the percentile corresponding to the VaR
    var = -np.percentile(daily_returns, 100 * (1 - confidence_level))

    return var, daily_returns

def plot_var_distribution(losses, var, confidence_level):
    """
    Plot the distribution of portfolio losses and highlight the VaR.

    Parameters:
    - losses (ndarray): Simulated portfolio losses in percentage.
    - var (float): Estimated Value at Risk as a percentage.
    - confidence_level (float): Confidence level for VaR.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(losses * 100, bins=200, color='lightblue', edgecolor='black', alpha=0.7)
    plt.axvline(-var * 100, color='red', linestyle='dashed', linewidth=2, label=f"{confidence_level * 100}% VaR: {var * 100:.2f}%")
    plt.title("Historical Data: Portfolio Loss Distribution")
    plt.xlabel("Portfolio Loss (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# Main execution logic
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('D:\\Boobogo\\coding\\trade\\data\\us100\\data_engineered\\US_TECH100_4hour_indicators.csv', parse_dates=True, index_col='datetime')
    
    # Filter data to only include US session (13:00 to 20:00)
    us_session_data = data.between_time('12:00', '20:00')
    close_prices = us_session_data['close_4hour']

    # Parameters for the VaR calculation
    confidence_level = 0.95  # 99% confidence level

    # Calculate VaR using historical data
    var, losses = calculate_historical_var(close_prices, confidence_level)

    # Print results
    print(f"Estimated {confidence_level * 100}% Value at Risk (VaR): {var * 100:.2f}%")

    # Plot the results
    plot_var_distribution(losses, var, confidence_level)