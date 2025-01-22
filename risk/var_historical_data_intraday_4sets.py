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
    - var (float): Estimated Value at Risk.
    - losses (ndarray): Simulated portfolio losses in percentage.
    """
    # Calculate daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Determine the percentile corresponding to the VaR
    var = -np.percentile(daily_returns, 100 * (1 - confidence_level))

    return var, daily_returns


def plot_var_distribution(losses_list, var_list, std_list, mean_list, confidence_level, date_ranges):
    """
    Plot the distribution of portfolio losses and highlight the VaR for multiple sets.

    Parameters:
    - losses_list (list of ndarray): List of simulated portfolio losses for each set.
    - var_list (list of float): List of estimated Value at Risk for each set.
    - std_list (list of float): List of standard deviations for each set.
    - mean_list (list of float): List of mean values for each set.
    - confidence_level (float): Confidence level for VaR.
    - date_ranges (list of tuple): List of date ranges for each set.
    """
    plt.figure(figsize=(12, 8))
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightgray']
    labels = [f'Set {i+1} ({date_ranges[i][0]} to {date_ranges[i][1]})' for i in range(len(date_ranges))]

    for i, (losses, var, std, mean) in enumerate(zip(losses_list, var_list, std_list, mean_list)):
        plt.hist(losses * 100, bins=50, color=colors[i], edgecolor='black', alpha=0.5, label=f"{labels[i]}: {confidence_level * 100}% VaR: {var * 100:.2f}%, Std Dev: {std * 100:.2f}%, Mean: {mean * 100:.2f}%")
        plt.axvline(-var * 100, color=colors[i], linestyle='dashed', linewidth=2)

    plt.title("Historical Data: Portfolio Loss Distribution Comparison")
    plt.xlabel("Portfolio Loss (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()


# Main execution logic
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_4hour_indicators.csv')  # Replace with your data file path
    close_prices = data['close_4hour']
    dates = data['datetime']

    # Divide data into 4 sets
    n = len(close_prices)
    set1 = close_prices[:n//4]
    set2 = close_prices[n//4:n//2]
    set3 = close_prices[n//2:3*n//4]
    set4 = close_prices[3*n//4:]

    # Divide data into 6 sets
    set1 = close_prices[:n//6]
    set2 = close_prices[n//6:n//3]
    set3 = close_prices[n//3:n//2]
    set4 = close_prices[n//2:2*n//3]
    set5 = close_prices[2*n//3:5*n//6]
    set6 = close_prices[5*n//6:]

    # Get date ranges for each set
    date_ranges = [
        (dates.iloc[0], dates.iloc[n//6 - 1]),
        (dates.iloc[n//6], dates.iloc[n//3 - 1]),
        (dates.iloc[n//3], dates.iloc[n//2 - 1]),
        (dates.iloc[n//2], dates.iloc[2*n//3 - 1]),
        (dates.iloc[2*n//3], dates.iloc[5*n//6 - 1]),
        (dates.iloc[5*n//6], dates.iloc[-1])
    ]

    # Parameters for the VaR calculation
    confidence_level = 0.95  # 95% confidence level

    # Calculate VaR, standard deviation, and mean for each set
    var1, losses1 = calculate_historical_var(set1, confidence_level)
    var2, losses2 = calculate_historical_var(set2, confidence_level)
    var3, losses3 = calculate_historical_var(set3, confidence_level)
    var4, losses4 = calculate_historical_var(set4, confidence_level)
    var5, losses5 = calculate_historical_var(set5, confidence_level)
    var6, losses6 = calculate_historical_var(set6, confidence_level)

    std1 = np.std(losses1)
    std2 = np.std(losses2)
    std3 = np.std(losses3)
    std4 = np.std(losses4)
    std5 = np.std(losses5)
    std6 = np.std(losses6)

    mean1 = np.mean(losses1)
    mean2 = np.mean(losses2)
    mean3 = np.mean(losses3)
    mean4 = np.mean(losses4)
    mean5 = np.mean(losses5)
    mean6 = np.mean(losses6)

    # Calculate standard deviation and mean for the whole duration
    _, all_losses = calculate_historical_var(close_prices, confidence_level)
    std_all = np.std(all_losses)
    mean_all = np.mean(all_losses)

    # Print results with date ranges, standard deviations, and means
    print(f"Set 1 ({date_ranges[0][0]} to {date_ranges[0][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var1 * 100:.2f}%, Std Dev: {std1 * 100:.2f}%, Mean: {mean1 * 100:.2f}%")
    print(f"Set 2 ({date_ranges[1][0]} to {date_ranges[1][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var2 * 100:.2f}%, Std Dev: {std2 * 100:.2f}%, Mean: {mean2 * 100:.2f}%")
    print(f"Set 3 ({date_ranges[2][0]} to {date_ranges[2][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var3 * 100:.2f}%, Std Dev: {std3 * 100:.2f}%, Mean: {mean3 * 100:.2f}%")
    print(f"Set 4 ({date_ranges[3][0]} to {date_ranges[3][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var4 * 100:.2f}%, Std Dev: {std4 * 100:.2f}%, Mean: {mean4 * 100:.2f}%")
    print(f"Set 5 ({date_ranges[4][0]} to {date_ranges[4][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var5 * 100:.2f}%, Std Dev: {std5 * 100:.2f}%, Mean: {mean5 * 100:.2f}%")
    print(f"Set 6 ({date_ranges[5][0]} to {date_ranges[5][1]}): Estimated {confidence_level * 100}% Value at Risk (VaR): {var6 * 100:.2f}%, Std Dev: {std6 * 100:.2f}%, Mean: {mean6 * 100:.2f}%")
    print(f"Whole Duration: Std Dev: {std_all * 100:.2f}%, Mean: {mean_all * 100:.2f}%")

    # Plot the results
    plot_var_distribution([losses1, losses2, losses3, losses4, losses5, losses6], [var1, var2, var3, var4, var5, var6], [std1, std2, std3, std4, std5, std6], [mean1, mean2, mean3, mean4, mean5, mean6], confidence_level, date_ranges)