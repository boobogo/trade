import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_investment(initial_investment, years, mean_return, volatility, num_simulations):
    """
    Perform Monte Carlo simulation to estimate future investment value.

    Parameters:
    - initial_investment (float): Initial investment amount.
    - years (int): Investment horizon in years.
    - mean_return (float): Average annual return (in decimal, e.g., 0.07 for 7%).
    - volatility (float): Annual return volatility (standard deviation, e.g., 0.15 for 15%).
    - num_simulations (int): Number of simulation runs.

    Returns:
    - final_values (ndarray): Simulated final values of the investment.
    """
    # Initialize an array to store final investment values
    final_values = np.zeros(num_simulations)

    # Simulate each investment trajectory
    for i in range(num_simulations):
        # Simulate annual returns for the given number of years
        annual_returns = np.random.normal(mean_return, volatility, years)
        
        # Calculate the compounded growth over the years
        final_values[i] = initial_investment * np.prod(1 + annual_returns)

    return final_values

def plot_simulation_results(final_values):
    """
    Plot the results of the Monte Carlo simulation.

    Parameters:
    - final_values (ndarray): Simulated final values of the investment.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    mean_value = np.mean(final_values)
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f"Mean Value: ${mean_value:.2f}")
    plt.title("Monte Carlo Simulation: Distribution of Investment Outcomes")
    plt.xlabel("Final Investment Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# Main execution logic
if __name__ == "__main__":
    # Load your historical data
    data = pd.read_csv('D:\\Boobogo\\coding\\trade\\data\\us100\\data_engineered\\US_TECH100_1d_indicators.csv')  # Replace with your data file path
    close_prices = data['close_1d']

    # Calculate historical daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Convert daily returns to annual returns
    mean_return = daily_returns.mean() * 252  # Assuming 252 trading days in a year
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Parameters for the simulation
    initial_investment = 10000  # Starting with $10,000
    years = 10                  # Investment horizon of 10 years
    num_simulations = 100000     # Run 100,000 simulations

    # Run Monte Carlo simulation
    final_values = monte_carlo_investment(initial_investment, years, mean_return, volatility, num_simulations)

    # Print results
    print(f"Mean final investment value: ${np.mean(final_values):.2f}")
    print(f"Median final investment value: ${np.median(final_values):.2f}")
    print(f"10th percentile: ${np.percentile(final_values, 10):.2f}")
    print(f"90th percentile: ${np.percentile(final_values, 90):.2f}")

    # Plot simulation results
    plot_simulation_results(final_values)