import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Load oil price data (example using a CSV file)
# Replace 'oil_prices.csv' with the path to your dataset
# The dataset should have columns: 'Date' and 'Price'
data = pd.read_csv('https://raw.githubusercontent.com/datasets/oil-prices/master/data/brent-daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Extract price data
oil_prices = data['Price'].values
time = np.arange(len(oil_prices))

# Apply Fourier Transform
fft_coeffs = fft(oil_prices)
frequencies = np.fft.fftfreq(len(oil_prices))

# Visualize the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(np.abs(frequencies[:len(oil_prices)//2]), np.abs(fft_coeffs[:len(oil_prices)//2]), label="Frequency Spectrum")
plt.title("Frequency Spectrum of Oil Prices")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Filter high-frequency noise
threshold = 0.005  # Adjust threshold to retain only significant frequencies
filtered_coeffs = fft_coeffs.copy()
filtered_coeffs[np.abs(frequencies) > threshold] = 0

# Inverse Fourier Transform for smoothing
smoothed_prices = np.real(ifft(filtered_coeffs))

# Calculate 20-day moving average
moving_average_20d = pd.Series(oil_prices).rolling(window=100).mean().values

# Plot original vs. smoothed prices and 20-day moving average
plt.figure(figsize=(12, 6))
plt.plot(time, oil_prices, label="Original Oil Prices", alpha=0.7)
plt.plot(time, smoothed_prices, label="Smoothed Oil Prices", color='red', linewidth=2)
plt.plot(time, moving_average_20d, label="20-Day Moving Average", color='green', linewidth=2)
plt.title("Oil Prices: Original vs. Smoothed vs. 20-Day Moving Average")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
