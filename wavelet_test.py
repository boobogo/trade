import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import yfinance as yf

# 1. Load Oil Price Data
oil_data = yf.download("CL=F", start="2015-01-01", end="2025-01-01")
oil_prices = oil_data['Close'].dropna()

# 2. Preprocess Data
time = np.arange(len(oil_prices))
signal = oil_prices.values
dt = 1  # Assuming daily data

# 3. Apply Continuous Wavelet Transform (CWT)
scales = np.arange(1, 128)  # Scales correspond to frequency bands
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', dt)

# Convert scales to frequencies for labeling
frequencies = 1 / (scales * dt)  # Approximate frequency (cycles per day)

# 4. Visualize Original Data
plt.figure(figsize=(14, 5))
plt.plot(time, signal, label='Oil Prices')
plt.title('Original Brent Oil Prices')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 5. Visualize Wavelet Transform with Frequency Labels
plt.figure(figsize=(14, 7))
plt.imshow(np.abs(coefficients), extent=[time[0], time[-1], frequencies[-1], frequencies[0]],
           cmap='jet', aspect='auto')
plt.colorbar(label='Magnitude')
plt.title('Wavelet Transform of Oil Prices (Frequency Analysis)')
plt.xlabel('Time (Days)')
plt.ylabel('Frequency (Cycles per Day)')
plt.yscale('log')  # Log scale to emphasize lower frequencies
plt.grid(True)
plt.show()

# 6. Plot Wavelet Coefficients at Different Scales
num_plots = min(6, len(scales))  # Limit the number of plots for clarity
plt.figure(figsize=(14, 10))
for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(coefficients[i], label=f'Scale {scales[i]}')
    plt.title(f'Wavelet Coefficients at Scale {scales[i]}')
    plt.xlabel('Time (Days)')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()