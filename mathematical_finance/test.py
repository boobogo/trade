import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate non-normal distribution data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(loc=0, scale=0.5, size=2000),
    np.random.normal(loc=0, scale=1.5, size=2000)
])

# Take logarithm of each data point
log_data = np.log(data - data.min() + 1)  # Shift data to avoid log(0) or negative values

# Plot the original and log-transformed data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data, bins=50, kde=True, color='blue')
plt.title('Original Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(log_data, bins=50, kde=True, color='green')
plt.title('Log-Transformed Data Distribution')
plt.xlabel('Log(Value)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()