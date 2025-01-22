import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Example: Generate synthetic financial data
np.random.seed(42)
n = 1000
returns = np.random.normal(0, 1, n)  # Normally distributed returns

# Convert to pandas Series
returns = pd.Series(returns)

# Fit a GARCH(1,1) model
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp="off")

# Print the summary
print(garch_fit.summary())

# Plot conditional volatility
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(garch_fit.conditional_volatility, label='Conditional Volatility')
ax.set_title("GARCH(1,1) Conditional Volatility")
ax.legend()
plt.show()
