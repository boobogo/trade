# Python code for the plot

import numpy as np
import matplotlib.pyplot as plt

mu = 1
n = 50
dt = 0.1
x0 = 100
np.random.seed(1)

sigma = np.arange(0.8, 2, 0.2)

x = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(len(sigma), n)).T
)
x = np.vstack([np.ones(len(sigma)), x])
x = x0 * x.cumprod(axis=0)

plt.plot(x)
plt.legend(np.round(sigma, 2))
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.title(
    "Realizations of Geometric Brownian Motion with different variances\n $\mu=1$"
)
plt.show()


"""
In real stock prices, volatility changes over time (possibly stochastically), but in GBM, volatility is assumed constant.
In real life, stock prices often show jumps caused by unpredictable events or news, but in GBM, the path is continuous (no discontinuity)
"""