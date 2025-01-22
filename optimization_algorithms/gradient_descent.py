import numpy as np
import matplotlib.pyplot as plt

# import sympy as sp

# # Define the variables
# x, y = sp.symbols('x y')

# # Define the function
# f = x**2 + y**2 - 4*x - 6*y + 13

# # Calculate partial derivatives
# partial_x = sp.diff(f, x)  # Partial derivative w.r.t. x
# partial_y = sp.diff(f, y)  # Partial derivative w.r.t. y

# print(type(partial_x))
# # Display the results
# print("Partial derivative with respect to x:", partial_x)
# print("Partial derivative with respect to y:", partial_y)


def f(x, y):
    # Define the function to be minimized
    return x**2 + y**2 - 4*x - 6*y + 13

def gradient(x, y):
    # Compute the gradient of the function
    return np.array([2*x - 4, 2*y - 6]) # Gradient vector [df/dx, df/dy]

def gradient_descent(learning_rate=0.1, iterations=100):
    x = 0  # Initial guess for x
    y = 0  # Initial guess for y
    history_x = []
    history_y = []
    history_f = []

    for i in range(iterations):
        grad = gradient(x, y)  # Compute the gradient at the current point
        x = x - learning_rate * grad[0]  # Update x using the gradient
        y = y - learning_rate * grad[1]  # Update y using the gradient
        history_x.append(x)  
        history_y.append(y)  
        history_f.append(f(x, y))  # Store the function value at the updated point

    return x, y, history_x, history_y, history_f  # Return the final values and history

# Run gradient descent
optimal_x, optimal_y, history_x, history_y, history_f = gradient_descent()

print(f"Optimal x: {optimal_x}")
print(f"Optimal y: {optimal_y}")
print(f"Minimum function value: {f(optimal_x, optimal_y)}")


# Plotting the convergence
plt.figure(figsize=(12, 4))

# Plot x and y values over iterations
plt.subplot(1, 2, 1)
plt.plot(history_x, label="x")
plt.plot(history_y, label="y")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Convergence of x and y")
plt.legend()

# Plot the function value over iterations
plt.subplot(1, 2, 2)
plt.plot(history_f)
plt.xlabel("Iteration")
plt.ylabel("f(x, y)")
plt.title("Convergence of f(x, y)")

plt.tight_layout()
plt.show()