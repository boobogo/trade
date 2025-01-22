import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + y**2 - 4*x - 6*y + 13

def gradient(x, y):
    return np.array([2*x - 4, 2*y - 6])

def hessian():
    return np.array([[2, 0], [0, 2]])

def newton_method(initial_guess=np.array([0, 0]), tolerance=1e-6, max_iterations=100):
    x = initial_guess
    path = [x.copy()] # Store the path of x values
    for _ in range(max_iterations):
        grad = gradient(x[0], x[1])
        hess = hessian()
        hess_inv = np.linalg.inv(hess)
        x_new = x - hess_inv @ grad
        if np.linalg.norm(x_new - x) < tolerance:
            path.append(x_new.copy())
            return x_new, np.array(path)
        x = x_new
        path.append(x.copy())
    return x, np.array(path)

# Run Newton's method
optimal_point, path = newton_method(initial_guess=np.array([-1, -1])) # start from (-1,-1)

print(f"Optimal point (x, y): {optimal_point}")
print(f"Minimum function value: {f(optimal_point[0], optimal_point[1])}")

# Create the plot
x_range = np.linspace(-3, 5, 100)
y_range = np.linspace(-3, 7, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis') # surface plot
ax.plot(path[:,0], path[:,1], f(path[:,0],path[:,1]), marker='o', c='r', label='Newton\'s Path') # path plot
ax.scatter(2,3,0, marker='x', c='black', s=100, label='Optimal Point') # optimal point

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Newton\'s Method Visualization')
ax.legend()
plt.show()


# Example with a different initial guess
optimal_point2, path2 = newton_method(initial_guess=np.array([5,5]))

print(f"Optimal point (x, y): {optimal_point2}")
print(f"Minimum function value: {f(optimal_point2[0], optimal_point2[1])}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis') # surface plot
ax.plot(path2[:,0], path2[:,1], f(path2[:,0],path2[:,1]), marker='o', c='r', label='Newton\'s Path') # path plot
ax.scatter(2,3,0, marker='x', c='black', s=100, label='Optimal Point') # optimal point

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Newton\'s Method Visualization - different start')
ax.legend()
plt.show()