'''
Maximize: Z = 50x1 + 40x2 + 30x3 (Profit)

Subject to:

2x1 + x2 + 3x3 ≤ 240 (Machining hours constraint)
3x1 + 2x2 + x3 ≤ 300 (Assembly hours constraint)
x1 + 2x2 + x3 ≤ 180 (Finishing hours constraint)
x1 ≤ 70 (Market demand for P1)
x2 ≤ 100 (Market demand for P2)
x3 ≤ 80 (Market demand for P3)
x1, x2, x3 ≥ 0 (Non-negativity constraints)

'''

from scipy.optimize import linprog

# Objective function coefficients (to be MINIMIZED, so we negate them)
c = [-50, -40, -30]

# Inequality constraint coefficients (Ax <= b)
A = [[2, 1, 3],
     [3, 2, 1],
     [1, 2, 1],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

# Inequality constraint bounds
b = [240, 300, 180, 70, 100, 80]

# Variable bounds (x1, x2, x3 >= 0)
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)

# Solve the linear program
result = linprog(c, A_ub=A, b_ub=b, bounds=[x1_bounds, x2_bounds, x3_bounds])

# Print the results
print("Optimal production plan:")
print("Product 1 (x1):", int(result.x[0]))
print("Product 2 (x2):", int(result.x[1]))
print("Product 3 (x3):", int(result.x[2]))
print("Maximum profit:", -result.fun)
print("Is optimal:", result.success)
print("Message:", result.message)