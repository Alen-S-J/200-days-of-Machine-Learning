import sympy as sp

# Define the symbol and the function
x = sp.symbols('x')
f = x**2 + 3*x + 5  # Example function: x^2 + 3x + 5

# Differentiate the function
derivative_f = sp.diff(f, x)
print("Derivative of f:", derivative_f)

# Define the function to integrate
g = x**3 + 2*x**2 + 1  # Example function: x^3 + 2x^2 + 1

# Integrate the function
integral_g = sp.integrate(g, x)
print("$$nIntegral of g:", integral_g)

# Define the differential equation: dy/dx = x
y = sp.Function('y')
diff_eq = sp.Eq(y(x).diff(x), x)

# Solve the differential equation
solution_diff_eq = sp.dsolve(diff_eq)
print("$$nSolution of the differential equation:", solution_diff_eq)
