import numpy as np
from plot import plot

x = np.arange(1, 6).reshape(-1, 1)
y = np.array([[3.74013816], [3.61473236], [
             4.57655287], [4.66793434], [5.95585554]])

# Example 1:
theta1 = np.array([[4.5], [-0.2]])
plot(x, y, theta1)

# Example 2:
theta2 = np.array([[-1.5], [2]])
plot(x, y, theta2)

# Example 3:
theta3 = np.array([[3], [0.3]])
plot(x, y, theta3)

# Errors

assert plot('42', y, theta3) is None, "Invalid call should be None"
assert plot(x, '42', theta3) is None, "Invalid call should be None"
assert plot(x, y, '42') is None, "Invalid call should be None"
assert plot(np.array([]), y, theta3) is None, "Invalid call should be None"
assert plot(x, np.array([]), theta3) is None, "Invalid call should be None"
assert plot(x, y, np.array([])) is None, "Invalid call should be None"
assert plot(np.array([[]]), y, theta3) is None, "Invalid call should be None"
assert plot(x, np.array([[]]), theta3) is None, "Invalid call should be None"
assert plot(x, y, np.array([[]])) is None, "Invalid call should be None"
