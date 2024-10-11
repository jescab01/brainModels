


import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eigvals

"""
1. Linearize the System
First, define the Jacobian matrix of your system.
"""

# Define the system dynamics
def dfun(state_variables, params):
    c = 135

    He, Hi_slow, Hi_fast = params["He"], params["Hi_slow"], params["Hi_fast"]
    taue, taui_slow, taui_fast = params["taue"], params["taui_slow"], params["taui_fast"]
    c_exc2pyrsup, c_pyrsup2pyrinf = params["c_exc2pyrsup"], params["c_pyrsup2pyrinf"]
    c_pyrsup2inhslow, c_exc2inhfast = params["c_pyrsup2inhslow"], params["c_exc2inhfast"]
    c_inhfast2pyrsup, c_inhslow2pyrinf = params["c_inhfast2pyrsup"], params["c_inhslow2pyrinf"]
    e0, r, v0 = params["e0"], params["r"], params["v0"]

    vPyr_l2l3, vExc_l4, vPyr_l5l6, vInh_fast, vInh_slow = state_variables[:5]
    xPyr_l2l3, xExc_l4, xPyr_l5l6, xInh_fast, xInh_slow = state_variables[5:]

    S_pyrsup = (2 * e0) / (1 + np.exp(r * (v0 - c_exc2pyrsup * vExc_l4 - c_inhfast2pyrsup * vInh_fast)))
    S_pyrinf = (2 * e0) / (1 + np.exp(r * (v0 - c_pyrsup2pyrinf * vPyr_l2l3 - c_inhslow2pyrinf * vInh_slow)))
    S_inhfast = (2 * e0) / (1 + np.exp(r * (v0 - c_exc2inhfast * vExc_l4)))
    S_inhslow = (2 * e0) / (1 + np.exp(r * (v0 - c_pyrsup2inhslow * vPyr_l2l3)))

    vPyr_l2l3 = xPyr_l2l3
    vExc_l4 = xExc_l4
    vPyr_l5l6 = xPyr_l5l6
    vInh_fast = xInh_fast
    vInh_slow = xInh_slow

    xPyr_l2l3 = He / taue * S_pyrsup - (2 * xPyr_l2l3) / taue - vPyr_l2l3 / taue ** 2
    xExc_l4 = He / taue * (params["input_u"]) - (2 * xExc_l4) / taue - vExc_l4 / taue ** 2
    xPyr_l5l6 = He / taue * S_pyrinf - (2 * xPyr_l5l6) / taue - vPyr_l5l6 / taue ** 2
    xInh_fast = Hi_fast / taui_fast * S_inhfast - (2 * xInh_fast) / taui_fast - vInh_fast / taui_fast ** 2
    xInh_slow = Hi_slow / taui_slow * S_inhslow - (2 * xInh_slow) / taui_slow - vInh_slow / taui_slow ** 2

    derivative = [vPyr_l2l3, vExc_l4, vPyr_l5l6, vInh_fast, vInh_slow, xPyr_l2l3, xExc_l4, xPyr_l5l6, xInh_fast, xInh_slow]

    return np.array(derivative)

# Define parameters
params = {
    "He": 3.25, "Hi_slow": 22, "Hi_fast": 10,
    "taue": 10, "taui_slow": 20, "taui_fast": 2,
    "c_exc2pyrsup": 135, "c_pyrsup2pyrinf": 0.8 * 135,
    "c_pyrsup2inhslow": 0.25 * 135, "c_exc2inhfast": 0.25 * 135,
    "c_inhfast2pyrsup": 0.25 * 135, "c_inhslow2pyrinf": 0.25 * 135,
    "input_u": 0, "sigma": 0, "e0": 0.0025, "r": 0.56, "v0": 6
}

# Initial conditions
init_conds = np.random.rand(10)

# Compute Jacobian numerically
def jacobian(f, x0, params, eps=1e-5):
    n = len(x0)
    J = np.zeros((n, n))
    f0 = f(x0, params)
    for i in range(n):
        x1 = np.array(x0)
        x1[i] += eps
        fi = f(x1, params)
        J[:, i] = (fi - f0) / eps
    return J

# Find fixed point (this could be refined or found via a root-finding method)
fixed_point = np.zeros(10)

# Compute Jacobian at the fixed point
J = jacobian(dfun, fixed_point, params)

# Compute eigenvalues of the Jacobian
eigvals_J = eigvals(J)
print("Eigenvalues:", eigvals_J)

# Check stability
stable = np.all(np.real(eigvals_J) < 0)
print("Is the system stable?", stable)





from scipy.optimize import minimize

"""
2. Perform Numerical Parameter Search
If the system is not stable with the current parameters, 
use a numerical optimization technique to find a stable set of parameters.
"""

# Define a cost function that quantifies stability (e.g., sum of positive real parts of eigenvalues)
def stability_cost(params_array):
    params = {
        "He": params_array[0], "Hi_slow": params_array[1], "Hi_fast": params_array[2],
        "taue": params_array[3], "taui_slow": params_array[4], "taui_fast": params_array[5],
        "c_exc2pyrsup": params_array[6], "c_pyrsup2pyrinf": params_array[7],
        "c_pyrsup2inhslow": params_array[8], "c_exc2inhfast": params_array[9],
        "c_inhfast2pyrsup": params_array[10], "c_inhslow2pyrinf": params_array[11],
        "input_u": 0, "sigma": 0, "e0": 0.0025, "r": 0.56, "v0": 6
    }
    J = jacobian(dfun, fixed_point, params)
    eigvals_J = eigvals(J)
    cost = np.sum(np.maximum(0, np.real(eigvals_J)))  # Positive real parts contribute to the cost
    return cost

# Initial parameter guess
params_initial = np.array([3.25, 22, 10,
                           10, 20, 2,
                           135, 108, 33.75, 33.75, 33.75, 33.75])

# Minimize the cost function to find stable parameters
result = minimize(stability_cost, params_initial, method='L-BFGS-B')
print("Optimized parameters:", result.x)
print("Minimum cost:", result.fun)

