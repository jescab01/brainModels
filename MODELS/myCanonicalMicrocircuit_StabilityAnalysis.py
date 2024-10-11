

from scipy.optimize import fsolve
import numpy as np


def dfun(state_variables):
    r"""
    The dynamic equations were taken from:
    TODO: add equations and finish the model ...
    """

    c = 135

    params = dict(He=np.array([3.25]), Hi_slow=np.array([22]), Hi_fast=np.array([10]),
                  taue=np.array([10]), taui_slow=np.array([20]), taui_fast=np.array([2]),
                  c_exc2pyrsup=np.array([c]), c_pyrsup2pyrinf=np.array([0.8 * c]),
                  c_pyrsup2inhslow=np.array([0.25 * c]), c_exc2inhfast=np.array([0.25 * c]),
                  c_inhfast2pyrsup=np.array([0.25 * c]), c_inhslow2pyrinf=np.array([0.25 * c]),
                  p=np.array([0]), sigma=np.array([0]),
                  e0=np.array([0.0025]), r=np.array([0.56]), v0=np.array([6]))

    vPyr_l2l3 = state_variables[0]
    vExc_l4 = state_variables[1]
    vPyr_l5l6 = state_variables[2]
    vInh_fast = state_variables[3]
    vInh_slow = state_variables[4]

    xPyr_l2l3 = state_variables[5]
    xExc_l4 = state_variables[6]
    xPyr_l5l6 = state_variables[7]
    xInh_fast = state_variables[8]
    xInh_slow = state_variables[9]

    lrcFF, lrcFB, srcL, input_u = 0,0,0,0

    # Afferences to subpopulations in FR.
    # S_pyrsup = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["c_exc2pyrsup"] * vExc_l4 - params["c_inhfast2pyrsup"] * vInh_fast))) - params["e0"]
    # S_pyrinf = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["c_pyrsup2pyrinf"] * vPyr_l2l3 - params["c_inhslow2pyrinf"] * vInh_slow))) - params["e0"]
    # S_inhfast = (2 * params["e0"]) / (1 + np.exp(params["r"] * params["c_exc2inhfast"] * vExc_l4)) - params["e0"]
    # S_inhslow = (2 * params["e0"]) / (1 + np.exp(params["r"] * params["c_pyrsup2inhslow"] * vInh_slow)) - params["e0"]

    S_pyrsup = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["v0"] - params["c_exc2pyrsup"] * vExc_l4 - params["c_inhfast2pyrsup"] * vInh_fast)))
    S_pyrinf = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["v0"] - params["c_pyrsup2pyrinf"] * vPyr_l2l3 - params["c_inhslow2pyrinf"] * vInh_slow)))
    S_inhfast = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["v0"] - params["c_exc2inhfast"] * vExc_l4)))
    S_inhslow = (2 * params["e0"]) / (1 + np.exp(params["r"] * (params["v0"] - params["c_pyrsup2inhslow"] * vInh_slow)))

    ## NOTE, for local couplings:
    # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
    # vInh, xInh inhibitory interneurons
    # vPyr, xPyr, vPyr_aux, xPyr_aux pyramidal neurons in deep and superficial layers

    vPyr_l2l3 = xPyr_l2l3
    vExc_l4 = xExc_l4
    vPyr_l5l6 = xPyr_l5l6
    vInh_fast = xInh_fast
    vInh_slow = xInh_slow

    xPyr_l2l3 = params["He"] / params["taue"] * (lrcFB + srcL + S_pyrsup) - (2 * xPyr_l2l3) / params["taue"] - vPyr_l2l3 / params["taue"] ** 2
    xExc_l4 = params["He"] / params["taue"] * (input_u + lrcFF + srcL) - (2 * xExc_l4) / params["taue"] - vExc_l4 / params["taue"] ** 2
    xPyr_l5l6 = params["He"] / params["taue"] * (lrcFB + srcL + S_pyrinf) - (2 * xPyr_l5l6) / params["taue"] - vPyr_l5l6 / params["taue"] ** 2
    xInh_fast = params["Hi_fast"] / params["taui_fast"] * (lrcFB + srcL + S_inhfast) - (2 * xInh_fast) / params["taui_fast"] - vInh_fast / params["taui_fast"] ** 2
    xInh_slow = params["Hi_slow"] / params["taui_slow"] * (lrcFB + srcL + S_inhslow) - (2 * xInh_slow) / params["taui_slow"] - vInh_slow / params["taui_slow"] ** 2

    derivative = [vPyr_l2l3, vExc_l4, vPyr_l5l6, vInh_fast, vInh_slow, xPyr_l2l3, xExc_l4, xPyr_l5l6, xInh_fast, xInh_slow]

    return derivative



init_conds = np.random.rand(10)


# Use fsolve to find the equilibrium points
equilibrium_points = fsolve(dfun, init_conds)
print("Equilibrium Points:", equilibrium_points)






