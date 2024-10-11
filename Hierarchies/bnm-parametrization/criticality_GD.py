

import os

import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
import time
from mpi4py import MPI
import datetime


from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2005, JansenRit1995


import warnings
warnings.filterwarnings("ignore")

"""
Gradient descent algorithm to adjust BNMs to criticality.
1) adjust globally until having a first node passing criticality
2) adjust (p) per node to critical point

-) Gather relevant info at the end of each process:
emp-sim (FC, dFC, Spectra, avalanches?, rel pow?,

"""

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.Data\\"

    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER BRIGIT
else:
    wd = "/mnt/lustre/home/jescab01/"
    data_dir = wd + "SURFdata/"

    import sys

    sys.path.append(wd)
    from toolbox.fft import FFTpeaks
    from toolbox.mixes import timeseries_spectra




def simulate(simparams, theta, conn):


    simLength = simparams["simLength"]
    transient = simparams["transient"]
    samplingFreq = simparams["samplingFreq"]

    # OTHER PARAMETERS   ###
    integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([0])))  # ms

    mon = (monitors.Raw(),)


    # NEURAL MASS MODEL    #########################################################
    m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                           tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                           gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                           gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                           p=np.array(theta[1:]), sigma=np.array([simparams["sigma"]]),
                           e0=np.array([0.0025]), r=np.array([0.56]))

    m.stvar = np.array([1])  # Define where to input the stimulation


    # COUPLING    #########################################################
    aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)

    aV = dict(F=simparams["ff"], B=simparams["fb"], L=simparams["l"])  # Scaling factor for each type of hierarchical connection

    aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
    aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
    aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

    coup = coupling.SigmoidalJansenRitDavid2005(a=np.array(theta[0]), aF=aF, aB=aB, aL=aL)

    ## RUN SIMULATION
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)

    sim.configure()

    output = sim.run(simulation_length=simLength)

    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSP

    return data


def cost_function(data):
    """
    El costo va a ser el ratio entre la amplitud del primer máximo (max_mv_init),
    y la amplitud del último máximo (max_mv_end)
    si max_mv_init > max_mv_end :: prebif
    si max_mv_init <= max_mv_end :: post bif

    :param data:
    :return:
    """

    maxV_init = np.max(data[:, :1000], axis=1)

    maxV_end = np.max(data[:, -1000:], axis=1)

    cost = 0.25 - maxV_end / maxV_init

    return cost



# STRUCTURAL CONNECTIVITY      #########################################
simpack = "HCPex-r426-surfdisc4k_pack/"

conn = connectivity.Connectivity.from_file(data_dir + simpack + "connectivity.zip")
conn.weights = conn.scaled_weights(mode="tract")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022


# Parameters of the GD
iterations = 10
learning_rate = 1

nrois = len(conn.region_labels)

tic0 = time.time()
print("\n\nInitializing Gradient Descent Algortihm...")

history = {"theta": list(), "cost": list(), "stage": list()}

theta = [45] + [0 for roi in range(nrois)]  # g + regional p
cost = [1 for roi in range(nrois)]


simparams = dict(
    simLength=2500,  # ms
    transient=500,  # ms
    samplingFreq=1000,  # Hz
    ff=0.4, fb=0.6, l=0.05,
    sigma=0
)


# Stage 1 - global fit
it = 0
print("\n\nSTAGE 1: Global adjust over coupling factor (g)_")
while (np.min(cost) > 0.05) and (it <= iterations):

    tic = time.time()
    print('\n Iteration %i  -  Simulating for %s g=%0.2f ' % (it, simpack, theta[0]))

    # Simulate
    history["theta"].append(theta.copy())
    history["stage"].append("stage_1")

    data = simulate(simparams, theta, conn)

    # Cost
    cost = cost_function(data)
    history["cost"].append(cost)

    # Update theta
    update = learning_rate * np.min(cost)
    update = update if update < 5 else 5
    theta[0] = theta[0] + update

    print('    cost %0.2f (min %0.2e) - theta update %0.2f -  time: %0.2f/%0.2f min' % (
        np.average(cost), np.min(cost), update, (time.time() - tic) / 60, (time.time() - tic0) / 60,))

    it += 1


### Plot global fit data

timeseries_spectra(data, simparams["simLength"], simparams["transient"], conn.region_labels, title="timespectra_raw", width=1000, height=750)

# Stage 2 - local fit
# for it in range(iterations):
#
#     while np.sum(cost) > 1e-3:
#
#         print("\n\nSTAGE 2: Regional adjust over intrinsic input parameter (p)_")
#         learning_rate = 0.001
#
#         tic = time.time()
#         print('\n Iteration %i  -  Simulating for %s g=%i  -  simulation Time : '
#               % (it, simpack, theta[0]),end="")
#
#         history["theta"].append(list(theta.T))
#         history["stage"].append(["stage_2"])
#         data = simulate(simparams, theta)
#         print("%0.4f sec" % (time.time() - tic,))
#
#         cost = cost_function(data)
#         history["cost"].append(list(theta.T))
#         print(' cost %0.2f (min %0.2e)  -  time: %0.2f/%0.2f min' % (
#             np.average(cost), np.min(cost), (time.time() - tic) / 60, (time.time() - tic0) / 60,))
#
#         # Update theta
#         update = theta[1:] + (1/nrois) * learning_rate * cost
#         update = [up if up < 5 else 5 for up in update]  # top-up max change
#         theta[1:] = theta[1:] + update
