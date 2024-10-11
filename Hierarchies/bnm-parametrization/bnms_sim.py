
"""
Testing the surface-based simulations
 - w/ David model it takes TODO x min to simulate x sec
 - and implementing hierarchical connections on the HCP atlas
"""

import os
import time
import pandas as pd
import scipy
import numpy as np
from tvb.simulator.lab import *
import matplotlib.pylab as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995, JansenRitDavid2005

## Folder structure - Local
import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra


surfpack = "HCPex-r426-surfdisc4k_pack\\"

data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.Data\\"
main_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\2bnm-parametrization\\"

# viz options :: signals_raw | signals_avg | timespectra_raw | timespectra_avg |
viz = "timespectra_raw"



# Decide what mode to explore :: classical (reparam; cs) | hierarchy
mode, g,  sigma = "hierarchical", 50, 0
ff, fb, l = 0.5, 0.5, 0  # in case of hierarchical

Hi, taui = 29.3, 16  # Hi=[29.3, 22]; taui=[20,16]

speed = 3.9

nrois = 2  # 426

tic = time.time()

## Simulation parameters
simLength = 2000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

if "hier" in mode:

    # NEURAL MASS MODEL    #########################################################
    m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([Hi]),  # From David (2005)
                           tau_e=np.array([10]), tau_i=np.array([taui]),  # From Lemarechal (2022)
                           gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                           gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                           p=np.array([0]), sigma=np.array([sigma]),
                           e0=np.array([0.0025]), r=np.array([0.56]))

    m.stvar = np.array([1])  # Define where to input the stimulation

    # COUPLING    #########################################################
    aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)
    aM = aM[:, :nrois][:nrois]

    aV = dict(F=ff, B=fb, L=l)  # Scaling factor for each type of hierarchical connection

    aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
    aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
    aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

    coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


elif "classical" in mode:

    # NEURAL MASS MODEL    #########################################################
    m = JansenRit1995(He=np.array([3.25]), Hi=np.array([Hi]),
                      tau_e=np.array([10]), tau_i=np.array([taui]),
                      c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                      c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                      p=np.array([0.1025]), sigma=np.array([sigma]),
                      e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    m.stvar = np.array([1])  # Define where to input the stimulation

    # COUPLING :: Sigmoidal     #########################################################
    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))


# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")

conn.weights = conn.weights[:, :nrois][:nrois]
conn.tract_lengths = conn.tract_lengths[:, :nrois][:nrois]
conn.centres = conn.centres[:nrois]
conn.region_labels = conn.region_labels[:nrois]
conn.cortical = conn.cortical[:nrois]

conn.weights = conn.scaled_weights(mode="tract")
conn.speed = np.array([speed])  # Following Lemarechal et al. 2022 data.


# conn.tract_lengths = np.asarray([[1,1,1], [1,1,1], [1,1,1]])

# ##### STIMULUS
# weighting = np.zeros((len(conn.region_labels),))
# weighting[[361]] = 0.1
#
# stim = patterns.StimuliRegion(
#     # temporal=equations.PulseTrain(
#     #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
#     temporal=equations.DC(parameters=dict(dc_offset=1, t_start=1600.0, t_end=1625.0)),
#     weight=weighting, connectivity=conn)  # In the order of unmapped regions


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
sim.configure()


prep_time = time.time() - tic


output = sim.run(simulation_length=simLength)

if "hier" in mode:
    print("\n(BNM) Simulating %s for %is ended :: time consumed %0.2fs (preparation %0.2fs)\n "
          "(m%s; g%i, F%0.2f, B%0.2f, L%0.2f)\n"
          " Now PLOTTING ... " %
          (surfpack, simLength/1000, time.time()-tic, prep_time, mode, coup.a, aV["F"], aV["B"], aV["L"]))

    title = "%s_g%i-ff%0.2ffb%0.2fl%0.2f" % (mode, coup.a, aV["F"], aV["B"], aV["L"])

else:
    print("\n(BNM) Simulating %s for %is ended :: time consumed %0.2fs (preparation %0.2fs)\n "
          "(m%s; g%i)\n"
          " Now PLOTTING ... " %
          (surfpack, simLength/1000, time.time()-tic, prep_time, mode, coup.a))

    title = "%s_g%i" % (mode, coup.a)

# Plot signals
if "raw" in viz:
    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    time = output[0][0][transient:]

    labels = conn.region_labels

    if "signals_raw" in viz:
        fig = go.Figure()
        for i, signal in enumerate(data):
            fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
        fig.update_layout(template="plotly_white")
        pio.write_html(fig, file=main_dir + "figures/signals_raw.html", auto_open=True)

    if "timespectra_raw" in viz:
        timeseries_spectra(data, simLength, transient, labels, title=title, width=1000, height=750)





