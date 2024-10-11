
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
main_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\1Hierarchies\\"

# viz options :: signals_raw | signals_avg | timespectra_raw | timespectra_avg |
viz = "timespectra_raw"


# TODO initial conditions in zero?

tic = time.time()

## Simulation parameters
simLength = 2000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 500  # ms to exclude from timeseries due to initial transient



# NEURAL MASS MODEL    #########################################################
m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([128]), gamma2_exc2pyr=np.array([102]),  # From Kiebel (2008)
                       gamma3_pyr2inh=np.array([32]), gamma4_inh2pyr=np.array([32]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.weights = conn.scaled_weights(mode="tract")

conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022



# COUPLING :: Hierarchical      #########################################################

aV = dict(F=32, B=16, L=4)  # Scaling factor for each type of hierarchical connection

aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)

aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([0.25]), aF=aF, aB=aB, aL=aL)



##### STIMULUS
weighting = np.zeros((len(conn.region_labels),))
weighting[[0]] = 0

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=1, t_start=600.0, t_end=625.0)),
    weight=weighting, connectivity=conn)  # In the order of unmapped regions


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,
                          integrator=integrator, monitors=mon, stimulus=stim)
sim.configure()


prep_time = time.time() - tic


output = sim.run(simulation_length=simLength)

print("\n(BNM) Simulating %s for %is ended :: time consumed %0.2fs (preparation %0.2fs)\n (g%i, F%0.2f, B%0.2f, L%0.2f)\n"
      " Now PLOTTING. Takes time, be patient. " %
      (surfpack, simLength/1000, time.time()-tic, prep_time, coup.a, aV["F"], aV["B"], aV["L"]))



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
        timeseries_spectra(data, simLength, transient, labels, title="timespectra_raw", width=1000, height=750)





