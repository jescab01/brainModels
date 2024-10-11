import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
from mne.time_frequency import tfr_array_morlet

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995, JansenRitDavid2005, JansenRit1995_hierarchical

## Folder structure - Local
import sys

sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra


# surfpack = "defsub-r7-surf408_pack\\"
surfpack = "HCPex-r426-surfdisc17k_pack\\"
data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\.DataTemp\\"



# viz options :: (signals | timespectra) raw | tfr
viz = "timespectra_tfr"
nrois = 90


## Simulation parameters
simLength = 10000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

conn.weights = conn.weights[:, :nrois][:nrois]
conn.tract_lengths = conn.tract_lengths[:, :nrois][:nrois]
conn.centres = conn.centres[:nrois]
conn.region_labels = conn.region_labels[:nrois]
conn.cortical = conn.cortical[:nrois]

conn.weights = conn.scaled_weights(mode="region")



# conn.weights = np.array(
#             # FROM
#     [[0, 1, 0, 0, 0, 0, 0],
#      [1, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],  # TO
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0]])



# NEURAL MASS MODEL    #########################################################
m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                  tau_e=np.array([10]), tau_i=np.array([16]),
                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                  p=np.array([0.22]), sigma=np.array([0.022]),
                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

m.stvar = np.array([1])  # Define where to input the stimulation


# COUPLING    #########################################################

coup = coupling.SigmoidalJansenRit(a=np.array([40]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))


# STIMULUS    #########################################################
weighting = np.zeros((len(conn.region_labels),))
weighting[[0]] = 0.1

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=0.1, t_start=2000.0, t_end=2025.0)),
    weight=weighting, connectivity=conn)  # In the order of unmapped regions


# ## REGIONAL
# stimulus_regional = patterns.StimuliRegion(
#     connectivity=conn,
#     weight=weighting)
#
# # # Configure space and time
# # stimulus_regional.configure_space()
# # stimulus_regional.configure_time(np.arange(0, simLength, 1))

# #And take a look
# # plot_pattern(stimulus_surface)


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          stimulus=stim)

sim.configure()

output = sim.run(simulation_length=simLength)

data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
time = output[0][0][transient:]

labels = conn.region_labels

if "signals" in viz:
    fig = go.Figure()
    for i, signal in enumerate(data):
        fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file="figures/JR1995-signals_raw.html", auto_open=True)

if "timespectra" in viz:
    timeseries_spectra(data, simLength, transient, labels, title="JR1995-timespectra_raw", width=1000, height=750)

## TFR
if "tfr" in viz:

    freqs = np.arange(2, 60, 0.25)

    ## n_cycles determine the length of the wavelet.
    tfr = tfr_array_morlet(data[np.newaxis, 0:1, :], samplingFreq, freqs, zero_mean=True, n_cycles=7, output="power")
    tfr_single = tfr[0][0]  # tfr - [epoch,channel,freq,time]


    fig = go.Figure(go.Heatmap(x=time[2000:-2000], y=freqs, z=tfr_single[:, 2000:-2000], colorscale="Jet", colorbar=dict(thickness=10)))
    fig.update_layout(xaxis=dict(title="Time (s)"), yaxis=dict(title="Frequency (Hz)"),
                      width=900, height=400, title="Simulated TFR for " + surfpack)
    pio.write_html(fig, file="figures\\TFR_sim_" + surfpack[:-2] + ".html", auto_open=True)
