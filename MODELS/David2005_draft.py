
import numpy as np
import scipy
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

# surfpack = "defsub-r7-surf408_pack\\"
surfpack = "HCPex-r426-surfdisc17k_pack\\"

dataTemp_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\.DataTemp\\"
dataC3N_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\.DataC3N\\"
subj = "N1073"

# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"

## Simulation parameters
simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 500  # ms to exclude from timeseries due to initial transient

iroi = 0
nrois = 426

# STRUCTURAL CONNECTIVITY      #########################################
# conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn = connectivity.Connectivity.from_file(dataC3N_dir + "SC_matrices\\sub_" + subj + "-HCPex-end.zip")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

conn.weights = conn.weights[:, iroi:iroi+nrois][iroi:iroi+nrois]
conn.tract_lengths = conn.tract_lengths[:, iroi:iroi+nrois][iroi:iroi+nrois]
conn.centres = conn.centres[iroi:iroi+nrois]
conn.region_labels = conn.region_labels[iroi:iroi+nrois]
conn.cortical = conn.cortical[iroi:iroi+nrois]


# conn.weights[179] = 0

conn.weights = conn.scaled_weights(mode="region")


## Evaluate the weights and lengths for regions 179: frontal superior language area
# 1 medias sin 0 para todas regions
# wei = [np.average([row[row != 0]]) for row in conn.weights]



# NEURAL MASS MODEL    #########################################################

m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([10]), tau_i=np.array([15]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                       gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# COUPLING    #########################################################
aV = dict(FF=40, FB=60, L=0.05)  # Scaling factor for each type of hierarchical connection

# aM = np.array(   # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
#             # FROM
#     [[0,   "FB", 0, 0, 0, 0, 0],
#      ["FF", 0,  0, 0, 0, 0, 0],
#      [0,   0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0],  # TO
#      [0,    0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0]])

aM = np.loadtxt(dataTemp_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)
aM = aM[:, iroi:iroi+nrois][iroi:iroi+nrois]

aF = np.array([[aV["FF"] if val == "F" else 0 for val in row] for row in aM])
aB = np.array([[aV["FB"] if val == "B" else 0 for val in row] for row in aM])
aL = np.array([[aV["L"]  if val == "L" else 0 for val in row] for row in aM])

coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([3]), aF=aF, aB=aB, aL=aL)


# STIMULUS    #########################################################
weighting = np.zeros((len(conn.region_labels),))
weighting[[0]] = 0.1

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=1, t_start=2000.0, t_end=2025.0)),
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
# integrator = integrators.EulerDeterministic(dt=1000 / samplingFreq)


mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          stimulus=stim)

sim.configure()

output = sim.run(simulation_length=simLength)

data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
time = output[0][0][transient:]

labels = conn.region_labels

if "signals_raw" in viz:
    fig = go.Figure()
    for i, signal in enumerate(data):
        fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file="figures/David2005-signals_raw.html", auto_open=True)

if "timespectra_raw" in viz:
    timeseries_spectra(data, simLength, transient, labels, title="David2005-timespectra_raw", width=1000, height=750)






# SciPy implementation of time-frequency analysis

# from scipy import signal
# from scipy.ndimage import gaussian_filter
#
# # Set parameters for the STFT
# nperseg = 600  # Length (nº datapoints) of each frame in which FFT will be computed. Larger, lower temporal resolution but better frequency resolution.
# overlap = nperseg * 0.95  # Overlap between consecutive frames. Larger, temporally smoother.
# nfft = 3000  # Number of data points used in each block for the FFT. Larger, frequency smoother.
# freqs, time, spectrogram = signal.spectrogram(data[0, :], samplingFreq, nperseg=nperseg, noverlap=overlap, nfft=nfft)
#
# # Smooth the spectrogram using a Gaussian filter
# # Sigma=Standard deviation of the Gaussian filter
# spectrogram = gaussian_filter(spectrogram, sigma=1.5)
#
# spectrogram = spectrogram[freqs < 60]
# freqs = freqs[freqs < 60]
#
# fig = go.Figure(go.Heatmap(x=transient+time*1000, y=freqs, z=spectrogram, colorscale="Jet", colorbar=dict(thickness=5, title="dB")))
# fig.update_layout(template="plotly_white", xaxis=dict(title="Time (ms)"), yaxis=dict(title="Frequency (Hz)"), height=350, width=600)
# fig.show("browser")
