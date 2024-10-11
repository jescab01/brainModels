

import numpy as np
import pandas as pd
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
import time

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995, JansenRitDavid2005

## Folder structure - Local
import sys

sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra
from toolbox.fft import FFTpeaks

data_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\\data\\"

out_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\\Models\\PSE\\"


# PSE parameters

sim = "r2-FF_FB"


FF_vals = np.arange(0, 100, 2)
FB_vals = np.arange(0, 100, 2)

params = [(ff, fb) for ff in FF_vals for fb in FB_vals]





## Simulation parameters
simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# STRUCTURAL CONNECTIVITY      #########################################
surfpack = "subset-r7-surf408_pack\\"
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022
conn.weights = np.ones(shape=conn.weights.shape)


# NEURAL MASS MODEL    #########################################################
m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                       gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# STIMULUS    #########################################################
weighting = np.zeros((len(conn.region_labels),))
weighting[[0]] = 0.1

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=1, t_start=2000.0, t_end=2025.0)),
    weight=weighting, connectivity=conn)  # In the order of unmapped regions

# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)






### ITERATE OVER THE PARAMETER SPACE
results = []

for i, set_ in enumerate(params):

    tic = time.time()

    ff, fb = set_

    print("Simulating for FF%i and FB%i  ::  %i/%i " % (ff, fb, i + 1, len(params)))

    # COUPLING    #########################################################
    aV = dict(FF=ff, FB=fb, L=0)  # Scaling factor for each type of hierarchical connection

    aM = np.array(   # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
        [[0,   "FB", 0, 0, 0, 0, 0],
         ["FF", 0,  0, 0, 0, 0, 0],
         [0,    0,  0, 0, 0, 0, 0],
         [0,    0,  0, 0, 0, 0, 0],  # TO
         [0,    0,  0, 0, 0, 0, 0],
         [0,    0,  0, 0, 0, 0, 0],
         [0,    0,  0, 0, 0, 0, 0]])

    aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
    aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
    aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

    coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([1]), aF=aF, aB=aB, aL=aL)


    ## RUN SIMULATION
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                              stimulus=stim)

    sim.configure()

    output = sim.run(simulation_length=simLength)

    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSP
    raw_time = output[0][0][transient:]

    # Gather spectral data
    peaks_pre, _, band_modules_pre = FFTpeaks(data[:, :1000], transient+1000, transient, samplingFreq)
    peaks_post, _, band_modules_post = FFTpeaks(data[:, 1000:2000], transient+1000, transient, samplingFreq)



    # Gather duration
    signal_peaks = scipy.signal.find_peaks(data[0, 1000:2000])[0]
    signal_peaks_amps = [data[0, int(1000+peak_id)] for peak_id in signal_peaks]

    ids = np.array(signal_peaks)[signal_peaks_amps > signal_peaks_amps[0]*0.05]

    duration = max(raw_time[1000:2000][ids])-2000  # duration in ms

    results.append([ff, fb,
                    peaks_pre[0], peaks_pre[1], peaks_post[0], peaks_post[1],
                    band_modules_pre[0], band_modules_pre[1], band_modules_post[0], band_modules_post[1],
                    duration])

    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))


# Create and save dataframe
import pandas as pd
df = pd.DataFrame(results, columns=["FF", "FB", "peaks_pre_r0", "peaks_pre_r1",
                                        "peaks_post_r0", "peaks_post_r1",
                                        "band_modules_pre_r0", "band_modules_pre_r1",
                                        "band_modules_post_r0", "band_modules_post_r1",
                                        "duration"])
df.to_csv("out")

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
