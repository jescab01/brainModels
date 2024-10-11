

import numpy as np
from tvb.simulator.lab import *

import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2005

## Folder structure - Local
import sys

sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\IntegrativeRhythms\Models\PSE\\'
simulations_tag = "PSEmpi_4hierarchies-FF_FB-m02d29y2024-t11h.32m.51s"  # Tag cluster job

surfpack = "subset-r7-surf408_pack\\"
# surfpack = "default-r76-surf16k_pack\\"

data_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\\SURFdata\\"
# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"

## Simulation parameters
simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# STRUCTURAL CONNECTIVITY      #########################################
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



modes = ["r2_g1", "r3lin_g0.5", "r3all_g0.35", "r5"]

# Define a series of 3 points per mode that would be interesting to look-up
poi = [[(45, 40), (40, 40), (35, 35), (25, 20), (15, 15)],
       [(60, 55), (50, 50), (30, 25)],
       [(40, 98), (40, 85), (40, 60), (40, 35)],
       [(98, 98), (85, 85), (60, 50), (50, 40)]]


for i, mode in enumerate(modes):
    for (ff, fb) in poi[i]:

        title = mode + "-ff" + str(ff) + "fb" + str(fb)

        # COUPLING    #########################################################
        aV = dict(FF=ff, FB=fb, L=0)  # Scaling factor for each type of hierarchical connection

        if "r2" in mode:
            r1, g = 1, 1
            aM = np.array(   # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                        # FROM
                [["d",   "FB", 0, 0, 0, 0, 0],
                 ["FF", "d",  0, 0, 0, 0, 0],
                 [0,    0,  "d", 0, 0, 0, 0],
                 [0,    0,  0, "d", 0, 0, 0],  # TO
                 [0,    0,  0, 0, "d", 0, 0],
                 [0,    0,  0, 0, 0, "d", 0],
                 [0,    0,  0, 0, 0, 0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)

        elif "r3lin" in mode:
            r1, g = 2, 0.5
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d",   "FB", 0, 0, 0, 0, 0],
                 ["FF", "d",  "FB", 0, 0, 0, 0],
                 [0,   "FF", "d", 0, 0, 0, 0],
                 [0,    0,   0, "d", 0, 0, 0],  # TO
                 [0,    0,   0, 0, "d", 0, 0],
                 [0,    0,   0, 0, 0, "d", 0],
                 [0,    0,   0, 0, 0, 0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


        elif "r3all" in mode:
            r1, g = 2, 0.35
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d", "FB", "FB", 0, 0, 0, 0],
                 ["FF", "d", "FB", 0, 0, 0, 0],
                 ["FF", "FF", "d", 0, 0, 0, 0],
                 [0, 0, 0, "d", 0, 0, 0],  # TO
                 [0, 0, 0, 0, "d", 0, 0],
                 [0, 0, 0, 0, 0, "d", 0],
                 [0, 0, 0, 0, 0, 0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


        elif "r5" in mode:
            r1, g = 4, 1/8
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d",  "FB", "FB", "FB",  0,   0,  0],
                 ["FF", "d",  "FB",  0,   "FB", 0,  0],
                 ["FF", "FF", "d",  "FF", "FB", 0,  0],
                 ["FF",  0,   "FB", "d",  "FB", 0,  0],  # TO
                 [0,    "FF", "FF", "FF", "d",  0,  0],
                 [0,     0,    0,    0,    0,  "d", 0],
                 [0,     0,    0,    0,    0,   0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)

        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stim)

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
            timeseries_spectra(data, simLength, transient, labels, title=title, folder=main_folder+simulations_tag)






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





