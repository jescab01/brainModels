

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
main_folder = 'E:\\LCCN_Local\PycharmProjects\\BrainRhythms\Hierarchies\Complete\PSE\\'
simulations_tag = "PSEmpi_hierarchiesProgressive-m04d13y2024-t22h.47m.45s"  # Tag cluster job


surfpack = "HCPex-r426-surfdisc4k_pack/"


data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\\.Data\\"
# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"


## Simulation parameters
simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient



# NEURAL MASS MODEL    #########################################################

m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                       gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)



# Define a series of points that would be interesting to look-up
# poi = [[(0.1, 10, 10, 5), (0.1, 10, 20, 5), (0.1, 50, 50, 5), (0.1, 70, 70, 5), (0.1, 70, 70, 0)]]

# poi = [(7, "bin", 1, 25, 15, 0), (7, "bin", 1, 15, 18, 0), (7, "bin", 1, 18, 6, 0),
#        (7, "tract", 1, 30, 30, 0), (7, "tract", 1, 17, 17, 0)]

poi = [(426, "tract", 1, 90, 90, 0), (426, "tract", 1, 80, 80, 0), (426, "tract", 1, 70, 70, 0),
       (426, "tract", 1, 60, 60, 0), (426, "tract", 1, 50, 50, 0),
       (426, "tract", 1, 70, 70, 10), (426, "tract", 1, 70, 70, 25), (426, "tract", 1, 70, 70, 50)]

for (n_rois, sc_mode, g, ff, fb, l) in poi:

    title = "Hierarchical-r%i%s-F%iB%iL%i" % (n_rois, sc_mode, ff, fb, l)

    # STRUCTURAL CONNECTIVITY      #########################################
    conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
    conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

    # Subset the relevant data to the number of regions in mode
    conn.weights = conn.weights[:, :n_rois][:n_rois]
    conn.tract_lengths = conn.tract_lengths[:, :n_rois][:n_rois]
    conn.centres = conn.centres[:n_rois]
    conn.region_labels = conn.region_labels[:n_rois]
    conn.cortical = conn.cortical[:n_rois]

    # Transform weights
    if ("tract" or "region") in sc_mode:
        conn.weights = conn.scaled_weights(mode=sc_mode)
    else:
        conn.weights = conn.binarized_weights


    # COUPLING    #########################################################
    aV = dict(F=ff, B=fb, L=l)  # Scaling factor for each type of hierarchical connection

    aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy.txt', dtype=str)
    aM = aM[:, :n_rois][:n_rois]

    aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
    aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
    aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

    coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)

    # STIMULUS    #########################################################
    weighting = np.zeros((len(conn.region_labels),))
    weighting[[0]] = 1

    stim = patterns.StimuliRegion(
        # temporal=equations.PulseTrain(
        #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
        temporal=equations.DC(parameters=dict(dc_offset=0.01, t_start=2000.0, t_end=2025.0)),
        weight=weighting, connectivity=conn)  # In the order of unmapped regions

    ## RUN SIMULATION
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
        timeseries_spectra(data, simLength, transient, labels, title=title,
                           folder=main_folder+simulations_tag, width=1050, height=750)






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





