

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
main_folder = 'E:\\LCCN_Local\PycharmProjects\\BrainRhythms\Hierarchies\Compw_brain\PSE\\'
simulations_tag = "PSEmpi_hierarchiesBrain-m03d19y2024-t06h.01m.02s"  # Tag cluster job


surfpack = "HCPex-r426-surfdisc4k_pack/"
data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\\.Data\\"

# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"


## Simulation parameters
simLength = 2000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 500  # ms to exclude from timeseries due to initial transient


# NEURAL MASS MODEL    #########################################################

m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                       gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.weights = conn.scaled_weights(mode="tract")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)


# Define a series of points that would be interesting to look-up
hier_vals = [(90, 50), (60, 60), (50, 90),  # L = 1/2 * min (FF, FB)
             (75, 20), (40, 40), (20, 75),
             (65, 15), (25, 25), (10, 70),
             (18, 18)]

rois = [0, 289, 179]

poi = [(70, 110, 5, rois)] + [(vals[0], vals[1], int(min(vals)/2), rois) for vals in hier_vals]


for (ff, fb, l, rois) in poi:

    for roi in rois:

        title = "FF%iFB%iL%i_timeseries-r%i" % (ff, fb, l, roi)

        # COUPLING    #########################################################
        aV = dict(F=ff, B=fb, L=l)  # Scaling factor for each type of hierarchical connection

        aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy.txt', dtype=str)

        aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
        aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
        aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

        coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([1]), aF=aF, aB=aB, aL=aL)

        # STIMULUS    #########################################################
        weighting = np.zeros((len(conn.region_labels),))
        weighting[[roi]] = 0.1

        stim = patterns.StimuliRegion(
            # temporal=equations.PulseTrain(
            #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
            temporal=equations.DC(parameters=dict(dc_offset=1, t_start=600.0, t_end=625.0)),
            weight=weighting, connectivity=conn)  # In the order of unmapped regions

        ## RUN SIMULATION
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stim)

        sim.configure()

        output = sim.run(simulation_length=simLength)

        data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
        time = output[0][0][transient:]

        labels = [str(i) + '-' + roi for i, roi in enumerate(conn.region_labels)]

        if "timespectra_raw" in viz:
            timeseries_spectra(data, simLength, transient, labels, title=title, folder=main_folder + simulations_tag,
                               width=1000, height=750, auto_open=True)






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





