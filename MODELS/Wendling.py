
"""
Test wedling 2002 implementation.
Bifurcation diagram

Notes:
    - I saved a plot (figures/Aperiodic_wAlpha.html) simulated without removing the transient,
    with nodes oscillating in alpha p=0.22, noise=0.15. Lo que provoca el aperiodico es la potencia del
    transiente. ¿Quizá lo que buscas en el "aperiodico" [no es realmente aperiodico] son fluctuaciones grandes que provienen de
    neurotransmisión, a cierto ritmo?

"""

import os
import time
import numpy as np
from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import Wendling2002

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline

import warnings  # Only for the workshop; you could hide important messages.
warnings.filterwarnings("ignore")

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fft import FFTpeaks, multitapper, PSDplot
    from toolbox.mixes import timeseries_spectra, timeseries_phaseplane



wd = os.getcwd()  # Define working directory
tic0 = time.time()  # Measure simulation time



timescale, simLength, transient_t = "ms", 20000, 1000  # simulated timesteps in the NMM timescale - ms; transient time to exclude from analysis
samplingFreq = 1  # Hz - datapoints/timestep
transient_dp = transient_t * samplingFreq  # convert transient time into number of datapoints


## A1. Structural connectivity
conn = connectivity.Connectivity.from_file("paupau.zip")  # Loads structural connectivity
conn.weights = conn.scaled_weights(mode="tract")  # Scales number of streamlines using the max
conn.speed = np.array([3.9])  #(m/s) Defines conduction speed shaping delays

conn.weights = np.array([[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0]])

conn.tract_lengths = np.ones((4, 4)) * 50

## A2. Neural Mass Model
m = Wendling2002(He=np.array([3.25]), Hi=np.array([22]), Hif=np.array([10]),
                  tau_e=np.array([10]), tau_i=np.array([20]), tau_if=np.array([2]),
                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),

                  p=np.array([0.22, 0, 0, 0]), sigma=np.array([0.15, 0, 0.15, 0]),

                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

# A3. Coupling function - How NMMs will link through structural connectivity (what variable to use and transformations to apply)
coup = coupling.SigmoidalWendling(a=np.array([10]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))

# A4. integrator: dt = timestep/datapoint = 1/samplingFreq(Hz=datapoints/timestep) -with timestep in NMM's timescale-.
integrator = integrators.HeunStochastic(dt=1/samplingFreq, noise=noise.Additive(nsig=np.array([0])))

# A5. Monitor - what information to extract / apply transformation (e.g. BOLD, EEG)?
mon = (monitors.Raw(), )



# B1. Configure simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
sim.configure()

# B2. Run simulation
output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec  |  %0.5f sec/sim.%s" %
      (time.time() - tic0, (time.time() - tic0)/simLength, timescale))



# C1. Extract data cutting initial transient
# output[monitor_id][time=0|data=1][timepoints, stateVars, ROIs, modes]
raw_data = output[0][1][transient_dp:, 0, :, 0].T
raw_time = output[0][0][transient_dp:]

# C2. Plot signals and spectra
timeseries_spectra(raw_data, simLength, transient_t, conn.region_labels, timescale=timescale, mode="html")

# multitapper(raw_data, 1000, conn.region_labels, smoothing=100, plot=True, mode="html")

PSDplot(raw_data, 1000, conn.region_labels, title="test", highcut=200, overlap=0.5)
