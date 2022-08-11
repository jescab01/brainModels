
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

import sys
sys.path.append("E:\\LCCN_Local\PycharmProjects\\")  # temporal append
from toolbox.signals import timeseriesPlot
from toolbox.fft import FFTplot, FFTpeaks
# from toolbox import timeseriesPlot, FFTplot, FFTpeaks, multitapper

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = ".2003JansenRitDavid"
wd = os.getcwd()
main_folder = wd+"\\"+subjectid
ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"

emp_subj = "NEMOS_035"
g, s = 17, 12.5

tic0 = time.time()

samplingFreq = 1000  # Hz
simLength = 10000  # ms - relatively long simulation to be able to check for power distribution
transient = 2000  # seconds to exclude from timeseries due to initial transient


conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2.zip")
conn.weights = conn.scaled_weights(mode="tract")

p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0, 0)
sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0, 0)

w = np.array([0.8] * len(conn.region_labels))

# Parameters edited from David and Friston (2003).
m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                         tau_e1=np.array([10]), tau_i1=np.array([16.0]),

                         He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                         tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                         w=np.array([1]), c=np.array([135.0]),
                         c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                         c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                         v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                         p=np.array([0.1085]), sigma=np.array([0]))

# Remember to hold tau*H constant.
m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

# Coupling function
coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
conn.speed = np.array([s])

mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
sim.configure()

output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))
# Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
           (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")
# Fourier Analysis plot
fft = multitapper(raw_data, samplingFreq, regionLabels, 4, 4, 0.5, plot=True, peaks=True)

