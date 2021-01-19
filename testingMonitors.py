# import os
import time
# import subprocess

import numpy as np
# import scipy.signal
# import pandas as pd
# import scipy.stats

from tvb.simulator.lab import *
# from mne import time_frequency, filter
# import plotly.graph_objects as go  # for data visualisation
# import plotly.io as pio
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = ".1995JansenRit"
wd=os.getcwd()
main_folder=wd+"\\"+subjectid
ctb_folder=wd+"\\CTB_data\\output\\"
if subjectid not in os.listdir(ctb_folder):
    os.mkdir(ctb_folder+subjectid)
    os.mkdir(main_folder)

emp_subj = "subj04"

# Prepare bimodality test (i.e. Hartigans' dip test in an external R script via python subprocess
# Build subprocess command: [Rscript, script]
# cmd = ['C:\\Program Files\\R\\R-3.6.1\\bin\\Rscript.exe',
#        'C:\\Users\\F_r_e\\PycharmProjects\\brainModels\\diptest\\diptest.R']

tic0=time.time()

simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000 #Hz
transient=1000 # ms to exclude from timeseries due to initial transient

m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),

                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.05]),

                     mu=np.array([0.09]), nu_max=np.array([0.0025]), p_max=np.array([0.15]), p_min=np.array([0.3]),

                     r=np.array([0.56]), v0=np.array([6]))


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

# conn = connectivity.Connectivity.from_file(ctb_folder+"CTB_connx66_"+emp_subj+".zip")
conn = connectivity.Connectivity.from_file("connectivity_192.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Coupling function
coup = coupling.SigmoidalJansenRit(a=np.array([28]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))
conn.speed=np.array([6])

###########
# MONITORS
##########
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG, ProjectionSurfaceMEG
from tvb.datatypes.sensors import SensorsEEG, SensorsMEG

rm = RegionMapping.from_file('regionMapping_16k_192.txt')
sensorsEEG = SensorsEEG.from_file('eeg_unitvector_62.txt.bz2')
prEEG = ProjectionSurfaceEEG.from_file('projection_eeg_62_surface_16k.mat', matlab_data_name="ProjectionMatrix")

# sensorsMEG = SensorsMEG.from_file("meg_151.txt.bz2")
# prMEG = ProjectionSurfaceMEG.from_file("projection_meg_276_surface_16k.npy")


fsamp = 1e3/1024.0 # 1024 Hz

local_coupling_strength = np.array([2 ** -10])
region_mapping_data=RegionMapping.from_file('regionMapping_16k_192.txt')
region_mapping_data.surface=Surface.from_file()
default_cortex = Cortex.from_file()
default_cortex.region_mapping_data = region_mapping_data
default_cortex.coupling_strength = local_coupling_strength

mon = (monitors.Raw(),monitors.MEG.from_file())
       #monitors.EEG(sensors=sensorsEEG, projection=prEEG, region_mapping=rm, period=fsamp))
       # monitors.MEG(sensors=sensorsMEG, projection=prMEG, region_mapping=rm, period=fsamp),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
sim.configure()

output, EEG = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))
# Extract data cutting initial transient
raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels
regionLabels=list(regionLabels)
regionLabels.insert(0,"AVG")

# average signals to obtain mean signal frequency peak
data = np.asarray([np.average(raw_data, axis=0)])
data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

# Check initial transient and cut data
timeseriesPlot(data, raw_time, regionLabels, main_folder)

# Fourier Analysis plot
FFTplot(data, simLength, regionLabels, main_folder)

fft_peaks = FFTpeaks(data, simLength - transient)[:, 0]


