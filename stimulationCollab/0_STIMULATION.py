from tvb.simulator.lab import *
from toolbox import timeseriesPlot, FFTplot, FFTpeaks
import numpy as np
import matplotlib.pyplot as plt

simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000 #Hz

m = models.Generic2dOscillator(I=np.array([5]))

# coup = coupling.Linear(a=np.array([0]), b=np.array([0]))

m = models.WilsonCowan(P=np.array([0.51]), Q=np.array([0]), # Original P at 0.31
                       a_e=np.array([4]), a_i=np.array([4]),
                       alpha_e=np.array([1]), alpha_i=np.array([1]),
                       b_e=np.array([1]), b_i=np.array([1]),
                       c_e=np.array([1]), c_ee=np.array([3.25]), c_ei=np.array([2.5]),
                       c_i=np.array([1]), c_ie=np.array([3.75]), c_ii=np.array([0]),
                       k_e=np.array([1]), k_i=np.array([1]),
                       r_e=np.array([0]), r_i=np.array([0]),
                       tau_e=np.array([10]), tau_i=np.array([20]),
                       theta_e=np.array([0]), theta_i=np.array([0]))

coup = coupling.Linear(a=np.array([0.1]))
transient=500



# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)


f=os.getcwd()+"\\CTB_data\\output\\CTB_connx66_subj04.zip"
conn = connectivity.Connectivity.from_file("paupau.zip") # f | paupau_
mon = (monitors.Raw(),)

##############
##### STIMULUS
## Pulse train
# eqn_t = equations.PulseTrain()
# eqn_t.parameters['onset'] = 1535
# eqn_t.parameters['T'] = 100.0
# eqn_t.parameters['tau'] = 50.0

## Sinusoid input
eqn_t = equations.Sinusoid()
eqn_t.parameters['amp'] = 1
eqn_t.parameters['frequency'] = 10 #Hz
eqn_t.parameters['onset'] = 1000 #ms
eqn_t.parameters['offset'] = 3000 #ms

## Drifting Sinusoid input
# eqn_t = equations.DriftSinusoid()
# eqn_t.parameters['amp'] = 0.1
# eqn_t.parameters['f_init'] = 10 #Hz
# eqn_t.parameters['f_end'] = 20 #Hz
# eqn_t.parameters['onset'] = 500 #ms
# eqn_t.parameters['offset'] = 2000 #ms
# eqn_t.parameters['feedback'] = True # Grow&Shrink (True) - Grow|Shrink(False)
# eqn_t.parameters['sim_length'] = simLength # Grow&Shrink (True) - Grow|Shrink(False)
# eqn_t.parameters['dt'] = 1000/samplingFreq # in ms
# eqn_t.parameters['avg'] = 0.5 #


weighting = np.zeros((4, ))
weighting[[0]] = 0.05

stimulus = patterns.StimuliRegion(
    temporal=eqn_t,
    connectivity=conn,
    weight=weighting)

#Configure space and time
stimulus.configure_space()
stimulus.configure_time(np.arange(0, simLength, 1))
#And take a look
# plot_pattern(stimulus)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=simLength)
# Extract data cutting initial transient
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels
regionLabels=list(regionLabels)
regionLabels.insert(5,"stimulus")

# average signals to obtain mean signal frequency peak
data = np.concatenate((raw_data, stimulus.temporal_pattern), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]
# Check initial transient and cut data
timeseriesPlot(data, raw_time, regionLabels,title= "20HzStim", mode="html")


# Fourier Analysis plot
FFTplot(data, simLength, regionLabels,  mode="html")



fft_peaks = FFTpeaks(raw_data, simLength)[:, 0]

# history=output[0][1][-22:]
# plt.plot(np.arange(0,22,1),history[:,0,0,0].T)