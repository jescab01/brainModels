import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace
import plotly.express as px

# Choose a name for your simulation and define the empirical for SC
model_id = ".1973WilsonCowan"

# Structuring directory to organize outputs
wd=os.getcwd()
main_folder = wd+"\\"+"PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\""PSE" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)


simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024 #Hz
transient = 1000 # ms to exclude from timeseries due to initial transient

# Parameters from Abeysuriya 2018. Using P=0.60 for the nodes to self oscillate at 9.75Hz.
m = models.WilsonCowan(P=np.array([0.60]), Q=np.array([0]),
                       a_e=np.array([4]), a_i=np.array([4]),
                       alpha_e=np.array([1]), alpha_i=np.array([1]),
                       b_e=np.array([1]), b_i=np.array([1]),
                       c_e=np.array([1]), c_ee=np.array([3.25]), c_ei=np.array([2.5]),
                       c_i=np.array([1]), c_ie=np.array([3.75]), c_ii=np.array([0]),
                       k_e=np.array([1]), k_i=np.array([1]),
                       r_e=np.array([0]), r_i=np.array([0]),
                       tau_e=np.array([10]), tau_i=np.array([20]),
                       theta_e=np.array([0]), theta_i=np.array([0]))


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)
conn = connectivity.Connectivity.from_file("paupau_.zip") # ctb_folder+"CTB_connx66_"+emp_subj+".zip" |"paupau.zip"


## Choose coupling and speed following best working points in parameter space explorations
coup = coupling.Linear(a=np.array([0]))

mon = (monitors.Raw(),)

#########################
#########################
## WHAT dynamics do I want to explore?
aim=["fft", "signals"]# "fft" and/or "signals"

## Define the stimulus variable to loop over
stim_freqs = np.arange(1, 80, 1)#[3,12,22,27,45] #Hz  -  np.arange(1, 60, 1)
stim_weights = [0.15, 0.07, 0.03, 0.01]#np.arange(0,0.1,0.001)
# stim_weights = np.concatenate((np.arange(0, 0.0045, 0.0005),
#                                np.arange(0.0045, 0.005, 0.00001),
#                                np.arange(0.005, 0.02, 0.001),
#                                np.arange(0.02, 0.1, 0.01)))# np.arange(0, 0.1, 0.0005)

for w in stim_weights:

    dynamic_fft_data = np.ndarray((1, 5))
    dynamic_signal_data = np.ndarray((1, 5))

    for id, f in enumerate(stim_freqs):

        tic0 = time.time()

        ## Sinusoid input
        eqn_t = equations.Sinusoid()
        eqn_t.parameters['amp'] = 0.2
        eqn_t.parameters['frequency'] = f  # Hz
        eqn_t.parameters['onset'] = 0  # ms
        eqn_t.parameters['offset'] = 5000  # ms
        # if w != 0:
        #     eqn_t.parameters['DC'] = 0.0005 / w

        # Check the index of the region to stimulate and
        weighting = np.zeros((len(conn.weights),))
        weighting[[0]] = w

        stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

        # Configure space and time
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0, simLength, 1))

        # And take a look
        # plot_pattern(stimulus)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
        sim.configure()

        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.4f sec" % (time.time() - tic0,))
        # Extract data cutting initial transient
        raw_data = output[0][1][transient:, 0, :, 0].T
        raw_time = output[0][0][transient:]
        regionLabels = conn.region_labels
        regionLabels = list(regionLabels)
        regionLabels.insert(len(conn.weights[0]) + 1, "stimulus")

        # average signals to obtain mean signal frequency peak
        signals = np.concatenate((raw_data, stimulus.temporal_pattern[:, transient:]), axis=0)

        # save time, signals x3cols,
        if "signals" in aim:
            for i in range(len(signals)):
                temp1 = [[regionLabels[i]] * len(signals[i]), [f] * len(signals[i]), [w] * len(signals[i]), signals[i], raw_time]
                temp1 = np.asarray(temp1).transpose()
                dynamic_signal_data = np.concatenate((dynamic_signal_data, temp1))

        # Check initial transient and cut data
        # timeseriesPlot(raw_data, raw_time, conn.region_labels, main_folder, mode="html")

        # Fourier Analysis plot
        # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")

        #####
        if "fft" in aim:
            for i in range(len(signals)):
                fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
                fft = fft[range(np.int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
                freqs = np.arange(len(signals[i]) / 2)
                freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

                fft = fft[freqs > 0.5]  # remove undesired frequencies from peak analisis
                freqs = freqs[freqs > 0.5]

                temp = [[regionLabels[i]] * len(fft), [f] * len(fft), [w] * len(fft), list(fft), list(freqs)]
                temp = np.asarray(temp).transpose()
                dynamic_fft_data = np.concatenate((dynamic_fft_data, temp))
        #####
        print(str(id + 1) + " out of " + str(len(stim_freqs)))
        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n" % (time.time() - tic0,))





    ## GATHER RESULTS
    title = str(w) + "w"  # "12Hz" | "0.004w"
    simname = "isolated_nodes"+time.strftime("m%md%dy%Y")

    # dynamic ffts
    if "fft" in aim:
        df_fft = pd.DataFrame(dynamic_fft_data[1:, ], columns=["name", "stimfreq",  "weight", "fft", "freqs"])
        df_fft = df_fft.astype({"name": str, "stimfreq": float, "weight": float, "fft": float, "freqs": float})
        df_fft.to_csv(specific_folder+"/PSE_"+simname+"-dynamicFFTdf@"+title+".csv", index=False)

    # dynamic signals
    if "signals" in aim:
        df_s = pd.DataFrame(dynamic_signal_data[1:, ], columns=["name", "stimfreq",  "weight", "signal", "time"])

        df_s = df_s.astype({"name": str, "stimfreq": float, "weight": float, "signal": float, "time":float})
        df_s.to_csv(specific_folder+"/PSE_"+simname+"-dynamicSignalsdf"+title+".csv", index=False)

    # Load previously gathered data
    # df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")


    # Define frequency to explore
    # Plot FFT dynamic
    if "fft" in aim:
        fig = px.line(df_fft, x="freqs", y="fft", animation_frame="stimfreq", animation_group="name", color="name",
                      title="Dynamic FFT @ "+title)
        pio.write_html(fig, file=specific_folder+"/isolatedNodes-f&w_dynFFT_@%s.html" % title)#, auto_open="False")

    # Plot singals dynamic
    if "signals" in aim:
        fig = px.line(df_s, x="time", y="signal", animation_frame="stimfreq", animation_group="name", color="name",
                      title="Dynamic Signals @ "+title)
        pio.write_html(fig, file=specific_folder+"/isolatedNodes-f&w_dynSINGALs_@%s.html" % title)#, auto_open="False")
