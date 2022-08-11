
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

import sys
sys.path.append("E:\\LCCN_Local\PycharmProjects\\")  # temporal append
from toolbox.signals import timeseriesPlot
from toolbox.fft import FFTplot, FFTpeaks, multitapper
from toolbox.mixes import timeseries_spectra


def simulate(model, w=1, sigma=0, sigma_mode=None, title=None):

    tic0 = time.time()

    # This simulation will generate FC for a virtual "Subject".
    # Define identifier (i.e. could be 0,1,11,12,...)
    ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"

    emp_subj = "NEMOS_035"
    g, s = 17, 12.5

    figures_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\\figures"

    simLength = 6000  # ms - relatively long simulation to be able to check for power distribution
    transient = 2000  # seconds to exclude from timeseries due to initial transient
    samplingFreq = 1000  # Hz


    conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

    mon = (monitors.Raw(),)

    #### Thalamus noise
    if sigma_mode == "thalamus":
        # p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0, 0)
        sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), sigma, 0)
    elif sigma_mode == "all":
        sigma_array = np.array([sigma] * len(conn.region_labels))
    else:
        sigma_array = np.array([0] * len(conn.region_labels))

    w_array = np.array([w] * len(conn.region_labels))

    conn.speed = np.array([s])

    if model=="jrd":
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                 tau_e1=np.array([10]), tau_i1=np.array([16.0]),

                                 He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                 tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                 w=w_array, c=np.array([135.0]),
                                 c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                 c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                 v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                 p=np.array([0.22]), sigma=sigma_array)

        # Remember to hold tau*H constant.
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        # Coupling function
        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

    elif model=="jr":
        #### Simulate with myJR with stefanovski parameters
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([16]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),
                          p=np.array([0.22]), sigma=sigma_array)

        # Remember to hold tau*H constant.
        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])
        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))


    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=simLength)
    print("Simulation time: %0.2f sec" % (time.time() - tic0,))

    if model=="jrd":
        raw_data = m.w * output[0][1][transient:, 0, :, 0].T + (1 - m.w) * output[0][1][transient:, 4, :, 0].T

    elif model=="jr":
        raw_data = output[0][1][transient:, 0, :, 0].T

    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels


    timeseries_spectra(raw_data, simLength, transient, regionLabels, mode="html", folder=figures_folder,
                       title=title, auto_open=True)
    # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, regionLabels, mode="html", folder=figures_folder)
    # # Fourier Analysis plot
    # FFTplot(raw_data, simLength, regionLabels, folder=figures_folder, title="hello", mode="html", max_hz=80, min_hz=1, type="linear", auto_open=True)
    # # fft = multitapper(raw_data, samplingFreq, regionLabels, 4, 4, 0.5, plot=True, peaks=True, folder=figures_folder)

    print("Simulation time: %0.2f" % (time.time()-tic0,))
    return raw_data


### No-Noise simulations
signals_jr=simulate("jr",  sigma=0, sigma_mode="all", title="jr_noNoise")
jr_meanAmp = np.average([[max(s) - np.min(s)] for s in signals_jr])

signals_jrd = simulate("jrd", w=0.8, sigma_mode="all", sigma=0, title="jrd_noNoise")


## Noise simulations
signals_jrd = simulate("jrd", w=0.8, sigma_mode="all", sigma=0.022, title="jrd_Noise")
jrd_meanAmp = np.average([[max(s) - np.min(s)] for s in signals_jrd])

# Calculate proportional noise for jr model (sigma_jr)
sigma_jr = 0.022 * jr_meanAmp / jrd_meanAmp
signals_jr = simulate("jr",  sigma=sigma_jr, sigma_mode="all", title="jr_Noise")


## Just Thalamic noise simulations
signals_jrd = simulate("jrd", w=0.8, sigma_mode="thalamus", sigma=0.022, title="jrd_thNoise")

signals_jr = simulate("jr",  sigma=sigma_jr, sigma_mode="thalamus", title="jr_thNoise")



