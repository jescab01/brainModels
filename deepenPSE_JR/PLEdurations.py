import time

import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd

from tvb.simulator.lab import *
from mne import filter
from toolbox import PLE, epochingTool



# Common simulation requirements
subjectid = ".1995JansenRit"
emp_subj = "AVG_NEMOS"

ple_matrices = list()
ple_parameters = list()

for s in [10,20,30]:
    # Prepare simulation parameters
    simLength = s * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms
    epoch_length=4000

    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    mon = (monitors.Raw(),)

    g, s, r = 15, 15, 1

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([s])

    tic = time.time()

    print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, epoch_length//1000, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))



    ## PLE - Phase Lag Entropy
    ## PLE parameters - Phase Lag Entropy
    tau_ = 25  # ms
    m_ = 3  # pattern size
    # IF it takes too long use subsampling: choose one point out of each "subsampling"=5
    # Some tests - (subsample, time, patterns): (20, 5m, 197p), (100, 81s, 35p), (150, 63s, 27p),

    # subsample points to test
    subs=np.arange(5, 150, 20)

    for s in subs:
        tic=time.time()
        ple, _ = PLE(efPhase, tau_, m_, samplingFreq, subsampling=s)
        ple_matrices.append(ple)

        n_patterns=(epoch_length-tau_*m_)/s
        ple_parameters.append([simLength, samplingFreq, epoch_length, n_patterns, s, time.time()-tic, np.average(ple[np.triu_indices(len(ple[0]), 1)])])


df=pd.DataFrame(ple_parameters, columns=["simLength", "SamplingFreq", "epoch", "patts", "subsample", "time", "ple_avg"])

df["time"]=df["time"]/60
import plotly.express as px

fig= px.line(df, x="subsample", y="ple_avg", color="simLength")
fig.update_traces(mode='lines+markers')
fig.show(renderer="browser")