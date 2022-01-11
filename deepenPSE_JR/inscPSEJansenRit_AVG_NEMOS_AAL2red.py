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
from plotly.subplots import make_subplots
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace

# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".1995JansenRit"
emp_subj = "AVG_NEMOS"

wd = os.getcwd()
main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"
if subjectid not in os.listdir(ctb_folder):
    os.mkdir(ctb_folder + subjectid)

specific_folder = main_folder + "\\""inscPSE" + subjectid + "-" + emp_subj + "-" + time.strftime(
    "m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

scales = [ 4, 6]

# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1024  # Hz
transient = 500  # ms
n_rep = 1

# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

fig = make_subplots(rows=1, cols=2, subplot_titles=("e-4",  "e-6"),
                    specs=[[{}, {}]], shared_yaxes=True, shared_xaxes=True,
                    x_title="Conduction speed (m/s)", y_title="Coupling factor")

for j, scale in enumerate(scales):
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red_in-%i.zip" % scale)
    conn.weights = conn.scaled_weights(mode="tract")

    mon = (monitors.Raw(),)

    coupling_vals = np.arange(0, 120, 1)
    speed_vals = np.arange(0.5, 25, 1)
    # coupling_vals = np.arange(0, 2, 1)
    # speed_vals = np.arange(0.5, 1, 1)

    results_fft_peak = list()
    results_fc = list()

    for g in coupling_vals:
        for s in speed_vals:
            for r in range(n_rep):

                coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                                   r=np.array([0.56]))
                conn.speed = np.array([s])
                tic = time.time()
                print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

                # Run simulation
                sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator,
                                          monitors=mon)
                sim.configure()
                output = sim.run(simulation_length=simLength)

                # Extract data: "output[a][b][:,0,:,0].T" where:
                # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
                raw_data = output[0][1][transient:, 0, :, 0].T
                raw_time = output[0][0][transient:]

                # average signals to obtain mean signal frequency peak
                data = np.asarray([np.average(raw_data, axis=0)])
                data = np.concatenate((data, raw_data),
                                      axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

                # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
                results_fft_peak.append((g, s, r, FFTpeaks(data, simLength - transient)[0][0][0],
                                         FFTpeaks(data, simLength - transient)[1][0][0],
                                         FFTpeaks(data, simLength - transient)))

                newRow = [g, s, r]
                bands = [["3-alpha"], [(8, 12)]]
                # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]
                for b in range(len(bands[0])):
                    (lowcut, highcut) = bands[1][b]

                    # Band-pass filtering
                    filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

                    # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
                    efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

                    # Obtain Analytical signal
                    efPhase = list()
                    efEnvelope = list()
                    for i in range(len(efSignals)):
                        analyticalSignal = scipy.signal.hilbert(efSignals[i])
                        # Get instantaneous phase and amplitude envelope by channel
                        efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                        efEnvelope.append(np.abs(analyticalSignal))

                    # CONNECTIVITY MEASURES
                    ## PLV
                    plv_sim = PLV(efPhase)

                    # Comparisons
                    t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
                    t1[0, :] = conn.weights[np.triu_indices(len(conn.region_labels), 1)]
                    t1[1, :] = plv_sim[np.triu_indices(len(conn.region_labels), 1)]
                    plv_r = np.corrcoef(t1)[0, 1]
                    newRow.append(plv_r)

                results_fc.append(newRow)

                print("ROUND = %i " % r)
                print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))

    ## GATHER RESULTS
    simname = "in-" + str(scale) + "sc-simfc" + subjectid + "-" + emp_subj + "-t" + str(
        simLength) + "-" + time.strftime("m%md%dy%Y")
    # Working on FFT peak results
    df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed", "round", "mS_peak", "mS_module", "allSignals"])
    df1.to_csv(specific_folder + "/PSE_FFTpeaks" + simname + ".csv", index=False)
    paramSpace(df1, title=simname, folder=specific_folder)

    # Working on FC results
    df = pd.DataFrame(results_fc, columns=["G", "speed", "round", "Alpha"])
    df.to_csv(specific_folder + "/PSE_FC" + simname + ".csv", index=False)
    # paramSpace(df, 0.5, simname+"_PLV", folder=specific_folder)

    if scale == 1:
        fig.add_trace(go.Heatmap(z=df.Alpha, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                                 reversescale=True, zmin=-0.5, zmax=0.5), row=1, col=1)
    else:
        fig.add_trace(
            go.Heatmap(z=df.Alpha, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                       showscale=False), row=1, col=j + 1)

fig.update_layout(
    title_text='empSC-simFC by Coupling factor and Conduction speed')
pio.write_html(fig, file=specific_folder + "/paramSpace-g&s_inSC.html", auto_open=True)
