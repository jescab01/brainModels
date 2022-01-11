import os
import time
# import subprocess

import numpy as np
import scipy.signal
# import pandas as pd
# import scipy.stats

from tvb.simulator.lab import connectivity
from mne import filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from toolbox import epochingTool, PLE

results_ple=list()
patts=list()

for i in [35,49,50,58,59,64,65,71,75,77]:

    # Define the name of the NMM to test
    # and the connectome to use from the available subjects
    # subjectid = ".1995JansenRit"
    emp_subj = "NEMOS_0" + str(i)
    samplingFreq = 1000  # Hz

    wd = os.getcwd()
    ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"
    emp_signals = np.loadtxt(ctb_folder + 'FC_' + emp_subj + '/.RawSignals.txt', delimiter=",")  # Lasts half a minute

    conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")


    # Validate PLV matrics in front of empirical PLV from Brainstorm
    plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "\\3-alpha_plv.txt")
    (lowcut, highcut) = (8, 12)

    # Band-pass filtering
    filterSignals = filter.filter_data(emp_signals, samplingFreq, lowcut, highcut)

    # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
    efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

    # Obtain Analytical signal
    efPhase = list()
    efEnvelope = list()
    for i in range(len(efSignals)):
        analyticalSignal = scipy.signal.hilbert(efSignals[i])
        # Get instantaneous phase and amplitude envelope by channel
        efPhase.append(np.angle(analyticalSignal))
        efEnvelope.append(np.abs(analyticalSignal))


    ### PLE Phase Lag Entropy
    # parameters - Phase Lag Entropy
    tau_ = 25  # ms
    m_ = 3  # pattern size

    ple, patterns = PLE(efPhase[:10], tau_, m_, samplingFreq, subsampling=20)
    results_ple.append(ple)
    patts.append(patterns)



####### Sliding Window Approach
# # dPLV
# def dFCm(data, samplingFreq, window, step, measure="PLV"):
#     window = window * 1000
#     step = step * 1000
#
#     if len(data[0]) > window:
#
#         (lowcut, highcut) = (8, 12)
#         # Band-pass filtering
#         filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut)
#
#         dyn_fc = list()
#         for w in np.arange(0, (len(data[0])) - window, step, 'int'):
#
#             print('%s %i / %i' % (measure, w / step, ((len(data[0])) - window) / step))
#
#             signals = filterSignals[:, w:w + window]
#
#             # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
#             if window>=4000:
#                 efSignals = epochingTool(signals, 4, samplingFreq, "signals")
#             else:
#                 efSignals = epochingTool(signals, window//1000, samplingFreq, "signals")
#
#             # Obtain Analytical signal
#             efPhase = list()
#             efEnvelope = list()
#             for i in range(len(efSignals)):
#                 analyticalSignal = scipy.signal.hilbert(efSignals[i])
#                 # Get instantaneous phase and amplitude envelope by channel
#                 efPhase.append(np.unwrap(np.angle(analyticalSignal)))
#                 efEnvelope.append(np.abs(analyticalSignal))
#
#             # Check point
#             # from toolbox import timeseriesPlot, plotConversions
#             # regionLabels = conn.region_labels
#             # # timeseriesPlot(emp_signals, raw_time, regionLabels)
#             # plotConversions(emp_signals[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], band="alpha")
#
#             if measure=="PLV":
#                 dyn_fc.append(PLV(efPhase))
#
#             elif measure=="AEC":
#                 dyn_fc.append(AEC(efEnvelope))
#
#         return dyn_fc
#
#     else:
#         print('Error: Signal length should be above window length (%i sec)' % window)
#
# window, step = 1, 0.05# seconds
# plv_matrices = dFCm(emp_signals, samplingFreq, window, step, "PLV")
# # aec_matrices = dFCm(emp_signals, samplingFreq, window, step, "AEC")
#
#
# # corr dPLV
# def dFCplot(matrices, time, step, folder='figures', plot="ON", auto_open=False):
#     r_matrix = np.zeros((len(matrices), len(matrices)))
#     for t1 in range(len(matrices)):
#         for t2 in range(len(matrices)):
#             r_matrix[t1, t2] = np.corrcoef(matrices[t1][np.triu_indices(len(matrices[0]), 1)],
#                                            matrices[t2][np.triu_indices(len(matrices[0]), 1)])[1, 0]
#
#     if plot == "ON":
#         fig = go.Figure(data=go.Heatmap(z=r_matrix, x=np.arange(0, time, step), y=np.arange(0, time, step),
#                                         colorscale='Viridis'))
#         fig.update_layout(title='Functional Connectivity Dynamics')
#         fig.update_xaxes(title="Time 1 (seconds)")
#         fig.update_yaxes(title="Time 2 (seconds)")
#         pio.write_html(fig, file=folder + "/PLV_" ".html", auto_open=auto_open)
#
# dFCplot(plv_matrices, len(emp_signals[0]), step, auto_open=True)
# # dFCplot(aec_matrices, len(emp_signals[0]), step, auto_open=True)

# clustering



#
# print("ROUND = %i " % r)
# print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))
#
# ## GATHER RESULTS
# simname = subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
# # Working on FFT peak results
# df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed", "round", "mS_peak", "mS_module", "allSignals"])
#
# df1.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)
#
# dfFFT_m = df1.groupby(["G", "speed"])[["G", "speed", "mS_peak"]].mean()
# # Load previously gathered data
# # df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")
# # df1a=df1[df1.G<33]
# paramSpace(df1, title=simname, folder=specific_folder)
#
# # Working on FC results
# df = pd.DataFrame(results_fc, columns=["G", "speed", "round", "plvD_r",  "plvT_r", "plvA_r", "plvB_r", "plvG_r"])
#                             # columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
#                                   #"pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])
#
# df.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)
#
#
# dfPLV_m = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
# dfPLV_sd = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].std().reset_index()
#
# dfPLV_m.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# dfPLV_sd.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
#
# # dfPLI = df[["G", "speed", "pliD_r", "pliT_r", "pliA_r", "pliB_r", "pliG_r"]]
# # dfPLI.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# # dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
# # dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
#
#
# paramSpace(dfPLV_m, 0.5, emp_subj+subjectid+"_PLV", folder=specific_folder)
# # paramSpace(dfPLV_sd, 0.5, emp_subj+subjectid+"_PLV_sd", folder=specific_folder)
#
# # paramSpace(dfPLI, 0.5, subjectid+"_PLI", folder=specific_folder)
# # paramSpace(dfAEC, 0.5, subjectid+"_AEC", folder=specific_folder)
#
# # df.to_csv("loop0-2m.csv", index=False)
# # b=df.sort_values(by=["noise", "G"])
# # fil=df[["G", "noise", "plv_avg","aec_avg","Tavg"]]
# # fil=fil.sort_values(by=["noise", "G"])
