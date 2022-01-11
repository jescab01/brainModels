import os
import time
import subprocess
from collections import Counter

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLE, epochingTool, paramSpace


# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".1995JansenRit"
emp_subj = "AVG_NEMOS"


wd=os.getcwd()
main_folder = wd+"\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"

if subjectid not in os.listdir(ctb_folder):
    os.mkdir(ctb_folder+subjectid)

specific_folder = main_folder+"\\""PSE"+subjectid+"-"+emp_subj+"-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)



# Prepare simulation parameters
simLength = 10*1000 # ms
samplingFreq = 1000 #Hz
transient = 1000 #ms
n_rep=1


# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)


conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")

mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 120, 2)
speed_vals = np.arange(0.5, 25, 2)
# coupling_vals = np.arange(0, 2, 1)
# speed_vals = np.arange(0.5, 1, 1)

results_fft_peak=list()
results_fc=list()
results_ple=list()

for g in coupling_vals:
    for s in speed_vals:
        for r in range(n_rep):
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

            # average signals to obtain mean signal frequency peak
            data = np.asarray([np.average(raw_data, axis=0)])
            data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

            # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
            results_fft_peak.append((g, s, r, FFTpeaks(data, simLength-transient)[0][0][0],
                                     FFTpeaks(data, simLength-transient)[1][0][0],
                                     FFTpeaks(data, simLength-transient)))


            newRow_fc = [g,s,r]
            newRow_ple = [g,s,r]
            bands = [["3-alpha"], [(8, 12)]]
            #            bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

            for b in range(len(bands[0])):
                (lowcut, highcut) = bands[1][b]

                # Band-pass filtering
                filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

                # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
                efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

                # Obtain Analytical signal
                efPhase=list()
                efEnvelope = list()
                for i in range(len(efSignals)):
                    analyticalSignal = scipy.signal.hilbert(efSignals[i])
                    # Get instantaneous phase and amplitude envelope by channel
                    efPhase.append(np.angle(analyticalSignal))
                    efEnvelope.append(np.abs(analyticalSignal))


                # Check point
                # from toolbox import timeseriesPlot, plotConversions
                # regionLabels = conn.region_labels
                # timeseriesPlot(efPhase[0], np.arange(len(efPhase[0][0])), regionLabels)
                # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)


                # CONNECTIVITY MEASURES
                ## PLV
                plv = PLV(efPhase)
                # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
                # np.savetxt(fname, plv)

                ## PLE - Phase Lag Entropy
                ## PLE parameters - Phase Lag Entropy
                tau_ = 25  # ms
                m_ = 3  # pattern size
                # IF it takes too long use subsampling: choose one point out of each "subsampling"=5
                ple, _ = PLE(efPhase, tau_, m_, samplingFreq, subsampling=85)
                newRow_ple.append(np.average(ple[np.triu_indices(len(ple[0]), 1)]))



                # ## PLI
                # pli = PLI(efPhase)
                # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"pli"
                # np.savetxt(fname, pli)

                ## AEC
                # aec = AEC(efEnvelope)
                # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
                # np.savetxt(fname, aec)

                # Load empirical data to make simple comparisons
                plv_emp=np.loadtxt(ctb_folder+"FC_"+emp_subj+"\\"+bands[0][b]+"_plv.txt")
                #pli_emp=np.loadtxt(wd+"\\CTB_data\\output\\FC_"+emp_subj+"\\"+bands[0][b]+"pli.txt")
                # aec_emp=np.loadtxt(wd+"\\CTB_data\\output\\FC_"+emp_subj+"\\"+bands[0][b]+"_aec.txt")

                # Comparisons
                t1 = np.zeros(shape=(2, len(conn.region_labels)**2//2-len(conn.region_labels)//2))
                t1[0, :] = plv[np.triu_indices(len(conn.region_labels), 1)]
                t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels), 1)]
                plv_r = np.corrcoef(t1)[0, 1]
                newRow_fc.append(plv_r)
                # t2 = np.zeros(shape=(2, 2145))
                # t2[0, :] = pli[np.triu_indices(66, 1)]
                # t2[1, :] = pli_emp[np.triu_indices(66, 1)]
                # pli_r = np.corrcoef(t2)[0, 1]
                # newRow.append(pli_r)
                #
                # t3 = np.zeros(shape=(2, 2145))
                # t3[0, :] = aec[np.triu_indices(66, 1)]
                # t3[1, :] = aec_emp[np.triu_indices(66, 1)]
                # aec_r = np.corrcoef(t3)[0, 1]
                # newRow.append(aec_r)

            results_fc.append(newRow_fc)
            results_ple.append(newRow_ple)

            print("ROUND = %i " % r)
            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))

## GATHER RESULTS
simname = subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")

# Working on FFT peak results
df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed", "round", "mS_peak", "mS_module", "allSignals"])

df1.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)

dfFFT_m = df1.groupby(["G", "speed"])[["G", "speed", "mS_peak"]].mean()
paramSpace(df1, title=simname, folder=specific_folder)



# Working on FC results
df = pd.DataFrame(results_fc, columns=["G", "speed", "round", "plvD_r",  "plvT_r", "plvA_r", "plvB_r", "plvG_r"])
                            # columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
                                  #"pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])

df.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)


dfPLV_m = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
dfPLV_sd = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].std().reset_index()

dfPLV_m.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
dfPLV_sd.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]

# dfPLI = df[["G", "speed", "pliD_r", "pliT_r", "pliA_r", "pliB_r", "pliG_r"]]
# dfPLI.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
# dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]


paramSpace(dfPLV_m, 0.5, emp_subj+subjectid+"_PLV", folder=specific_folder)
# paramSpace(dfPLV_sd, 0.5, emp_subj+subjectid+"_PLV_sd", folder=specific_folder)

# paramSpace(dfPLI, 0.5, subjectid+"_PLI", folder=specific_folder)
# paramSpace(dfAEC, 0.5, subjectid+"_AEC", folder=specific_folder)

# df.to_csv("loop0-2m.csv", index=False)
# b=df.sort_values(by=["noise", "G"])
# fil=df[["G", "noise", "plv_avg","aec_avg","Tavg"]]
# fil=fil.sort_values(by=["noise", "G"])


# Working on PLE results
df2 = pd.DataFrame(results_fc, columns=["G", "speed", "round", "pleD",  "pleT", "pleA", "pleB", "pleG"])
                            # columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
                                  #"pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])

df2.to_csv(specific_folder+"/PSE_PLE"+simname+".csv", index=False)

# dfPLE_m = df2.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
# dfPLE_sd = df2.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].std().reset_index()

df2.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# dfPLE_sd.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]


paramSpace(df2, 0.5, emp_subj+subjectid+"_PLE", folder=specific_folder)

