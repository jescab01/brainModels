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

# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".2003LarterBreakspear"
emp_subj = "subj04"


wd=os.getcwd()
main_folder = wd+"\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
ctb_folder = wd + "\\CTB_data\\output\\"
if subjectid not in os.listdir(ctb_folder):
    os.mkdir(ctb_folder+subjectid)

specific_folder = main_folder+"\\""PSE"+subjectid+"-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)



# Prepare simulation parameters
simLength = 5*1000 # ms
samplingFreq = 1024 #Hz
transient=1000 #ms

# Parameters from Roberts (2019)
m = models.LarterBreakspear(C=np.array([0.6]), Iext=np.array([0]),
                            QV_max=np.array([1]), QZ_max=np.array([1]),
                            TCa=np.array([-0.01]), TK=np.array([0]), TNa=np.array([0.3]),
                            VCa=np.array([1]), VK=np.array([-0.7]), VL=np.array([-0.5]), VNa=np.array([0.53]),
                            VT=np.array([0]), ZT=np.array([0]),
                            aee=np.array([0.36]), aei=np.array([2]), aie=np.array([2]), ane=np.array([1]),
                            ani=np.array([0.4]),
                            b=np.array([0.1]),
                            d_Ca=np.array([0.15]), d_K=np.array([0.3]), d_Na=np.array([0.15]), d_V=np.array([0.65]),
                            d_Z=np.array([0.65]),
                            gCa=np.array([1]), gK=np.array([2]), gL=np.array([0.5]), gNa=np.array([6.7]),
                            phi=np.array([0.7]), rNMDA=np.array([0.25]), t_scale=np.array([0.1]), tau_K=np.array([1]))



# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)


conn = connectivity.Connectivity.from_file(ctb_folder+"CTB_connx66_"+emp_subj+".zip")
conn.weights = conn.scaled_weights(mode="tract")

mon = (monitors.Raw(),)

coupling_vals = [15]#np.arange(0.1, 15, 0.1)
speed_vals = [3]#np.arange(0.5, 25, 1)

results_fft_peak=list()
results_fc=list()

for g in coupling_vals:
    for s in speed_vals:

        coup = coupling.HyperbolicTangent(a=np.array([g]), b=np.array([1]), midpoint=np.array([0]), sigma=np.array([1]))
        conn.speed = np.array([s])
        tic = time.time()
        print("Simulating for Coupling factor = %0.2f and speed = %0.2f" % (g, s))

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
        results_fft_peak.append((g, s, FFTpeaks(data, simLength)[0][0][0],
                                 FFTpeaks(data, simLength)[1][0][0],
                                 FFTpeaks(data, simLength)))


        newRow = [g,s]
        bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
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
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], regionLabels, 1, raw_time)


            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)
            # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
            # np.savetxt(fname, plv)

            # ## PLI
            pli = PLI(efPhase)
            # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"pli"
            # np.savetxt(fname, pli)

            ## AEC
            aec = AEC(efEnvelope)
            # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
            # np.savetxt(fname, aec)

            # Load empirical data to make simple comparisons
            plv_emp=np.loadtxt(wd+"\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"plv.txt")
            pli_emp=np.loadtxt(wd+"\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"pli.txt")
            aec_emp=np.loadtxt(wd+"\\CTB_data\\output\\FC_subj04\\"+bands[0][b]+"corramp.txt")

            # Comparisons
            t1 = np.zeros(shape=(2, 2145))
            t1[0, :] = plv[np.triu_indices(66, 1)]
            t1[1, :] = plv_emp[np.triu_indices(66, 1)]
            plv_r = np.corrcoef(t1)[0, 1]
            newRow.append(plv_r)

            t2 = np.zeros(shape=(2, 2145))
            t2[0, :] = pli[np.triu_indices(66, 1)]
            t2[1, :] = pli_emp[np.triu_indices(66, 1)]
            pli_r = np.corrcoef(t2)[0, 1]
            newRow.append(pli_r)

            t3 = np.zeros(shape=(2, 2145))
            t3[0, :] = aec[np.triu_indices(66, 1)]
            t3[1, :] = aec_emp[np.triu_indices(66, 1)]
            aec_r = np.corrcoef(t3)[0, 1]
            newRow.append(aec_r)

        results_fc.append(newRow)

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))

## GATHER RESULTS
simname = subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
# Working on FFT peak results
df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed", "mS_peak", "mS_module", "allSignals"])

df1.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)

# Load previously gathered data
# df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")
# df1a=df1[df1.G<33]
paramSpace(df1, title=simname, folder=specific_folder)

# Working on FC results
df=pd.DataFrame(results_fc, columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
                                  "pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])

df.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)


dfPLV = df[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]]
dfPLV.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
dfPLI = df[["G", "speed", "pliD_r", "pliT_r", "pliA_r", "pliB_r", "pliG_r"]]
dfPLI.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]

del g, s, i, b, aec, aec_emp, aec_r, plv, plv_emp, plv_r, pli, pli_emp, pli_r
del analyticalSignal, bands, efPhase, efSignals, efEnvelope, filterSignals, highcut, lowcut
del newRow, t1, t2, t3, tic
del sim, coup, conn, models, samplingFreq, simLength, mon, output, raw_time, raw_data


paramSpace(dfPLI, 0.5, subjectid+"_PLI", folder=specific_folder)
paramSpace(dfAEC, 0.5, subjectid+"_AEC", folder=specific_folder)
paramSpace(dfPLV, 0.5, subjectid+"_PLV", folder=specific_folder)

# df.to_csv("loop0-2m.csv", index=False)
# b=df.sort_values(by=["noise", "G"])
# fil=df[["G", "noise", "plv_avg","aec_avg","Tavg"]]
# fil=fil.sort_values(by=["noise", "G"])

# for i in range(len(plv)):
#     plv[i][i]=np.average(plv)