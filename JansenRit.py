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

conn = connectivity.Connectivity.from_file(ctb_folder+"CTB_connx66_"+emp_subj+".zip")
conn.weights = conn.scaled_weights(mode="tract")

# Coupling function
coup = coupling.SigmoidalJansenRit(a=np.array([28]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))
conn.speed=np.array([6])

mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
sim.configure()

output = sim.run(simulation_length=simLength)
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


# # Test bimodality in frequency amplitudes; We will use an external R script to do it
# pvalsBimodality = list()
# for i, signal in enumerate(data):
#     print("Bimodality test for signal %i/%i; regions mode" % (i + 1, len(data)), end="\r")
#     # morlet wavelet convolution
#     power = time_frequency.tfr_array_morlet([[signal]], 1000, [fft_peaks[i]], output="avg_power")  # avg_power=trials_avg
#     # Save data so R script can access it
#     np.savetxt('C:\\Users\\F_r_e\\PycharmProjects\\brainModels\\diptest\\powers.csv',
#                power[0][0][2500:-2500])  # remove starting and ending pieces of convolution due to distortion
#     # subprocess will call an external process: Rscript
#     # .check_output will run the command in R and store to result
#     p_value = subprocess.check_output(cmd)  ## p-value==0 means p-value<2.2e-16
#     pvalsBimodality.append(np.float(p_value))
# print("Bimodality test for signal %i/%i; regions mode" % (i + 1, len(data)))


fc_result = []
bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
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

    #CONNECTIVITY MEASURES
    ## PLV
    plv = PLV(efPhase)
    fname = ctb_folder+subjectid+"\\"+bands[0][b]+"plv.txt"
    np.savetxt(fname, plv)

    ## dPLV
    # dPLV= dPLV(efPhase)

    ## AEC
    aec = AEC(efEnvelope)
    fname = ctb_folder+subjectid+"\\"+bands[0][b]+"corramp.txt"
    np.savetxt(fname, aec)

    ## PLI
    pli = PLI(efPhase)
    fname = ctb_folder+subjectid+"\\"+bands[0][b]+"pli.txt"
    np.savetxt(fname, pli)


    # Load empirical data to make simple comparisons
    plv_emp = np.loadtxt(ctb_folder+"FC_" + emp_subj + "\\" + bands[0][b] + "plv.txt")
    aec_emp = np.loadtxt(ctb_folder+"FC_" + emp_subj + "\\" + bands[0][b] + "corramp.txt")

    pli_emp = np.loadtxt(ctb_folder+"FC_" + emp_subj + "\\" + bands[0][b] + "pli.txt")

    # Comparisons
    t1 = np.zeros(shape=(2, 2145))
    t1[0, :] = plv[np.triu_indices(66, 1)]
    t1[1, :] = plv_emp[np.triu_indices(66, 1)]
    plv_r = np.corrcoef(t1)[0, 1]
    fc_result.append(plv_r)

    t2 = np.zeros(shape=(2, 2145))
    t2[0, :] = pli[np.triu_indices(66, 1)]
    t2[1, :] = pli_emp[np.triu_indices(66, 1)]
    pli_r = np.corrcoef(t2)[0, 1]
    fc_result.append(pli_r)

    t3 = np.zeros(shape=(2, 2145))
    t3[0, :] = aec[np.triu_indices(66, 1)]
    t3[1, :] = aec_emp[np.triu_indices(66, 1)]
    aec_r = np.corrcoef(t3)[0, 1]
    fc_result.append(aec_r)

fc_result = pd.DataFrame([fc_result], columns=["plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r",
                                               "plvA_r", "pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r",
                                               "pliG_r", "aecG_r"])


# Copy structural connetivity weights in FC folder
weights = conn.weights
fname = ctb_folder+subjectid+"\\weights.txt"
np.savetxt(fname, weights)

del i, highcut, lowcut, t1, t3, filterSignals, efPhase,
del efEnvelope, efSignals, aec, aec_r, aec_emp, plv, plv_r, plv_emp


# Empirical vs Simulated analysis based on CTB folder data
# Time reference
tic = time.time()

# Getting number of subjects (could vary due to simulated new subjects)
files=set(os.listdir(ctb_folder))
rm=set(['CTB_connx66_subj02.zip', 'CTB_connx66_subj03.zip', 'CTB_connx66_subj04.zip', 'CTB_connx66_subj05.zip',
 'CTB_connx66_subj06.zip', 'CTB_connx66_subj07.zip', 'CTB_connx66_subj08.zip', 'CTB_connx66_subj09.zip',
 'CTB_connx66_subj10.zip'])
subjects=sorted(list(files.difference(rm)))
n=len(subjects)

# Load bands
bands = [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

# Checking correlations between subjects
# This is resting state FC, measures between subject should correlate to some extent
FC_corr=pd.DataFrame(columns=["FC_measure", "band", "frm","to","r"])
tic=time.time()
print("Calculating FC correlations BETWEEN subjects", end="")
for i1, s1 in enumerate(subjects):
    print("Calculating FC correlations BETWEEN subjects - subject %i/%i" % (i1 + 1, len(subjects)), end="\r")
    for i2, s2 in enumerate(subjects):
      s1dir=ctb_folder+s1+"\\"
      s2dir=ctb_folder+s2+"\\"

      for b in range(len(bands[0])):
         s1bdir = s1dir + bands[0][b]
         AEC1 = np.loadtxt(s1bdir + "corramp.txt")
         PLV1 = np.loadtxt(s1bdir + "plv.txt")

         s2bdir = s2dir + bands[0][b]
         AEC2 = np.loadtxt(s2bdir + "corramp.txt")
         PLV2 = np.loadtxt(s2bdir + "plv.txt")

         t1 = np.zeros(shape=(2, 2145))
         t1[0, :] = AEC1[np.triu_indices(66, 1)]
         t1[1, :] = AEC2[np.triu_indices(66, 1)]
         newRow_aec=pd.Series({"FC_measure":"aec", "band":bands[0][b], "frm":s1, "to":s2, "r":np.corrcoef(t1)[0, 1]})

         t2 = np.zeros(shape=(2, 2145))
         t2[0, :] = PLV1[np.triu_indices(66, 1)]
         t2[1, :] = PLV2[np.triu_indices(66, 1)]
         newRow_plv=pd.Series({"FC_measure":"plv", "band":bands[0][b], "frm":s1, "to":s2, "r":np.corrcoef(t2)[0, 1]})

         FC_corr=FC_corr.append(newRow_aec, ignore_index=True)
         FC_corr=FC_corr.append(newRow_plv, ignore_index=True)

FC_corr.to_csv(main_folder+"/FC_corrs.csv")
paramSpace(FC_corr,title="FC_comparisons", folder=main_folder)
print("Calculating FC correlations BETWEEN subjects - subject %i/%i - %0.3f seconds.\n"
      % (i1 + 1, len(subjects), time.time() - tic,))
del AEC1, AEC2, PLV1, PLV2, s1dir, s1, s2dir, s2, t1, t2, b, i1, i2


# Checking structural - functional correlations within subjects
sf_corr=pd.DataFrame(columns=["FC_measure", "band", "subj","r"])
tic=time.time()
print("Calculating structural - functional correlations WITHIN subjects (PLV and AEC, independently)", end="")
for i, name in enumerate(subjects):
    print("Calculating structural - functional correlations WITHIN subjects - subject %i/%i"
          % (i + 1, len(subjects)), end="\r")
    sdir=ctb_folder+name+"\\"
    for b in range(len(bands[0])):
      AEC = np.loadtxt(sdir+bands[0][b]+"corramp.txt")
      PLV = np.loadtxt(sdir+bands[0][b]+"plv.txt")

      weights=np.loadtxt(sdir+"weights.txt")

      c1=np.zeros(shape=(2, 2145))
      c1[0,:]=AEC[np.triu_indices(66, 1)]
      c1[1,:]=weights[np.triu_indices(66, 1)]

      c2=np.zeros(shape=(2, 2145))
      c2[0,:]=PLV[np.triu_indices(66, 1)]
      c2[1,:]=weights[np.triu_indices(66, 1)]

      newRow_aec = pd.Series({"FC_measure":"aec", "band":bands[0][b], "subj": name, "r":np.corrcoef(c1)[0,1]})
      newRow_plv = pd.Series({"FC_measure":"plv", "band":bands[0][b], "subj": name, "r":np.corrcoef(c1)[0,1]})
      sf_corr = sf_corr.append(newRow_aec, ignore_index=True)
      sf_corr = sf_corr.append(newRow_plv, ignore_index=True)

sf_corr.to_csv(main_folder+"/sf_corrs.csv")
paramSpace(sf_corr, title="sf_comparisons", folder=main_folder)
print("Calculating structural - functional correlations WITHIN subjects - subject %i/%i - %0.3f seconds.\n"
      % (i + 1, len(subjects), time.time() - tic,))

del sdir, AEC, PLV, weights, c1, c2

print("Mean signal frequency peak: %0.2f \n"
      "Total time spent: %0.2f min"
      % (float(fft_peaks[0]), (time.time() - tic0)/60,))

