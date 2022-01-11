import os
import numpy as np
import scipy.signal
from mne.connectivity import spectral_connectivity
from mne import filter
from toolbox import PLV, epochingTool


emp_subj = "NEMOS_035"
samplingFreq = 1000  # Hz

wd = os.getcwd()
ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"

emp_signals = np.loadtxt(ctb_folder + 'FC_' + emp_subj + '/.RawSignals.txt', delimiter=",")  # Lasts half a minute


#####  Validate PLV matrics in front of empirical PLV from Brainstorm

# BRAINSTORM plv loaded
plv_bst = np.loadtxt(wd + "\\CTB_data\\output\\FC_" + emp_subj + "\\3-alpha_plv.txt")


# MINE plv
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

plv = PLV(efPhase)


# MNE plv
epoched_emp_signals = epochingTool(emp_signals, 4, samplingFreq, "signals")
plv_mneRaw=spectral_connectivity(epoched_emp_signals, method='plv', sfreq=samplingFreq, fmin=8, fmax=12, faverage=True)
plv_mne=np.squeeze(plv_mneRaw[0])


# Calculate correlations
t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
t1[0, :] = plv[np.tril_indices(len(plv), -1)]
t1[1, :] = plv_bst[np.tril_indices(len(plv), -1)]
rMine_Brainstorm = np.corrcoef(t1)[0, 1]

t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
t1[0, :] = plv[np.tril_indices(len(plv), -1)]
t1[1, :] = plv_mne[np.tril_indices(len(plv), -1)]
rMine_MNE = np.corrcoef(t1)[0, 1]

t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
t1[0, :] = plv_bst[np.tril_indices(len(plv), -1)]
t1[1, :] = plv_mne[np.tril_indices(len(plv), -1)]
rBrainstorm_MNE = np.corrcoef(t1)[0, 1]




