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

from toolbox import FFTpeaks,  PLV, AEC, epochingTool, paramSpace
from dynamics import dynamic_fc, kuramoto_order


wd = os.getcwd()
ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"

# Sliding window parameters
window, step = 4, 2  # seconds
samplingFreq = 1000  # Hz

# Subset of AAL2red rois to analyze: omitting subcorticals
fc_rois_cortex = list(range(0, 40)) + list(range(46, 74)) + list(range(78, 90))
bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

for nemos_id in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]:

    tic = time.time()

    emp_subj = "NEMOS_0" + str(nemos_id)
    emp_signals = np.loadtxt(ctb_folder + 'FC_' + emp_subj + '/.RawSignals.txt', delimiter=",")  # Lasts 30 secs by now; Whole signal 3 minutes

    print("Working on " + emp_subj)

    for band in range(len(bands[0])):

        # Compute PLVs in time
        dynamicFC_matrix = dynamic_fc(emp_signals, samplingFreq, window, step, "PLV", lowcut=bands[1][band], highcut=bands[1][band])
        np.savetxt(ctb_folder + 'FC_' + emp_subj + '/'+bands[0][band]+'_dPLV'+str(window)+'s.txt', dynamicFC_matrix, delimiter=" ")

        sdKO, meanKO = kuramoto_order(emp_signals, samplingFreq)
        np.savetxt(ctb_folder + 'FC_' + emp_subj + '/'+bands[0][band]+'_sdKO.txt', [sdKO])
        np.savetxt(ctb_folder + 'FC_' + emp_subj + '/' + bands[0][band] + '_meanKO.txt', [meanKO])
    print("SUBJECT ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))