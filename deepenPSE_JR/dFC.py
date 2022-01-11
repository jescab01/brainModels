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
simLength = 2*60*1000 # ms
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

# coupling_vals = np.arange(0, 120, 1)
# speed_vals = np.arange(0.5, 25, 1)
# coupling_vals = np.arange(0, 1, 1)
# speed_vals = np.arange(0.5, 1, 1)
g=63
s=15
r=1

results_fc=list()

coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))
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

print("%0.2f min simulation took %0.3f seconds.\n\n\n\n" % (simLength/60000, time.time() - tic))


## dPLV
def dPLV(raw_data, samplingFreq, window, step):
    window = window * 1000
    step = step * 1000

    if len(raw_data[0]) > window:

        (lowcut, highcut) = (8, 12)
        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        dyn_plv = list()
        for w in np.arange(0, (len(raw_data[0])) - window, step):

            print('PLV %i / %i' % (w/step, ((len(raw_data[0])) - window) / step))

            signals = filterSignals[:, w:w + window]

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(signals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            dyn_plv.append(PLV(efPhase))

        return dyn_plv

    else:
        print('Error: Signal length should be above window length (%i sec)' % (4 * window))


window, step = 30, 2  # seconds
plv_matrices = dPLV(raw_data, samplingFreq, window, step)

# corr dPLV
def FCD(plv_matrices, length, step, folder='figures', plot="ON", auto_open=False):

    r_matrix = np.zeros((len(plv_matrices), len(plv_matrices)))
    for t1 in range(len(plv_matrices)):
        for t2 in range(len(plv_matrices)):
            r_matrix[t1, t2] = np.corrcoef(plv_matrices[t1][np.triu_indices(len(plv_matrices[0]), 1)], plv_matrices[t2][np.triu_indices(len(plv_matrices[0]), 1)])[1,0]

    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=r_matrix, x=np.arange(0,length, step ), y=np.arange(0,length, step ), colorscale='Viridis'))
        fig.update_layout(title='Functional Connectivity Dynamics')
        pio.write_html(fig, file=folder + "/PLV_" ".html", auto_open=auto_open)


FCD(plv_matrices, simLength-transient, step, auto_open=True)


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