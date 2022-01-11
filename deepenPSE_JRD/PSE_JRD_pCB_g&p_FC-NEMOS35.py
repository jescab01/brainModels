import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N


# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".2003JansenRitDavid_N"
emp_subj = "NEMOS_035"
test = "p_CB"

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

main_folder = os.getcwd() + "\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

specific_folder = main_folder+"\\""PSE"+test+subjectid+"-"+emp_subj+"-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

# Prepare simulation parameters
simLength = 10 * 1000 # ms
samplingFreq = 1000 #Hz
transient = 2000 #ms
n_rep = 3

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)


conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")

CB_rois = [18, 19, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 62, 63, 64, 65, 70, 71, 76, 77]
conn.weights = conn.weights[:, CB_rois][CB_rois]
conn.tract_lengths = conn.tract_lengths[:, CB_rois][CB_rois]
conn.region_labels = conn.region_labels[CB_rois]

mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 300, 4)
input_vals = np.arange(0, 0.22, 0.01)

results_amp = list()
results_fc = list()

for g in coupling_vals:
    for p in input_vals:
        for th in ["p", "p+0.22", "0.22"]:
            # Construct p array
            if th == "p":
                p_array = np.array([p]*len(conn.region_labels))
            elif th == "p+0.22":
                p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), p + 0.22, p)
            elif th == "0.22":
                p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.22, p)

            sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.022, 0)

            # Parameters from Stefanovski 2019.
            m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                     tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                     He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                     tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
                                     w=np.array([0.8]), c=np.array([135.0]),

                                     c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                     c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                     v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                     p=np.array([p_array]), sigma=np.array([sigma_array]))

            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
            conn.speed = np.array([15])

            tic = time.time()
            print("Simulation for g = %i, input = %0.2f and th = %s " % (g, p, th))

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)

            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            raw_data = output[0][1][transient:, 0, :, 0].T
            # raw_time = output[0][0][transient:]

            mean = np.average(np.average(raw_data, axis=1))
            sd = np.average(np.std(raw_data, axis=1))

            # timeseriesPlot(raw_data, raw_time, conn.region_labels)

            # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
            results_amp.append((g, p, th, mean, sd))

            newRow = [g, p, th]
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

                # Load empirical data to make simple comparisons
                plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_plv.txt")[:, CB_rois][CB_rois]

                # Comparisons
                t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
                t1[0, :] = plv[np.triu_indices(len(conn.region_labels), 1)]
                t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels), 1)]
                plv_r = np.corrcoef(t1)[0, 1]
                newRow.append(plv_r)

            results_fc.append(newRow)
            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))



## GATHER RESULTS
simname = test+subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
# Working on FFT peak results
df_amp = pd.DataFrame(results_amp, columns=["g", "p", "th", "amp_mean", "amp_sd"])
df_amp.to_csv(specific_folder+"/PSE_"+simname+".csv", index=False)

# Working on FC results
# df = pd.DataFrame(results_fc, columns=["g", "sigma", "w", "round", "Delta", "Theta", "Alpha", "Beta", "Gamma"])
df_fc = pd.DataFrame(results_fc, columns=["g", "p", "th", "Alpha"])
df_fc.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)

# df_m = df.groupby(["g", "p", "th"])[["g", "p", "th", "Alpha"]].mean()
# df_sd = df.groupby(["g", "p", "th"])[["g", "p", "th", "Alpha"]].std().reset_index()


fig = make_subplots(rows=1, cols=6, subplot_titles=("Signal std - th = p", "rFC - th = p",
                                                    "Signal std - th = p + 0.22", "rFC - th = p + 0.22",
                                                    "Signal std - th = 0.22", "rFC - th = 0.22"),
                    specs=[[{}, {}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                    x_title="Input (p)", y_title="Coupling factor")

df_subset = df_amp.loc[df_amp["th"] == "p"]
fig.add_trace(go.Heatmap(z=df_subset.amp_sd, x=df_subset.p, y=df_subset.g, colorscale='Viridis'), row=1, col=1)
df_subset = df_fc.loc[df_fc["th"] == "p"]
fig.add_trace(go.Heatmap(z=df_subset.Alpha, x=df_subset.p, y=df_subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5, showscale=False), row=1, col=2) # colorbar=dict(title="mV", thickness=10, x=1.02/4.3*2)

df_subset = df_amp.loc[df_amp["th"] == "p+0.22"]
fig.add_trace(go.Heatmap(z=df_subset.amp_sd, x=df_subset.p, y=df_subset.g, colorscale='Viridis'), row=1, col=3)
df_subset = df_fc.loc[df_fc["th"] == "p+0.22"]
fig.add_trace(go.Heatmap(z=df_subset.Alpha, x=df_subset.p, y=df_subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5, showscale=False), row=1, col=4) # colorbar=dict(title="mV", thickness=10, x=1.02/4.3*2)

df_subset = df_amp.loc[df_amp["th"] == "0.22"]
fig.add_trace(go.Heatmap(z=df_subset.amp_sd, x=df_subset.p, y=df_subset.g, colorscale='Viridis'), row=1, col=5)
df_subset = df_fc.loc[df_fc["th"] == "0.22"]
fig.add_trace(go.Heatmap(z=df_subset.Alpha, x=df_subset.p, y=df_subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5), row=1, col=6) # colorbar=dict(title="mV", thickness=10, x=1.02/4.3*2)


fig.update_layout(
    title_text='rFC and signals amplitude by coupling factor (g) and intrinsic input (p)|| %s' % emp_subj)
pio.write_html(fig, file=specific_folder + "/paramSpace-%s.html" % test, auto_open=True)