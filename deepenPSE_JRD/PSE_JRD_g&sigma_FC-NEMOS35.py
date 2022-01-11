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
mode = "cb"
w = 0.8
test = "fc_w" + str(w) + mode

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

main_folder = os.getcwd() + "\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

specific_folder = main_folder+"\\PSE"+test+subjectid+"-"+emp_subj+"-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 2000  # ms
n_rep = 3

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")
if "cb" in mode:
    CB_rois = [18, 19, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 62, 63, 64, 65, 70, 71, 76, 77]
    conn.weights = conn.weights[:, CB_rois][CB_rois]
    conn.tract_lengths = conn.tract_lengths[:, CB_rois][CB_rois]
    conn.region_labels = conn.region_labels[CB_rois]

mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 300, 5)
sigma_vals = np.geomspace(0.0022, 2.2, 10)

new_p = 0.22 - 0.22 * (len(CB_rois)/92)

p_vals = [0, 0.1085, new_p, new_p]

results_amp = list()
results_fc = list()

for g in coupling_vals:
    for sigma in sigma_vals:
        sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), sigma, 0)

        for j, p in enumerate(p_vals):
            if j == 3:
                p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), p + 0.22, p)
            else:
                p_array = np.array([p]*len(conn.region_labels))

            # Parameters from Stefanovski 2019.
            m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                     tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                     He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                     tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                     w=np.array([w]), c=np.array([135.0]),
                                     c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                     c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                     v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                     p=np.array([p_array]), sigma=np.array([sigma_array]))

            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
            conn.speed = np.array([15])

            for r in range(n_rep):

                tic = time.time()
                print("Simulation %i for Coupling factor = %i, p = %0.2f, sigma = %0.3f and w = %0.1f" % (r, g, p, sigma, w))

                # Run simulation
                sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
                sim.configure()
                output = sim.run(simulation_length=simLength)

                # Extract data: "output[a][b][:,0,:,0].T" where:
                # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
                raw_data = output[0][1][transient:, 0, :, 0].T
                raw_time = output[0][0][transient:]

                mean = np.average(np.average(raw_data, axis=1))
                sd = np.average(np.std(raw_data, axis=1))

                # timeseriesPlot(raw_data, raw_time, conn.region_labels)

                # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
                results_amp.append((g, p, sigma, w, mean, sd))


                newRow = [g, p, sigma, w, r]
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
                    if "cb" in mode:
                        plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_plv.txt")[:, CB_rois][CB_rois]
                    else:
                        plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_plv.txt")

                    # Comparisons
                    t1 = np.zeros(shape=(2, len(conn.region_labels)**2//2-len(conn.region_labels)//2))
                    t1[0, :] = plv[np.triu_indices(len(conn.region_labels), 1)]
                    t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels), 1)]
                    plv_r = np.corrcoef(t1)[0, 1]
                    newRow.append(plv_r)

                results_fc.append(newRow)
                print("ROUND = %i " % r)
    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))


## GATHER RESULTS
simname = test + subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")

# Working on signals amp results
df1 = pd.DataFrame(results_amp, columns=["g", "p", "sigma", "w", "amp_mean", "amp_sd"])
df1.to_csv(specific_folder+"/PSE_amp"+simname+".csv", index=False)

# Working on FC results
# df = pd.DataFrame(results_fc, columns=["g", "sigma", "w", "round", "Delta", "Theta", "Alpha", "Beta", "Gamma"])
df = pd.DataFrame(results_fc, columns=["g", "p", "sigma", "w", "round", "Alpha"])
df.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)

df_m = df.groupby(["g", "p", "sigma"])[["g", "p", "sigma", "Alpha"]].mean()
df_sd = df.groupby(["g", "p", "sigma"])[["g", "p", "sigma", "Alpha"]].std().reset_index()


fig = make_subplots(rows=4, cols=3, subplot_titles=("Signals std", "avg rFC (alpha)", "sd rFC (alpha)"),
                    specs=[[{}, {}, {}],[{}, {}, {}],[{}, {}, {}],[{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                    row_titles=["p = 0", "p = 0.1085", "p = 0.22", "p(th) = 0.22"],
                    x_title="Sigma (noise)", y_title="Coupling factor")

for i, p in enumerate(p_vals):

    df1_subset = df1.loc[df1["p"] == p]
    df_m_subset = df_m.loc[df_m["p"] == p]
    df_sd_subset = df_sd.loc[df_sd["p"] == p]

    fig.add_trace(go.Heatmap(z=df1_subset.amp_sd, x=df1_subset.sigma, y=df1_subset.g, colorscale='Viridis',
                             colorbar=dict(title="mV", x=1.02/3.5, thickness=10), zmin=0, zmax=2.5, showlegend=False), row=1+i, col=1)
    fig.add_trace(go.Heatmap(z=df_m_subset.Alpha, x=df_m_subset.sigma, y=df_m_subset.g, colorscale='RdBu', reversescale=True,
                             zmin=-0.5, zmax=0.5, colorbar=dict(title="avg rFC", x=1.02/3.15*2, thickness=10)), row=1+i, col=2)
    fig.add_trace(go.Heatmap(z=df_sd_subset.Alpha, x=df_sd_subset.sigma, y=df_sd_subset.g,
                             colorbar=dict(title="sd", thickness=10), zmin=0, zmax=0.2), row=1+i, col=3)
fig.update_xaxes(type="log")
fig.update_layout(
    title_text='Thalamic noise simulations || [w = %0.2f] %s' % (w, emp_subj))
pio.write_html(fig, file=specific_folder + "/paramSpace-%s.html" % test, auto_open=True)


