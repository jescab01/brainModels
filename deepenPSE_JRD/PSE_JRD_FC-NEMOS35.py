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

from dynamics import dynamic_fc, kuramoto_order
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N


# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".2003JansenRitDavid_N"
emp_subj = "NEMOS_035"
w = 1
test = "JRD"

params = {"basicJR": {"tau_e1": 10.8, "tau_i1": 22.0, "tau_e2": 4.6, "tau_i2": 2.9, "w": 1, "p": 0.1085, "sigma": 0},
          "JRD": {"tau_e1": 10.8, "tau_i1": 22.0, "tau_e2": 4.6, "tau_i2": 2.9, "w": 0.8, "p": 0.22, "sigma": 0.022}}


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

fc_rois_cortex = list(range(0, 40)) + list(range(46, 74)) + list(range(78, 90))
sc_rois_cortex = fc_rois_cortex  ## Here using AAL2red

mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 200, 4)
speed_vals = np.arange(0.5, 25, 5)

results_amp = list()
results_fc = list()

# Prepare current simulation
tau_e1, tau_i1, tau_e2, tau_i2, w, p, sigma = \
    params[test]["tau_e1"], params[test]["tau_i1"], params[test]["tau_e2"], \
    params[test]["tau_i2"], params[test]["w"], params[test]["p"], params[test]["sigma"]

if test == "JRD":
    sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), sigma, 0)
    p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), p, 0)
else:
    sigma_array = sigma
    p_array = p


# Simulate
for g in coupling_vals:
    for s in speed_vals:

        # Parameters from Stefanovski 2019.
        m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                 tau_e1=np.array([tau_e1]), tau_i1=np.array([tau_i1]),
                                 He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                 tau_e2=np.array([tau_e2]), tau_i2=np.array([tau_i2]),

                                 w=np.array([w]), c=np.array([135.0]),
                                 c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                 c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                 v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                 p=np.array([p_array]), sigma=np.array([sigma_array]))

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
        conn.speed = np.array([s])

        for r in range(n_rep):

            tic = time.time()
            print("Simulation %i for Coupling factor = %i, s = %0.2f, and w = %0.1f" % (r, g, s, w))

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


            newRow = [g, s, r]
            # bands = [["3-alpha"], [(8, 12)]]
            bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

            for b in range(len(bands[0])):
                (lowcut, highcut) = bands[1][b]

                # Band-pass filtering
                filterSignals = filter.filter_data(raw_data[sc_rois_cortex, :], samplingFreq, lowcut, highcut)

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
                # Load empirical data to make simple comparisons
                plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_plv.txt")

                # Comparisons
                plv_cx = plv
                plv_emp_cx = plv_emp[:, fc_rois_cortex][fc_rois_cortex]
                t1 = np.zeros(
                    shape=(2, len(conn.region_labels[sc_rois_cortex]) ** 2 // 2 - len(
                        conn.region_labels[sc_rois_cortex]) // 2))
                t1[0, :] = plv_cx[np.triu_indices(len(conn.region_labels[sc_rois_cortex]), 1)]
                t1[1, :] = plv_emp_cx[np.triu_indices(len(conn.region_labels[sc_rois_cortex]), 1)]
                plv_r = np.corrcoef(t1)[0, 1]

                ## dFC
                # Sliding window parameters
                window, step = 4, 2  # seconds

                dFC = dynamic_fc(filterSignals, samplingFreq, window, step, "PLV", filtered=True)
                dFC_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")

                # Compare dFC vs dFC_emp
                t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
                t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
                t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
                dFC_ksd = \
                scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

                ## Metastability: Kuramoto Order Parameter
                sdKO, avgKO = kuramoto_order(filterSignals, samplingFreq, filtered=True)
                sdKO_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_sdKO.txt")

                results_fc.append((g, s, r, bands[0][b], plv_r, dFC_ksd, sdKO, sdKO_emp))
            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))


## GATHER RESULTS
simname = test + subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")

# Working on results
df = pd.DataFrame(results_fc, columns=["g", "s", "rep", "band", "PLVr", "dFC_ksd", "meanKO", "sdKO", "sdKO_emp"])
df.to_csv(specific_folder+"/PSE_"+simname+".csv", index=False)

df_m = df.groupby(["g", "s", "band"])[["PLVr", "dFC_ksd", "sdKO", "sdKO_emp"]].mean().reset_index()
df_sd = df.groupby(["g", "s", "band"])[["PLVr", "dFC_ksd", "sdKO", "sdKO_emp"]].std().reset_index()


fig = make_subplots(rows=3, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                    specs=[[{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}]],
                    shared_yaxes=True, shared_xaxes=True, row_titles=["FCr", "dFC (ksd)", "KO"],
                    x_title="Conduction speed (m/s)", y_title="Coupling factor")

for i, band in enumerate(range(len(bands[0]))):

    df_subset = df_m.loc[df_m["band"] == bands[0][band]]

    fig.add_trace(go.Heatmap(z=df_subset.PLVr, x=df_subset.s, y=df_subset.g, colorscale='RdBu', reversescale=True,
                             colorbar=dict(title="avg rFC", y=0.9, len=0.3, thickness=10), zmin=-0.5, zmax=0.5, showlegend=False), row=1, col=i+1)

    fig.add_trace(go.Heatmap(z=df_subset.dFC_ksd, x=df_subset.s, y=df_subset.g, colorscale='Viridis',
                              zmin=0, zmax=1, colorbar=dict(title="ksd", y=0.5, len=0.3, thickness=10)), row=2, col=i+1)

    fig.add_trace(go.Heatmap(z=df_subset.sdKO, x=df_subset.s, y=df_subset.g,
                             colorbar=dict(title="ko", thickness=10, y=0.1, len=0.3), zmin=0, zmax=1), row=3, col=i+1)

fig.update_layout(title_text='%s simulations || [w = %0.2f] %s' % (test, w, emp_subj))
pio.write_html(fig, file=specific_folder + "/paramSpace-%s.html" % test, auto_open=True)


