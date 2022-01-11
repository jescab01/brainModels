import time

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import filter
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dynamics import dynamic_fc, kuramoto_order

if "Jesus CabreraAlvarez" in os.getcwd():
    import sys
    sys.path.append("D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\")  # temporal append
    from toolbox.fft import FFTpeaks
    from toolbox.fc import PLV
    from toolbox.signals import epochingTool
else:
    from toolbox import FFTpeaks,  PLV, PLE, epochingTool, paramSpace

# Common simulation requirements
model_id = ".1995JansenRit"
emp_subj = "NEMOS_035"
struct_cer = "wCer"
label = "dynamics" + struct_cer

## Folder structure - Local
if "Jesus CabreraAlvarez" in os.getcwd():
    wd = os.getcwd()
    ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

    main_folder = wd + "\\" + "PSE"
    if os.path.isdir(main_folder) == False:
        os.mkdir(main_folder)
    specific_folder = main_folder + "\\PSE" + label + "-" + emp_subj + model_id + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

    if os.path.isdir(specific_folder) == False:
        os.mkdir(specific_folder)

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/ths/"
    ctb_folder = wd + "CTB_data/output/"

    main_folder = "PSE"
    if os.path.isdir(main_folder) == False:
        os.mkdir(main_folder)

    os.chdir(main_folder)

    specific_folder = "PSE" + label + "-" + emp_subj + model_id + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
    if os.path.isdir(specific_folder) == False:
        os.mkdir(specific_folder)

    os.chdir(specific_folder)


## Simulation parameters
coupling_vals = np.arange(0, 3, 1)
n_rep = 1

# Prepare simulation parameters
simLength = 120 * 1000  # xs * 1000ms/1s = ms
samplingFreq = 1000  # Hz
transient = 1000  # ms

# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

results_fft_peak = list()
results_fc = list()
results_ple = list()

for struct_th in ['wpTh', 'wTh', 'woTh']:

    mode = struct_th + '-' + struct_cer

    acc_ids = {"wpTh-wpCer": [146, 147], "wpTh-wCer": [122, 123], "wpTh-woCer": [120, 121], "wTh-wpCer": [118, 119],
               "wTh-wCer": [94, 95], "wTh-woCer": [92, 93], "woTh-wpCer": [116, 117], "woTh-wCer": [92, 93], "woTh-woCer": [90, 91]}

    acc = acc_ids[mode]

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL3_" + mode + ".zip")
    fc_rois_cortex = list(range(0, 40)) + list(range(46, 74)) + list(range(78, 90))
    sc_rois_cortex = list(range(0, 34)) + acc + list(range(34, 38)) + list(range(44, 72)) + list(range(78, 90))
    fc_rois_dmn = [2, 3, 34, 35, 38, 39, 64, 65, 70, 71, 84, 85]
    sc_rois_dmn = [2, 3] + acc + [36, 37, 62, 63, 68, 69, 84, 85]

    conn.weights = conn.scaled_weights(mode="tract")

    mon = (monitors.Raw(),)

    for g in coupling_vals:
        for r in range(n_rep):

            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
            conn.speed = np.array([15.0])

            tic = time.time()

            print("Simulating for Coupling factor = %i " % g)

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)

            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            raw_data = output[0][1][transient:, 0, :, 0].T
            raw_time = output[0][0][transient:]

            bands = [["3-alpha"], [(8, 12)]]
            # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

            for b in range(len(bands[0])):
                (lowcut, highcut) = bands[1][b]

                # Band-pass filtering
                filterSignals = filter.filter_data(raw_data[sc_rois_cortex, :], samplingFreq, lowcut, highcut)

                # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))

                efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")
                # Obtain Analytical signal
                efPhase=list()
                for i in range(len(efSignals)):
                    analyticalSignal = scipy.signal.hilbert(efSignals[i])
                    # Get instantaneous phase and amplitude envelope by channel
                    efPhase.append(np.angle(analyticalSignal))

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


                # Sliding window parameters
                window, step = 4, 2  # seconds

                ## dFC
                dFC = dynamic_fc(raw_data[sc_rois_cortex, :], samplingFreq, window, step, "PLV")
                dFC_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")

                #Compare dFC vs dFC_emp
                t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
                t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
                t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
                dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

                ## Metastability: Kuramoto Order Parameter
                ko = kuramoto_order(raw_data[sc_rois_cortex, :], samplingFreq)
                ko_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_KO.txt")

                results_fc.append((g, r, mode, 'cortex', plv_r, dFC_ksd, ko, ko_emp))
                print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

simname = emp_subj + struct_cer + "-t" + str(simLength) + '_' + time.strftime("t%Hh.%Mm.%Ss")
df_fc = pd.DataFrame(results_fc, columns=["G", "round", "mode", "rois", "Alpha", "dFC_KSD", "KOsim", "KOemp"])

if "Jesus CabreraAlvarez" in os.getcwd():
    # df_fft.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)
    df_fc.to_csv(specific_folder + "/PSE_dynamics" + simname + ".csv", index=False)
    # df_ple.to_csv(specific_folder+"/PSE_PLE"+simname+".csv", index=False)
else:
    # df_fft.to_csv("PSE_FFTpeaks"+simname+".csv", index=False)
    df_fc.to_csv("PSE_FC" + simname + ".csv", index=False)
    # df_ple.to_csv("PSE_PLE"+simname+".csv", index=False)

# fig = make_subplots(rows=1, cols=6, subplot_titles=("Cortex - wpTh", "Cortex - wTh", "Cortex - woTh", "DMN - wpTh", "DMN - wTh", "DMN - woTh"),
#                     specs=[[{}, {}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
#                     x_title="Conduction speed (m/s)", y_title="Coupling factor")
#
# for i, mode in enumerate(["wpTh", "wTh", "woTh"]):
#     for j, rois in enumerate(["cortex", "dmn"]):
#
#         df_fc_temp = df_fc.loc[(df_fc["mode"] == mode) & (df_fc["rois"] == rois)]
#
#         fig.add_trace(go.Heatmap(z=df_fc_temp.Alpha, x=df_fc_temp.speed, y=df_fc_temp.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
#                                  reversescale=True, zmin=-0.5, zmax=0.5), row=1, col=i+1 + j*3)
#
# fig.update_layout(
#     title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % emp_subj)
# pio.write_html(fig, file=specific_folder + "/paramSpace-g&s_%s.html" % simname, auto_open=True)
