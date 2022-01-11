import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_th, JansenRitDavid2003_N

import scipy
from mne import filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


from toolbox import timeseriesPlot, FFTplot, FFTpeaks, multitapper, epochingTool, PLV

"""

This is a whole brain model. Integrating David et al. extensions of the Jansen-Rit model and
some thalamo-cortical, cortico-cortical connectivity details in Jones et al. research, we build
this whole brain model where thalamic input to cortex will be processed differently (see in coupling script) 
as FeedForward connections and cortical inputs to other rois will be processed as feedback connections. 

"""

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
modelid = ".2021JansenRitDavid-Jones"
emp_subj = "NEMOS_035"

wd = os.getcwd()
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data2\\"

main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
specific_folder = main_folder + "\\PSE" + emp_subj + "jrd-j-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)

tic0 = time.time()

samplingFreq = 1000  # Hz
simLength = 16000  # ms - relatively long simulation to be able to check for power distribution
transient = 4000  # seconds to exclude from timeseries due to initial transient


conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
                 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R',
                 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
                 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L',
                 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R',
                 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L',
                 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
                 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
                 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L',
                 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
                 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L',
                 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R']
cingulum_rois = ['Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
                 'Cingulum_Post_L',
                 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R',
                 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L',
                 'Amygdala_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                 'Parietal_Inf_L', 'Parietal_Inf_R', 'Precuneus_L',
                 'Precuneus_R', 'Thalamus_L', 'Thalamus_R']

# load text with FC rois; check if match SC
FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]
SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois



sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.022, 0)
p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.22, 0)

w = np.array([0.8] * len(conn.region_labels))

# Parameters edited from David and Friston (2003).
m = JansenRitDavid2003_th(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                          tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),

                          He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                          tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                          w=np.array([0.8]), c=np.array([135.0]),
                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                          v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                          p=np.array([p_array]), sigma=np.array([sigma_array]))

# Remember to hold tau*H constant.
m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

ff_vals = np.arange(0.01, 0.75, 0.05)  # original 0.1
fb_vals = np.arange(0.01, 1.5, 0.1)  # original 0.2

results_fft_peak = []
results_fc = []

g, s = 60, 5.5  # rFC(cortex) = 0.38

for ff in ff_vals:
    for fb in fb_vals:
        tic = time.time()
        # Coupling function; where thalamus will behave differently
        th_ids = np.squeeze(np.argwhere((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L')))
        coup = coupling.SigmoidalJansenRitDavid_th(a=np.array([g]), th=th_ids, FF=np.array([ff]), FB=np.array([fb]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
        conn.speed = np.array([5.5])

        mon = (monitors.Raw(),)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()

        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.2f sec" % (time.time() - tic0,))
        # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
        raw_time = output[0][0][transient:]
        regionLabels = conn.region_labels

        # Subset analysis to cortical rois
        raw_data = raw_data[SC_cortex_idx, :]

        # Check initial transient and cut data
        # timeseriesPlot(raw_data, raw_time, regionLabels, main_folder, mode="html")
        # Fourier Analysis plot
        # fft = multitapper(raw_data, samplingFreq, regionLabels, 4, 4, 0.5, plot=True, peaks=True, folder=main_folder)
        indexes=[int(id) for id in th_ids]
        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
        _, _, IAF, _, bmodules = multitapper(raw_data, samplingFreq, peaks=True)
        results_fft_peak.append((g, s, np.average(IAF), np.average(bmodules), np.average(bmodules)/np.average(bmodules[indexes])))

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
                efPhase.append(np.angle(analyticalSignal))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)

            ## PLE - Phase Lag Entropy
            ## PLE parameters - Phase Lag Entropy
            # tau_ = 25  # ms
            # m_ = 3  # pattern size
            # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)
            # results_ple.append((g, s, r, np.average(ple[np.triu_indices(len(ple[0]), 1)])))

            # Load empirical data to make simple comparisons
            plv_emp = np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:, FC_cortex_idx][
                FC_cortex_idx]

            # Comparisons
            t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
            t1[0, :] = plv[np.triu_indices(len(plv), 1)]
            t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
            plv_r = np.corrcoef(t1)[0, 1]
            results_fc.append((g, s, plv_r))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

simname = emp_subj + "-t" + str(simLength) + "_" + str(s) + "-" + time.strftime("t%Hh.%Mm.%Ss")

if "Jesus CabreraAlvarez" in os.getcwd():
    df_fft = pd.DataFrame(results_fft_peak, columns=["G", "speed", "mean_freqPeak", "mean_bmodule", "AllvsTh_mod"])
    df_fft.to_csv(specific_folder+"/PSE_FFTpeaks" + simname + ".csv", index=False)

    df_fc = pd.DataFrame(results_fc, columns=["G", "speed",  "Alpha"])
    df_fc.to_csv(specific_folder+"/PSE_FC" + simname + ".csv", index=False)

# else:
#     df_fft = pd.DataFrame(results_fft_peak, columns=["G", "speed", "mean_freqPeak", "mean_bmodule", "AllvsTh_mod"])
#     df_fft.to_csv("PSE_FFTpeaks" + simname + ".csv", index=False)
#
#     df_fc = pd.DataFrame(results_fc, columns=["G", "speed",  "Alpha"])
#     df_fc.to_csv("PSE_FC" + simname + ".csv", index=False)

    # df_ple = pd.DataFrame(results_ple, columns=["G", "speed", "round", "Alpha"])
    # df_ple.to_csv("PSE_PLE" + simname + ".csv", index=False)


## Plotting
fig = make_subplots(rows=1, cols=3, subplot_titles=("allFreq", "rFC _Alpha", "allModule/thModule"),
                    specs=[[{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                    x_title="FeedForward connectivity", y_title="Feedback connectivity")

fig.add_trace(go.Heatmap(z=df_fft.mean_freqPeak, x=df_fft.ff, y=df_fft.fb, colorscale='Viridis',showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=df_fc.Alpha, x=df_fc.ff, y=df_fc.fb, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5, showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z=df_fft.AllvsTh_mod, x=df_fft.ff, y=df_fft.fb, showscale=False, zmin=0, zmax=3, colorscale='Viridis'), row=1, col=3)

title = 'ff&fb'
fig.update_layout(
    title_text='FC correlation (empirical - simulated data) || %s' % title)
pio.write_html(fig, file=specific_folder + "/paramSpace-g&s_%s.html" % title, auto_open=True)
