
import time
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import sys

sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.fft import multitapper
from toolbox.signals import epochingTool
from toolbox.fc import PLV
from toolbox.mixes import timeseries_spectra

output_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES\\"
ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"


def simulate(params):

    model, struct, emp_subj, rois, r, g, s, w, p, sigma = params

    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"

    # Prepare simulation parameters
    simLength = 20 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_" + struct + ".zip")
    conn.weights = conn.scaled_weights(mode="tract")

    FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
    SClabs = list(conn.region_labels)

    if "Net" not in space:  # If we want to simulate with just a subset
        conn.weights = conn.weights[:, rois][rois]
        conn.tract_lengths = conn.tract_lengths[:, rois][rois]
        conn.region_labels = conn.region_labels[rois]

    elif "cbNet" in space:
        cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R'] + [roi for roi in conn.region_labels if
                                                                            'Thal' in roi]
        cb_rois = [SClabs.index(roi) for roi in cingulum_rois]

        conn.weights = conn.weights[:, cb_rois][cb_rois]
        conn.tract_lengths = conn.tract_lengths[:, cb_rois][cb_rois]
        conn.region_labels = conn.region_labels[cb_rois]

        cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R', 'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R']  # Removing subcorticals for FC analysis
        FC_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cb_rois
        SC_idx = [SClabs.index(roi) for roi in cingulum_rois]

    elif "bigNet" in space:
        cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                         'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                         'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                         'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                         'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                         'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                         'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                         'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                         'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                         'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Calcarine_L',
                         'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                         'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                         'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                         'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                         'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                         'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                         'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                         'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                         'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                         'Temporal_Inf_R']  # For Functiona analysis: remove subcorticals (i.e. Cerebelum, Thalamus, Caudate)
        FC_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
        SC_idx = [SClabs.index(roi) for roi in cortical_rois]

    # NEURAL MASS MODEL  &    COUPLING FUNCTION         ###################################################

    p_array = np.asarray([0.22 if 'Thal' in roi else p for roi in conn.region_labels])
    sigma_array = np.asarray([sigma if 'Thal' in roi else 0 for roi in conn.region_labels])

    w_array = np.array([w]) if type(w)==float else np.array(w)

    if model == "jrd":  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10]), tau_i1=np.array([20]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=w_array, c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=p_array, sigma=sigma_array,

                               variables_of_interest=["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                                                      "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"])

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

        # # Remember to hold tau*H constant: Spiegler (2010) pp.1045;
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    elif model == "jr":  # JANSEN-RIT
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),

                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                          c=np.array([135.0]), p=p_array, sigma=sigma_array,
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),

                          variables_of_interest=["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"])

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
        # Remember to hold tau*H constant.
        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

    conn.speed = np.array([s])

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    print("Simulating %s (%is)  || structure: %s \nPARAMS: g%i s%i w%0.2f p%0.4f sigma%0.4f" %
          (model, simLength / 1000, struct, g, s, w_array[0], p, sigma))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if model == "jr":
        psp_t = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
        psp_dt = output[0][1][transient:, 4, :, 0].T - output[0][1][transient:, 5, :, 0].T

    elif model == "jrd":
        psp_t = m.w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
                (1 - m.w) * (output[0][1][transient:, 7, :, 0].T - output[0][1][transient:, 8, :, 0].T)
        psp_dt = m.w * (output[0][1][transient:, 4, :, 0].T - output[0][1][transient:, 5, :, 0].T) + \
                 (1 - m.w) * (output[0][1][transient:, 10, :, 0].T - output[0][1][transient:, 11, :, 0].T)

    print("SIMULATING (%0.2f seconds) REQUIRED %0.3f seconds.\n\n" % (simLength / 1000, time.time() - tic,))

    return psp_t


def fft_func(signals_):

    fft_result = []

    for i, signal_ in enumerate(signals_):

        # Spectra
        fft_temp = abs(np.fft.fft(signal_))  # FFT for each channel signal
        fft_temp = np.asarray(fft_temp[range(int(len(signal_) / 2))])  # Select just positive side of the symmetric FFT

        fft_result.append(fft_temp)

    return fft_result


##### DECIDE What to Simulate ::
spaces = pd.DataFrame.from_dict(                            # r, g, s,  w,  p(cx), sigma(th)
    {"single_jr":   ["jr",  "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 0, 15, 0.8, 0.22, 0.022],
     "single_jrd":  ["jrd", "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 0, 15, 0.8, 0.22, 0.022],

     "coupled_jr":  ["jr",  "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 10, 15, 0.8, 0.22, 0.022],
     "coupled_jrd": ["jrd", "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 70, 15, 0.8, 0.22, 0.022],
     "coupled_jrd_w": ["jrd", "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 0, 15, [0.8, 0.2], 0.22, 0.022],

     "bigNet_jr":   ["jr",  "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 38, 15, 0.8, 0.05, 0.022],  # Optimized by 3DParamSpace
     "bigNet_jrd":  ["jrd", "AAL2pTh", "NEMOS_AVG", [140, 2], 0, 82, 15, 0.8, 0, 0.022]},

    orient="index",
    columns=["model", "struct", "emp_subj", "rois", "r", "g", "s", "w", "p", "sigma"])

# Prepare simulation parameters
simLength = 20 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 4000  # ms

for space, params in spaces.iloc[4:5].iterrows():

    model, struct, emp_subj, rois, r, g, s, w, p, sigma = params

    signals_reps = [simulate(params) for i in range(4)]

    ffts = np.average(np.asarray([fft_func(signals) for signals in signals_reps]), axis=0)

    freqs = np.arange(len(signals_reps[0][0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_" + struct + ".zip")
    if "Net" in space:
        psp_ts = signals_reps[0][rois, :]
        regionLabels = conn.region_labels[rois]
    else:
        psp_ts = signals_reps[0]
        regionLabels = conn.region_labels

    freqRange = [1, 50]

    plot_length = 2000  # ms

    ## PLOTs
    title = space + "_g" + str(g) + "-p" + str(p) + "-sigma" + str(sigma)


    timepoints = np.arange(start=transient, stop=simLength, step=len(signals_reps[0][0])/(simLength-transient))

    cmap = px.colors.qualitative.Plotly

    for i, psp_t in enumerate(psp_ts):

        fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], column_widths=[0.65, 0.35])

        # Timeseries
        fig.add_trace(go.Scatter(x=timepoints[:plot_length], y=psp_t[:plot_length], name=regionLabels[i],
                                 opacity=0.7, legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)], showlegend=False), row=1, col=1)

        fft = ffts[i, (freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
        fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                 marker_color=cmap[i%len(cmap)], name=regionLabels[i], opacity=0.7,
                                 legendgroup=regionLabels[i], showlegend=False), row=1, col=2)


        fig.update_layout(xaxis=dict(title="Time (ms)"), xaxis2=dict(title="Frequency (Hz)"),
                          yaxis=dict(title="Voltage (mV)"), yaxis2=dict(title="Power (dB)"),
                          template="plotly_white", title=title, height=300, width=1200)

        pio.write_html(fig, file=output_folder + "/" + title + "_" + str(i) + ".html", auto_open=True)

        pio.write_image(fig, file=output_folder + "/" + title + "_" + str(i) + ".svg", engine="kaleido")




    #
    # if "Net" in space:
    #     timeseries_spectra(psp_t[rois, :], simLength, transient, conn.region_labels[rois], mode="html",
    #                        folder=output_folder, opacity=0.75,
    #                        freqRange=[1, 50], title=title, auto_open=True)
    #
    # else:
    #     timeseries_spectra(psp_t, simLength, transient, conn.region_labels, mode="html", folder=output_folder, opacity=0.75,
    #                        freqRange=[1, 50], title=title, auto_open=True)