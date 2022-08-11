
import time
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import scipy.integrate
import pickle

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.express as px

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.fft import multitapper
from toolbox.signals import epochingTool
from toolbox.fc import PLV
from toolbox.mixes import timeseries_spectra


"""

Working points process: 1. afterCluster_WorkingPoints --> 2. afterCluster_GD --> 3. here! (complete with WP simulations-FFT)
     outputs: 1. PSE/dynSys-WorkingPoints/.csv   2. CORTICON/Working_points.pkl   3. CORTICON/Working_points_Full.pkl    
      
"""


ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
spectra_folder = "E:\LCCN_Local\PycharmProjects\SpectraAnalysis\\fieldtrip_data_AAL2\\"


# def simulate_fft(model, emp_subj, conn, g, s, w, signals=False):
#
#     simLength = 30000
#     transient = 10000
#     samplingFreq = 1000
#
#
#     conn.speed = np.array([s])
#
#     if "mix" in model:
#         p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])
#         sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
#
#     elif "def" in model:
#         p_array = np.asarray([0.22 if 'Thal' in roi else 0.22 for roi in conn.region_labels])
#         sigma_array = np.asarray([0.022 if 'Thal' in roi else 0.022 for roi in conn.region_labels])
#
#
#     if "jrd_" in model:  # JANSEN-RIT-DAVID
#         # Parameters edited from David and Friston (2003).
#         m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
#                                tau_e1=np.array([10]), tau_i1=np.array([20]),
#                                He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
#                                tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
#
#                                w=np.array([w]), c=np.array([135.0]),
#                                c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#                                c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#                                v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
#                                p=p_array, sigma=sigma_array,
#
#                                variables_of_interest=["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
#                                                       "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"])
#
#         coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
#
#         # # Remember to hold tau*H constant: Spiegler (2010) pp.1045;
#         m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
#         m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])
#
#     elif "jr_" in model:  # JANSEN-RIT
#         m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
#                           tau_e=np.array([10]), tau_i=np.array([20]),
#
#                           c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#                           c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#                           c=np.array([135.0]), p=p_array, sigma=sigma_array,
#                           e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),
#
#                           variables_of_interest=["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"])
#
#         coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
#                                            r=np.array([0.56]))
#         # Remember to hold tau*H constant.
#         m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])
#
#     conn.speed = np.array([s])
#
#     # OTHER PARAMETERS   ###
#     # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
#     # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
#     integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
#
#     mon = (monitors.Raw(),)
#
#     print("Simulating %s (%is)  || structure: %s \nPARAMS: g%i s%i w%0.2f" %
#           (model, simLength / 1000,  emp_subj,  g, s, w))
#
#     # Run simulation
#     sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
#     sim.configure()
#     output = sim.run(simulation_length=simLength)
#
#     # Extract data: "output[a][b][:,0,:,0].T" where:
#     # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
#     if "jr_" in model:
#         psp_t = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
#
#     elif "jrd_" in model:
#         psp_t = w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
#                 (1 - w) * (output[0][1][transient:, 7, :, 0].T - output[0][1][transient:, 8, :, 0].T)
#
#     # Calculate simulated spectra
#     sim_freqs, sim_spectra = multitapper(psp_t, samplingFreq, conn.region_labels, epoch_length=4, ntapper=4, smoothing=0.5,
#                                          plot=False)
#     if signals:
#         return sim_spectra, sim_freqs, psp_t
#     else:
#         return sim_spectra, sim_freqs


## Complete Working points dataframe by simulating and comparing spectra
# with open("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\WorkingPoints.pkl", 'rb') as f:
#     working_points = pickle.load(f)
#     f.close()
#
# for i, wp in working_points.loc[(working_points["subj"] != "NEMOS_AVG") & (working_points["model"] != "jrd-het_mix")].iterrows():
#
#     model, emp_subj, _, g, s, _, _, _, _, _, _, = wp
#
#     ## STRUCTURE
#     conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh.zip")
#     conn.weights = conn.scaled_weights(mode="tract")
#
#     simulations = np.asarray([simulate_fft(model, emp_subj, conn, g, s, 0.8) for r in range(3)])
#
#     # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
#     cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
#                      'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
#                      'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
#                      'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
#                      'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
#                      'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
#                      'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
#                      'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
#                      'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
#                      'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
#                      'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
#                      'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
#                      'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
#                      'ParaHippocampal_R', 'Calcarine_L',
#                      'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
#                      'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
#                      'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
#                      'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
#                      'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
#                      'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
#                      'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
#                      'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
#                      'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
#                      'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
#                      'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
#                      'Temporal_Inf_R']
#
#     SClabs = list(conn.region_labels)
#     SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]
#
#     FFTlabs = list(np.loadtxt(spectra_folder + emp_subj + "/labels.txt", delimiter=",", dtype=str))
#     FFT_cortex_idx = [FFTlabs.index(roi) for roi in cortical_rois]
#
#
#     sim_ffts = np.average(simulations[:, 0], axis=0)[SC_cortex_idx]
#     sim_freqs = simulations[:, 1][0]
#
#     emp_ffts = np.loadtxt(spectra_folder + emp_subj + "\\flat_fft.txt", delimiter=" ")[FFT_cortex_idx]
#     emp_freqs = np.loadtxt(spectra_folder + emp_subj + "\\flat_freqs.txt", delimiter=" ")
#
#     rFFTs = [np.corrcoef(sim_ffts[roi], emp_ffts[roi, :len(sim_ffts[0])])[0, 1] for roi in range(len(sim_ffts))]
#
#     working_points["rFFT"].iloc[i] = rFFTs
#
#     working_points["sim_ffts"].iloc[i] = sim_ffts
#     working_points["emp_ffts"].iloc[i] = emp_ffts
#     working_points["sim_freqs"].iloc[i] = sim_freqs
#     working_points["emp_freqs"].iloc[i] = emp_freqs
#
#
#
# ## To merge new GD info with already simulated working points
# with open("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON/WorkingPoints100it_Full.pkl", 'rb') as f:
#     working_points100 = pickle.load(f)
#     f.close()
#
# with open("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON/WorkingPoints.pkl", 'rb') as f:
#     working_points = pickle.load(f)
#     f.close()
#
# working_points["def_mix"] = [m.split("_")[1] for m in working_points["model"].values]
# working_points["model_"] = [m.split("_")[0] for m in working_points["model"].values]
# working_points = working_points.loc[working_points["subj"] != "NEMOS_AVG"]
#
# working_points.iloc[:-10] = working_points100.iloc[:-10]
#
# file_params = open("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\WorkingPoints250it_Full.pkl", "wb")
# pickle.dump(working_points, file_params)
# file_params.close()

### Load latest df
with open("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON/WorkingPoints100it_Full.pkl", 'rb') as f:
    working_points = pickle.load(f)
    f.close()

######    STATS: rPLV
import pingouin as pg

# T-tests
pw = pg.pairwise_ttests(working_points, dv="max_r_plv", within="model", subject="subj")
pw = pw.iloc[[2, 4, 6]]
pw["p-corr"] = pg.multicomp(pw["p-unc"].values, method="bonferroni")[1]

# Wilcoxon's  [This one: sample too small to test normality]
Wdef = pg.wilcoxon(x=working_points["max_r_plv"].loc[working_points["model"]=="jr_def"].values, y=working_points["max_r_plv"].loc[working_points["model"]=="jrd_def"].values)
Wmix = pg.wilcoxon(x=working_points["max_r_plv"].loc[working_points["model"]=="jr_mix"].values, y=working_points["max_r_plv"].loc[working_points["model"]=="jrd_mix"].values)
Wmix_het = pg.wilcoxon(x=working_points["max_r_plv"].loc[working_points["model"]=="jr_mix"].values, y=working_points["max_r_plv"].loc[working_points["model"]=="jrd-het_mix"].values)

# Friedmans anova
friedmans = pg.friedman(data=working_points.loc[(working_points["model"]=="jr_mix") | (working_points["model"]=="jrd_mix") | (working_points["model"]=="jrd-het_mix")],
            dv="max_r_plv", within="model", subject="subj")


fig = px.violin(working_points, x="def_mix", y="max_r_plv", color="model_", points="all")
fig.update_layout(template="plotly_white", legend=dict(orientation="h"))

text = "Wilcoxon=0.0, p<0.001<br><br>***"
fig.add_annotation(x="def", y=0.65, text=text, font=dict(size=14), showarrow=False)

text = "Friedman's=0.43, p<0.05"
fig.add_annotation(x="mix", y=0.7, text=text, font=dict(size=14), showarrow=False)

fig.show(renderer="browser")


#######    STATS: rFFT

# transform dataframe to long: model_, def_mix, rFFT, subj,
model_temp, subj_temp, rfft_temp, def_mix_temp, model__temp = [], [], [], [], []

for i, row in working_points.iterrows():

    model, subj, _, _, _, rFFT, _, _, _, _, _, def_mix, model_ = row

    model_temp += [model] * len(rFFT)
    subj_temp += [subj] * len(rFFT)
    rfft_temp += rFFT
    def_mix_temp += [def_mix] * len(rFFT)
    model__temp += [model_] * len(rFFT)

df_long = pd.DataFrame(np.asarray([model_temp, subj_temp, rfft_temp, def_mix_temp, model__temp]).T,
                       columns=["model", "subject", "rFFT", "def_mix", "model_"])
df_long["rFFT"] = df_long["rFFT"].astype('float')

# T-tests
pw = pg.pairwise_ttests(df_long, dv="rFFT", within="model", subject="subject")
pw = pw.iloc[[1, 4]]
pw["p-corr"] = pg.multicomp(pw["p-unc"].values, method="bonferroni")[1]

# Wilcoxon's
Wdef = pg.wilcoxon(x=df_long["rFFT"].loc[df_long["model"]=="jr_def"].values, y=df_long["rFFT"].loc[df_long["model"]=="jrd_def"].values)
Wmix = pg.wilcoxon(x=df_long["rFFT"].loc[df_long["model"]=="jr_mix"].values, y=df_long["rFFT"].loc[df_long["model"]=="jrd_mix"].values)
Wmix_het = pg.wilcoxon(x=df_long["rFFT"].loc[df_long["model"]=="jr_mix"].values, y=df_long["rFFT"].loc[df_long["model"]=="jrd-het_mix"].values)

# Friedmans anova
friedmans = pg.friedman(data=df_long.loc[(df_long["model"]=="jr_mix") | (df_long["model"]=="jrd_mix") | (df_long["model"]=="jrd-het_mix")],
            dv="rFFT", within="model", subject="subject")

fig = px.violin(df_long, x="def_mix", y="rFFT", color="model_", points="all")
fig.update_layout(template="plotly_white")

# text = ""
# fig.add_annotation(x="def", y=0.65, text=text, font=dict(size=14), showarrow=False)
# text = ""
# fig.add_annotation(x="mix", y=0.65, text=text, font=dict(size=14), showarrow=False)
fig.show(renderer="browser")



## Plot some spectra

subj = "NEMOS_035"

conn = connectivity.Connectivity.from_file(ctb_folder + subj + "_AAL2pTh.zip")
conn.region_labels
rois = [0,1,2,3,14,15,30,31,38,39,50,51,60,61,80,81]
conn.region_labels[rois]

cmap = px.colors.qualitative.Plotly

for i, model in enumerate(["jr_mix", "jrd_mix", "jrd-het_mix"]):

    temp = working_points.loc[(working_points["model"] == model) & (working_points["subj"] == subj)]

    # Spectra
    sim_fft = temp["sim_ffts"].values[0]
    sim_fft[sim_fft<0] = 0
    sim_freqs = temp["sim_freqs"].values[0]

    emp_fft = temp["emp_ffts"].values[0]
    emp_fft[emp_fft<0] = 0
    emp_freqs = temp["emp_freqs"].values[0]

    for roi in rois:

        fig = go.Figure(go.Scatter(x=sim_freqs, y=sim_fft[roi], marker_color=cmap[i],
                                   name=conn.region_labels[roi], opacity=0.8, showlegend=False))

        fig.update_layout(xaxis=dict(title="Frequency (Hz)"), yaxis=dict(title="Power (dB)"),
                          template="plotly_white", height=300, width=300)

        pio.write_image(fig, file="E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES/" + conn.region_labels[roi] + model + ".svg")

        if i==0:

            fig = go.Figure(go.Scatter(x=emp_freqs, y=emp_fft[roi], marker_color=cmap[4],
                                       name=conn.region_labels[roi], opacity=0.8, showlegend=False))

            fig.update_layout(xaxis=dict(title="Frequency (Hz)"), yaxis=dict(title="Power (dB)"),
                              template="plotly_white", height=300, width=300)

            pio.write_image(fig, file="E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES/" +
                                      conn.region_labels[roi] + "empirical.svg")





## COMPUTE SPECTRA AND COMPARE to EMPIRICAL   ######

tests = [["jr_def", "NEMOS_035", 0, 15, 0.8],
         ["jr_mix", "NEMOS_035", 0, 15, 0.8],
         ["jrd_def", "NEMOS_035", 0, 15, 0.8],       ## Single nodes
         ["jrd_mix", "NEMOS_035", 0, 15, 0.8],

         ["jr_def", "NEMOS_035", 50, 15, 0.8],
         ["jrd_def", "NEMOS_035", 50, 15, 0.8],
         ["jr_mix", "NEMOS_035", 50, 15, 0.8],    ## Coupled nodes
         ["jrd_mix", "NEMOS_035", 50, 15, 0.8]]

simLength = 30000
transient = 10000
samplingFreq = 1000

simulations = []
for test in tests:

    model, emp_subj, g, s, w = test

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    rois = [140, 2]  # Thal_PuM_L, Frontal_Sup_2_L  ::  weight = 0.03 in AAL2pTh

    conn.weights = conn.weights[:, rois][rois]
    conn.tract_lengths = conn.tract_lengths[:, rois][rois]
    conn.region_labels = conn.region_labels[rois]

    simulations.append(simulate_fft(model, emp_subj, conn, g, s, 0.8, signals=True))

    timeseries_spectra()

def timeseries_spectra(signals, simLength, transient, regionLabels, mode="html", folder="figures",
                       freqRange=[2,40], opacity=1, title=None, auto_open=True):

    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], column_widths=[0.65, 0.35])

    timepoints = np.arange(start=transient, stop=simLength, step=len(signals[0])/(simLength-transient))

    cmap = px.colors.qualitative.Plotly

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    for i, signal in enumerate(signals):

        # Timeseries
        fig.add_trace(go.Scatter(x=timepoints, y=signal, name=regionLabels[i], opacity=opacity,
                                 legendgroup=regionLabels[i], marker_color=cmap[i%len(cmap)]), row=1, col=1)

        # Spectra
        fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
        fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT

        fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies


        fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                 marker_color=cmap[i%len(cmap)], name=regionLabels[i], opacity=opacity,
                                 legendgroup=regionLabels[i], showlegend=False), row=1, col=2)


        fig.update_layout(xaxis=dict(title="Time (ms)"), xaxis2=dict(title="Frequency (Hz)"),
                          yaxis=dict(title="Voltage (mV)"), yaxis2=dict(title="Power (dB)"),
                          template="plotly_white", title=title)

    if mode == "html":
        pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=auto_open)








# ## POSTER plot
# # 3 subplots: 1 props, 2 corr + stats, 3 spectra
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.io as pio
#
# fig = make_subplots(rows=1, cols=4, column_widths=[0.2, 0.2, 0.2, 0.4], shared_yaxes=True,
#                     column_titles=["max prop", "r(emp-sim spectra)<br>"+rS_toprint, "rPLV<br>"+rP_toprint, "spectra"])
#
# cmap = px.colors.qualitative.Safe
# opacity = 0.9
# sample_subject = 49 # goods: 77
# roi = 'Occipital_Mid_L'
# for color, mode in enumerate(["jr",  "emp", "jrd_pTh", "jrd_pTh_hetero"]):
#
#     # props
#     x = [mode]*len(df["max_prop"].loc[df["mode"] == mode].values)
#     y = df["max_prop"].loc[df["mode"] == mode].values
#     fig.add_trace(go.Violin(x=x, y=y, marker_color=cmap[color], opacity=opacity, showlegend=False), row=1, col=1)
#
#     # spectra
#     fft = df["fft"].loc[(df["subj"] == "NEMOS_0" + str(sample_subject)) & (df["mode"] == mode) & (df["roi"] == roi)].iloc[0]
#     fft[fft < 0] = 0
#     freqs = df["freqs"].loc[(df["subj"] == "NEMOS_0" + str(sample_subject)) & (df["mode"] == mode) & (df["roi"] == roi)].iloc[0]
#     normalization = abs(np.trapz(freqs, fft))  # integral
#     # normalization = max(fft)
#     # normalization = sum(fft)
#     fig.add_trace(go.Scatter(x=freqs, y=fft / normalization, marker_color=cmap[color], name=mode), row=1, col=4)
#
#     # corr test
#     if mode != "emp":
#
#         x = [mode] * len(df["rSpectra"].loc[df["mode"] == mode].values)
#         y = df["rSpectra"].loc[df["mode"] == mode].values
#         fig.add_trace(go.Violin(x=x, y=y, marker_color=cmap[color], opacity=opacity, showlegend=False), row=1, col=2)
#
#         x = [mode] * len(df["rPLV"].loc[df["mode"] == mode].values)
#         y = df["rPLV"].loc[df["mode"] == mode].values
#         fig.add_trace(go.Scatter(x=x, y=y, marker_color=cmap[color], mode="markers", opacity=opacity, showlegend=False), row=1, col=3)
#
#
# fig.update_layout(xaxis4=dict(range=[0, 45]))
#
# pio.write_html(fig, file=specific_folder + '/results.html',auto_open=True)




