
import os

import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import pickle


folder = "E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\PSE\PSEmpi_dynSys-m03d27y2022-t11h.18m.31s\\"
file = folder + "p_single_results.pkl"
with open(file, "rb") as f:
    results = pickle.load(f)
    f.close()


## With simtime calculate freqs
lowcut, highcut = 2, 40

freqs = np.arange(results["simtime"].iloc[0] / 2)
freqs = freqs / ((results["simtime"].iloc[0]) / 1000)  # simLength (ms) / 1000 -> segs

freqs = freqs[(freqs > lowcut) & (freqs < highcut)]  # remove undesired frequencies


cmap = px.colors.qualitative.Plotly

model = "jr"

#######          SINGLE SPACES --
## 3d Phase Portraits
fig = make_subplots(rows=2, cols=3, shared_yaxes=True, column_titles=["p_single", "sigma_single", "w_single"],
                    horizontal_spacing=0, vertical_spacing=0,
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                           [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]])

for i, space in enumerate(["p_single", "sigma_single", "w_single"]):
    space_param = space.split("_")[0]
    ## SPACE 1 : p_single  ## SPACE 2 : sigma_single   ## SPACE 3 : w_single
    print("Working on %s" % space)

    file = folder + space + "_results.pkl"
    with open(file, "rb") as f:
        results = pickle.load(f)
        f.close()

    ## Results contain signals and fft for two regions: a thalamic node [0] with default parameters (noisy);
    # and a cortical node [1] without noise
    ### Now, work on results to average ffts and to select just one signal - add to plot_df

    # Average spectra
    df_avgFFT = results.groupby(["model", space_param]).mean().reset_index()
    df_avgFFT["color"] = [cmap[i%len(cmap)] for i in range(len(df_avgFFT))]

    # subset per model
    df_results = results.loc[results["model"] == model]


    # - Bifurcation
    for series in df_results.iterrows():

        ## select the avg curved to plot in Thalamus [0]
        temp_fft = df_avgFFT["fft"].loc[(df_avgFFT["model"]==model) & (df_avgFFT[space_param]==series[1][space_param])].values[0][0]
        ## Select color
        color = df_avgFFT["color"].loc[(df_avgFFT["model"]==model) & (df_avgFFT[space_param]==series[1][space_param])].values[0]


        fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(temp_fft), y=freqs, z=temp_fft,
                                   marker_color=color, showlegend=False), row=2, col=i+1)

        if series[1]["psp_t"][0].nbytes < 20:
            fig.add_trace(go.Scatter3d(x=[series[1][space_param]], y=[series[1]["psp_dt"][0]],
                                       z=[series[1]["psp_t"][0]], marker_color=color,
                                       showlegend=False), row=1, col=i+1)
        else:
            fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(series[1]["psp_t"][0]),
                                       y=series[1]["psp_dt"][0], z=series[1]["psp_t"][0], marker_color=color,
                                       showlegend=False), row=1, col=i+1)

# TODO legend: group by param (i.e. "p"), name param value (or range); initial camera perspective
fig.update_traces(marker_size=3)
camera_bif = dict(eye=dict(x=1.4, y=1.85, z=0.2))
camera_bif2 = dict(eye=dict(x=1.6, y=1.4, z=0.2))
camera_fft = dict(eye=dict(x=1.6, y=1.6, z=0.1))

fig.update_layout(title=model, template="plotly_white",
                  scene1=dict(camera=camera_bif, xaxis=dict(title="p", autorange="reversed"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),
                  scene2=dict(camera=camera_bif2, xaxis=dict(title="sigma", autorange="reversed"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),
                  scene3=dict(camera=camera_bif2, xaxis=dict(title="w", autorange="reversed"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),

                  scene4=dict(camera=camera_fft, xaxis=dict(title="p", autorange="reversed"), yaxis=dict(title="Frequency (Hz)"), zaxis=dict(title="Module (dB)")),
                  scene5=dict(camera=camera_fft, xaxis=dict(title="sigma", autorange="reversed"), yaxis=dict(title="Frequency (Hz)"),zaxis=dict(title="Module (dB)")),
                  scene6=dict(camera=camera_fft, xaxis=dict(title="w", autorange="reversed"), yaxis=dict(title="Frequency (Hz)"),zaxis=dict(title="Module (dB)")),

                  width=1400, height=800)

pio.write_html(fig, file="E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES\\" + model + '_spaces_SINGLE.html', auto_open=False)

fig.write_image("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES\\" + model + "_spaces_SINGLE.svg")





model = "jrd"

#######          SINGLE SPACES --
## 3d Phase Portraits
fig = make_subplots(rows=2, cols=3, shared_yaxes=True, column_titles=["p_single", "sigma_single", "w_single"],
                    horizontal_spacing=0, vertical_spacing=0,
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                           [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]])

for i, space in enumerate(["p_single", "sigma_single", "w_single"]):
    space_param = space.split("_")[0]
    ## SPACE 1 : p_single  ## SPACE 2 : sigma_single   ## SPACE 3 : w_single
    print("Working on %s" % space)

    file = folder + space + "_results.pkl"
    with open(file, "rb") as f:
        results = pickle.load(f)
        f.close()

    ## Results contain signals and fft for two regions: a thalamic node [0] with default parameters (noisy);
    # and a cortical node [1] without noise
    ### Now, work on results to average ffts and to select just one signal - add to plot_df

    # Average spectra
    df_avgFFT = results.groupby(["model", space_param]).mean().reset_index()
    df_avgFFT["color"] = [cmap[i%len(cmap)] for i in range(len(df_avgFFT))]

    # subset per model
    df_results = results.loc[results["model"] == model]


    # - Bifurcation
    for series in df_results.iterrows():

        ## select the avg curved to plot in Thalamus [0]
        temp_fft = df_avgFFT["fft"].loc[(df_avgFFT["model"]==model) & (df_avgFFT[space_param]==series[1][space_param])].values[0][0]
        ## Select color
        color = df_avgFFT["color"].loc[(df_avgFFT["model"]==model) & (df_avgFFT[space_param]==series[1][space_param])].values[0]


        fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(temp_fft), y=freqs, z=temp_fft,
                                   marker_color=color, showlegend=False), row=2, col=i+1)

        if series[1]["psp_t"][0].nbytes < 20:
            fig.add_trace(go.Scatter3d(x=[series[1][space_param]], y=[series[1]["psp_dt"][0]],
                                       z=[series[1]["psp_t"][0]], marker_color=color,
                                       showlegend=False), row=1, col=i+1)
        else:
            fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(series[1]["psp_t"][0]),
                                       y=series[1]["psp_dt"][0], z=series[1]["psp_t"][0], marker_color=color,
                                       showlegend=False), row=1, col=i+1)

fig.update_traces(marker_size=2)
camera_bif = dict(eye=dict(x=1.5, y=1.6, z=0.1))
camera_bif2 = dict(eye=dict(x=1.8, y=1.4, z=0.1))
camera_fft = dict(eye=dict(x=1.5, y=1.5, z=0.1))

fig.update_layout(title=model, template="plotly_white",
                  scene1=dict(camera=camera_bif, xaxis=dict(title="p", autorange="reversed"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),
                  scene2=dict(camera=camera_bif2, xaxis=dict(title="sigma", autorange="reversed"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),
                  scene3=dict(camera=camera_bif2, xaxis=dict(title="w"), yaxis=dict(title="d(PSP)"), zaxis=dict(title="PSP (mV)")),

                  scene4=dict(camera=camera_fft, xaxis=dict(title="p", autorange="reversed"), yaxis=dict(title="Frequency (Hz)"), zaxis=dict(title="Module (dB)")),
                  scene5=dict(camera=camera_fft, xaxis=dict(title="sigma", autorange="reversed"), yaxis=dict(title="Frequency (Hz)"),zaxis=dict(title="Module (dB)")),
                  scene6=dict(camera=camera_fft, xaxis=dict(title="w"), yaxis=dict(title="Frequency (Hz)"),zaxis=dict(title="Module (dB)")),

                  width=1400, height=800)

pio.write_html(fig, file="E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES\\" + model + '_spaces_SINGLE.html', auto_open=False)

fig.write_image("E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\poster_FIGURES\\" + model + "_spaces_SINGLE.svg")


