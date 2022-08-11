
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



for model in set(results["model"]):

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

        df = results.loc[results["model"] == model]

        # - Bifurcation
        for series in df.iterrows():
            if series[1]["psp_t"][0].nbytes < 20:
                fig.add_trace(go.Scatter3d(x=[series[1][space_param]],
                                           y=[series[1]["psp_dt"][0]], z=[series[1]["psp_t"][0]]), row=1, col=i+1)
            else:
                fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(series[1]["psp_t"][0]),
                                           y=series[1]["psp_dt"][0], z=series[1]["psp_t"][0]), row=1, col=i+1)

        # - FFTs
        # Here, average FFT repetitions
        for param_ in set(df[space_param]):
            temp, fft = df.loc[df[space_param] == param_], []
            for series in temp.iterrows():
                fft.append(series[1]["fft"][0])

            fft = np.average(np.asarray(fft), axis=0)
            fig.add_trace(go.Scatter3d(x=[param_] * len(series[1]["fft"][0]), y=freqs, z=fft), row=2, col=i+1)

    # legend: group by param (i.e. "p), name param value (or range); x,y,z labels; initial camera perspective
    fig.update_traces(marker_size=2)
    fig.update_layout(title=model)
    pio.write_html(fig, file=folder + model + '_spaces_SINGLE.html', auto_open=False)



    ####   SPACE :: p-sigma
    ## 3D Bifurcation
    space = "p-sigma_single"
    param1, param2 = space.split("_")[0].split("-")
    file = folder + space + "_results.pkl"
    with open(file, "rb") as f:
        results = pickle.load(f)
        f.close()

    df = results.loc[results["model"] == model].copy()

    df["max_psp"] = [np.max(series[1]["psp_t"][0]) if series[1]["psp_t"][0].nbytes > 20 else series[1]["psp_t"][0] for
                     series in df.iterrows()]
    df["min_psp"] = [np.min(series[1]["psp_t"][0]) if series[1]["psp_t"][0].nbytes > 20 else series[1]["psp_t"][0] for
                     series in df.iterrows()]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0, vertical_spacing=0,
                        specs=[[{"type": "scene"}, {"type": "scene"}]])

    # - FFTs
    # Here, average FFT repetitions
    for param1_ in set(df[param1]):
        for param2_ in set(df[param2]):
            temp, fft = df.loc[(df[param1] == param1_) & (df[param2] == param2_)], []
            for series in temp.iterrows():
                fft.append(series[1]["fft"][0])

            fft = np.average(np.asarray(fft), axis=0)
            fft_id = np.argmax(fft)
            fig.add_trace(go.Scatter3d(x=[param1_], y=[param2_], z=[freqs[fft_id]]), row=1, col=1+1)

    # Bifurcations
    df = df.groupby(["p", "sigma"]).mean().reset_index()

    fig.add_trace(go.Scatter3d(x=df["p"], y=df["sigma"], z=df["max_psp"], mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=df["p"], y=df["sigma"], z=df["min_psp"], mode="markers"), row=1, col=1)

    fig.update_traces(marker_size=2)
    fig.update_layout(title=model)
    pio.write_html(fig, file=folder + model + '_space_SINGLE_p-sigma.html', auto_open=False)



    ######      COUPLED SPACES --
    ##############     3D Phase portraits  COUPLED
    for space in ["g_coupled", "p_coupled", "sigma_coupled"]:

        fig = make_subplots(rows=2, cols=2, shared_yaxes=True, column_titles=["Thalamus_L", "Temporal_Inf_R"],
                            horizontal_spacing=0, vertical_spacing=0,
                            specs=[[{"type": "scene"}, {"type": "scene"}],
                                   [{"type": "scene"}, {"type": "scene"}]])

        space_param = space.split("_")[0]
        ## SPACE 1 : p_single  ## SPACE 2 : sigma_single   ## SPACE 3 : w_single
        print("Working on %s" % space)

        file = folder + space + "_results.pkl"
        with open(file, "rb") as f:
            results = pickle.load(f)
            f.close()

        df = results.loc[results["model"] == model]

        # - Bifurcation
        for i in range(2):
            for series in df.iterrows():
                if series[1]["psp_t"][i].nbytes < 20:
                    fig.add_trace(go.Scatter3d(x=[series[1][space_param]],
                                               y=[series[1]["psp_dt"][i]], z=[series[1]["psp_t"][i]]), row=1, col=i+1)
                else:
                    fig.add_trace(go.Scatter3d(x=[series[1][space_param]] * len(series[1]["psp_t"][i]),
                                               y=series[1]["psp_dt"][i], z=series[1]["psp_t"][i]), row=1, col=i+1)
            # - FFTs
            # Here, average FFT repetitions and extract peak
            for param_ in set(df[space_param]):
                temp, fft = df.loc[df[space_param] == param_], []
                for series in temp.iterrows():
                    fft.append(series[1]["fft"][i])

                fft = np.average(np.asarray(fft), axis=0)
                fig.add_trace(go.Scatter3d(x=[param_] * len(series[1]["fft"][i]), y=freqs, z=fft), row=2, col=i+1)

        # legend: group by param (i.e. "p), name param value (or range); x,y,z labels; initial camera perspective
        fig.update_traces(marker_size=2)
        fig.update_layout(title=model + " " + space)
        pio.write_html(fig, file=folder + model + '_' + space + '_COUPLED.html', auto_open=False)


    ################     3D Bifurcations COUPLED
    for space in ["g-p_coupled", "g-sigma_coupled", "p-sigma_coupled"]:
        param1, param2 = space.split("_")[0].split("-")

        fig = make_subplots(rows=2, cols=2, shared_yaxes=True, column_titles=["Thalamus_L", "Temporal_Inf_R"],
                            horizontal_spacing=0, vertical_spacing=0,
                            specs=[[{"type": "scene"}, {"type": "scene"}],
                                   [{"type": "scene"}, {"type": "scene"}]])

        file = folder + space + "_results.pkl"
        with open(file, "rb") as f:
            results = pickle.load(f)
            f.close()

        df = results.loc[results["model"] == model].copy()

        for i in range(2):
            df["max_psp"] = [np.max(series[1]["psp_t"][i]) if series[1]["psp_t"][i].nbytes > 20 else series[1]["psp_t"][i] for series in df.iterrows()]
            df["min_psp"] = [np.min(series[1]["psp_t"][i]) if series[1]["psp_t"][i].nbytes > 20 else series[1]["psp_t"][i] for series in df.iterrows()]

            # - FFTs
            # Here, average FFT repetitions
            for param1_ in set(df[param1]):
                for param2_ in set(df[param2]):
                    temp, fft = df.loc[(df[param1] == param1_) & (df[param2] == param2_)], []
                    for series in temp.iterrows():
                        fft.append(series[1]["fft"][i])

                    fft = np.average(np.asarray(fft), axis=0)
                    fft_id = np.argmax(fft)
                    fig.add_trace(go.Scatter3d(x=[param1_], y=[param2_], z=[freqs[fft_id]]), row=2, col=1+i)

        df = df.groupby([param1, param2]).mean().reset_index()

        for i in range(2):

            fig.add_trace(go.Scatter3d(x=df[param1], y=df[param2], z=df["max_psp"], mode="markers"), row=1, col=i+1)
            fig.add_trace(go.Scatter3d(x=df[param1], y=df[param2], z=df["min_psp"], mode="markers"), row=1, col=i+1)

            fig.update_traces(marker_size=2)
            fig.update_layout(title=model)
            pio.write_html(fig, file=folder + model + '_space_COUPLED_' + param1 + "-" + param2 + '.html', auto_open=False)


##### BIG NETWORK SPACES ::

## SPACE 1 : ParamSpace

space = "g-p-sigma_bigNet_plv"
version = "v2"
print("Working on %s" % space)

file = folder + space + "_results_"+version+".pkl"
with open(file, "rb") as f:
    results = pickle.load(f)
    f.close()


##### SIMPLE slider
import plotly.graph_objects as go
import numpy as np

df_avg = results.groupby(["model", "g", "p", "sigma"]).mean().reset_index()
df_avg["size"] = np.abs(df_avg["rPLV"] * 25)

# Create figure
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                    column_titles=["JR", "JRD"], horizontal_spacing=0)

template = "g (coupling factor): %{x} <br>p (input): %{y} <br>sigma (noise std): %{z} <br> rPLV: %{customdata}"

# Add traces, one for each slider step
steps = []
range_ = np.arange(0, 0.5, 0.01)
for i, th in enumerate(range_):

    for ii, model in enumerate(["jr", "jrd"]):

        df_temp = df_avg.loc[(df_avg["model"] == model) & (df_avg["rPLV"] > th)]

        fig.add_trace(go.Scatter3d(x=df_temp["sigma"], y=df_temp["g"], z=df_temp["p"], showlegend=False, mode="markers",
                                   marker=dict(size=df_temp["size"], color=df_temp["rPLV"], cmax=0.5, cmid=0, cmin=-0.5,
                                               colorscale="RdBu", reversescale=True, opacity=1, line=dict(width=0.01, color="black")),
                                   customdata=df_temp["rPLV"], hovertemplate=template), row=1, col=1+ii)


    step = dict(method="update", args=[{"visible": [False] * len(range_) * 2}], label=str(round(th, 2)))
              # {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    step["args"][0]["visible"][i*2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i*2+1] = True  # Toggle i'th trace to "visible"

    steps.append(step)


camera = dict(eye=dict(x=2.25, y=0.9, z=0.6))

sliders = [dict(active=25, currentvalue={"prefix": "rPLV threshold: "}, steps=steps, len=0.7)]
fig.update_layout(sliders=sliders, scene=dict(camera=camera,
                                              xaxis=dict(range=[0.05, 0], title="sigma"),
                                              yaxis=dict(range=[150, 0], title="g"),
                                              zaxis=dict(range=[0, 0.5], title="p")),
                  scene2=dict(camera=camera, xaxis=dict(range=[0.05, 0], title="sigma"),
                              yaxis=dict(range=[150, 0], title="g"),
                              zaxis=dict(range=[0, 0.5], title="p")), template="plotly_white")
pio.write_html(fig, file=folder + '5_space_BigNet_g-p-sigma_'+version+'.html', auto_open=True)





## SPACE 1 : ParamSpace 3D g - p - SPEED

space = "g-p-s_bigNet_plv"
version = ""
print("Working on %s" % space)

file = folder + space + "_results"+version+".pkl"
with open(file, "rb") as f:
    results = pickle.load(f)
    f.close()


##### SIMPLE slider
import plotly.graph_objects as go
import numpy as np

df_avg = results.groupby(["model", "g", "p", "speed"]).mean().reset_index()
df_avg["size"] = np.abs(df_avg["rPLV"] * 25)

# Create figure
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_titles=["JR", "JRD"])

template = "g (coupling factor): %{x} <br>p (input): %{y} <br>sigma (noise std): %{z} <br> rPLV: %{customdata}"

# Add traces, one for each slider step
steps = []
range_ = np.arange(0, 0.5, 0.01)
for i, th in enumerate(range_):

    for ii, model in enumerate(["jr", "jrd"]):

        df_temp = df_avg.loc[(df_avg["model"] == model) & (df_avg["rPLV"] > th)]

        fig.add_trace(go.Scatter3d(x=df_temp["g"], y=df_temp["p"], z=df_temp["speed"], showlegend=False, mode="markers",
                                   marker=dict(size=df_temp["size"], color=df_temp["rPLV"], cmax=0.5, cmid=0, cmin=-0.5,
                                               colorscale="RdBu", reversescale=True, opacity=1, line=dict(width=0.01, color="black")),
                                   customdata=df_temp["rPLV"], hovertemplate=template), row=1, col=1+ii)


    step = dict(method="update", args=[{"visible": [False] * len(range_) * 2}], label=str(round(th, 2)))
              # {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    step["args"][0]["visible"][i*2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i*2+1] = True  # Toggle i'th trace to "visible"

    steps.append(step)


sliders = [dict(active=25, currentvalue={"prefix": "rPLV threshold: "}, steps=steps, len=0.7)]
fig.update_layout(sliders=sliders, scene=dict(xaxis=dict(range=[150, 0]), yaxis=dict(range=[0.5, 0]), zaxis=dict(range=[0, 20]),
                                              xaxis_title='g', yaxis_title='p', zaxis_title='speed'),
                  scene2=dict(xaxis=dict(range=[150, 0]), yaxis=dict(range=[0.5, 0]), zaxis=dict(range=[0, 20]),
                              xaxis_title='g', yaxis_title='p', zaxis_title='speed'))
pio.write_html(fig, file=folder + '5_space_BigNet_g-p-speed_'+version+'.html', auto_open=True)



# ^ Done before 11/04/2022


#######   17/04/2022         SINGLE SPACES   for JRD with sigmas discrete--
## 3d Phase Portraits
fig = make_subplots(rows=1, cols=4, shared_yaxes=True, column_titles=["sigma0_single", "sigma01_single", "sigma02_single", "sigma04_single"],
                    horizontal_spacing=0, vertical_spacing=0,
                    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]])

# Load sigma discrete data
folder = "E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\PSE\PSEmpi_dynSys_singles_jrd_sigmas-m04d17y2022-t14h.32m.45s\\"
file = folder + "p-sigmadiscrete_single_results.pkl"
with open(file, "rb") as f:
    results = pickle.load(f)
    f.close()


for i, sigma in enumerate(set(results["sigma"].values)):

    print("Working on sigma %s" % sigma)

    df = results.loc[results["sigma"] == sigma]

    # - Bifurcation
    for series in df.iterrows():
        if series[1]["psp_t"][0].nbytes < 20:
            fig.add_trace(go.Scatter3d(x=[series[1]["p"]],
                                       y=[series[1]["psp_dt"][0]], z=[series[1]["psp_t"][0]]), row=1, col=i + 1)
        else:
            fig.add_trace(go.Scatter3d(x=[series[1]["p"]] * len(series[1]["psp_t"][0]),
                                       y=series[1]["psp_dt"][0], z=series[1]["psp_t"][0]), row=1, col=i + 1)

    # # - FFTs
    # # Here, average FFT repetitions
    # for param_ in set(df[space_param]):
    #     temp, fft = df.loc[df[space_param] == param_], []
    #     for series in temp.iterrows():
    #         fft.append(series[1]["fft"][0])
    #
    #     fft = np.average(np.asarray(fft), axis=0)
    #     fig.add_trace(go.Scatter3d(x=[param_] * len(series[1]["fft"][0]), y=freqs, z=fft), row=2, col=i + 1)

# legend: group by param (i.e. "p), name param value (or range); x,y,z labels; initial camera perspective
fig.update_traces(marker_size=2)
fig.update_layout(title="JRD with discrete sigmas bifurcations")
pio.write_html(fig, file=folder + '_JRD_sigmas-discrete_SINGLE.html', auto_open=True)








# ################     3D ParamSpace bigNet px
#
# fig = px.scatter_3d(df_avg, x="g", y="p", z="sigma", color="rPLV", size="size", opacity=1,
#                     color_continuous_scale=px.colors.sequential.RdBu_r, range_color=[-0.5, 0.5])
# fig.show(renderer="browser")



# for mode in set(results["mode"]):
#
#     fig = make_subplots(rows=2, cols=3, specs=[[{}, {}, {"type": "scene"}], [{}, {}, {"type": "scene"}]],
#                         column_titles=["rPLV", "bifurcation", "fft"], row_titles=["g", "sigma"])
#
#     ## g
#     temp = results.loc[(results["mode"] == mode) & (results["sigma"] == 0.022)].sort_values(by=["g"])
#
#     fig.add_trace(go.Scatter(x=temp.g, y=temp.rPLV), row=1, col=1)
#
#     fig.add_trace(go.Scatter(x=temp.g, y=temp.max_mV), row=1, col=2)
#     fig.add_trace(go.Scatter(x=temp.g, y=temp.min_mV), row=1, col=2)
#
#
#     for i, g in enumerate(temp.g.values):
#         x = list(temp.freqs.values[i])
#         y = [g]*len(x)
#         z = list(temp.fft.values[i])
#         fig.add_trace(go.Scatter3d(x=x, y=y, z=z), row=1, col=3)
#
#     ## sigma
#     temp = results.loc[(results["mode"] == mode) & (results["g"] == 60)].sort_values(by=["sigma"])
#
#     fig.add_trace(go.Scatter(x=temp.sigma, y=temp.rPLV), row=2, col=1)
#
#     fig.add_trace(go.Scatter(x=temp.sigma, y=temp.max_mV), row=2, col=2)
#     fig.add_trace(go.Scatter(x=temp.sigma, y=temp.min_mV), row=2, col=2)
#
#     for i, g in enumerate(temp.sigma.values):
#         x = list(temp.freqs.values[i])
#         y = [g]*len(x)
#         z = list(temp.fft.values[i])
#         fig.add_trace(go.Scatter3d(x=x, y=y, z=z), row=2, col=3)
#
#     # camera = dict(
#     #     up=dict(x=0, y=0, z=1),
#     #     center=dict(x=0, y=0, z=0),
#     #     eye=dict(x=-0.25, y=1.25, z=1.25)
#     # )
#
#     fig.update_layout(title=mode, xaxis=dict(title="g"), xaxis2=dict(title="g"), xaxis3=dict(title="freq"),
#                       xaxis4=dict(title="sigma"), xaxis5=dict(title="sigma"), xaxis6=dict(title="freq"),
#                       yaxis1=dict(title="correlation"), yaxis2=dict(title="mV"), yaxis3=dict(title="g"),
#                       yaxis4=dict(title="correlation"), yaxis5=dict(title="mV"), yaxis6=dict(title="sigma"))
#
#     fig.show(renderer="browser")
#
#
#
#
#
#
#
# #
# # def WPplot(df, z=None, title=None, type="linear", folder="figures", auto_open="True"):
# #
# #     fig_fc = make_subplots(rows=1, cols=6, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma", "Power"),
# #                         specs=[[{}, {}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
# #                         x_title="Conduction speed (m/s)", y_title="Coupling factor")
# #
# #     df_sub = df.loc[df["band"]=="1-delta"]
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
# #                              reversescale=True, zmin=-z, zmax=z), row=1, col=1)
# #
# #     df_sub = df.loc[df["band"] == "2-theta"]
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
# #                               showscale=False), row=1, col=2)
# #     df_sub = df.loc[df["band"] == "3-alpha"]
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
# #                               showscale=False), row=1, col=3)
# #     df_sub = df.loc[df["band"] == "4-beta"]
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
# #                               showscale=False), row=1, col=4)
# #     df_sub = df.loc[df["band"] == "5-gamma"]
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
# #                               showscale=False), row=1, col=5)
# #
# #     fig_fc.add_trace(go.Heatmap(z=df_sub.bModule, x=df_sub.speed, y=df_sub.G, colorscale='Viridis',
# #                              reversescale=True), row=1, col=6)
# #
# #     fig_fc.update_layout(yaxis1_type=type,yaxis2_type=type,yaxis3_type=type,yaxis4_type=type,yaxis5_type=type,
# #         title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
# #     pio.write_html(fig_fc, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)
# #
# #
# # simulations_tag = "mpi31Jan22_9modes_thcer"  # Tag cluster job
# #
# # # Define PSE folder
# # main_folder = 'E:\\LCCN_Local\PycharmProjects\\thalamusSync\mpi_processing\PSE\\' + simulations_tag
# # if os.path.isdir(main_folder) == False:
# #     os.mkdir(main_folder)
# #
# # # Load the data
# # df = pd.read_csv(main_folder + "/results.csv")
# #
# #
# # # Loop over Modes and Subjects
# # modes = ["jr_pTh",       "jr",       "jr_woTh",
# #          "jr_pTh_wCer",  "jr_wCer",  "jr_woTh_wCer",
# #          "jr_pTh_woCer", "jr_woCer", "jr_woTh_woCer"]
# #
# # for mode in modes:
# #     df_temp = df.loc[df["Mode"] == mode]
# #     df_temp = df_temp.groupby(["G", "speed"]).mean().reset_index()
# #     (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV
# #
# #     specific_folder = main_folder + "\\PSE_allWPs-AVGg" + str(g) + "s" + str(s) + "_" + mode + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
# #     os.mkdir(specific_folder)
# #
# #     for subj in list(set(df.Subject)):
# #
# #         # subset data per mode and subject
# #         df_temp = df.loc[(df["Subject"] == subj) & (df["Mode"] == mode)]
# #
# #         # Avg repetitions
# #         df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
# #         df_temp.drop("rep", inplace=True, axis=1)
# #
# #         # Calculate WP
# #         (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV
# #
# #         name = subj + "_" + mode + "-g" + str(g) + "s" + str(s)
# #
# #         # save data
# #         df_temp.to_csv(specific_folder + "/" + name +"-3reps.csv")
# #
# #         # plot paramspace
# #         WPplot(df_temp, z=0.5, title=name, type="linear", folder=specific_folder, auto_open=False)
# #
# #
# # # Plot 3 by 3 Alpha PSEs
# # for subj in list(set(df.Subject)):
# #
# #     fig_thcer = make_subplots(rows=3, cols=3, column_titles=("Parcelled Thalamus", "Single node Thalamus", "Without Thalamus"),
# #                            row_titles=("Parcelled Cerebellum", "Single node Cerebellum", "Without Cerebellum"),
# #                         specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
# #                         x_title="Conduction speed (m/s)", y_title="Coupling factor")
# #
# #     df_sub = df.loc[(df["band"] == "3-alpha") & (df["Subject"] == subj)]
# #
# #     for i, mode in enumerate(modes):
# #
# #         df_temp = df_sub.loc[df_sub["Mode"] == mode]
# #
# #         df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
# #         df_temp.drop("rep", inplace=True, axis=1)
# #
# #         fig_thcer.add_trace(go.Heatmap(z=df_temp.rPLV, x=df_temp.speed, y=df_temp.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
# #                              reversescale=True, zmin=-0.5, zmax=0.5), row=(i+3)//3, col=i%3+1)
# #
# #
# #     fig_thcer.update_layout(
# #         title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % subj)
# #     pio.write_html(fig_thcer, file=main_folder + "/ThCer_paramSpace-g&s_%s.html" % subj, auto_open=True)
# #
