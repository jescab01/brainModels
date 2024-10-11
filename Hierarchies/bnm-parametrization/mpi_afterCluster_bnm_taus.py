
import os

import pandas as pd
import numpy as np
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\AlphaOrigins\\2bnm-parametrization\\PSE\\'
simulations_tag = "PSEmpi_bnm-params-m06d05y2024-t19h.28m.28s"

df = pd.read_csv(main_folder + simulations_tag + "/results.csv")



def_vals = [("He", 3.25), ("Hi", 22), ("taue", 10), ("taui", 20), ("speed", 3.9)]
mode, ff, fb = df["mode"][0], 0.65, 0.35  # defines the mode

title = "modes-r426tract-taus"

modes = ["hierarchical", "classical"]

fig = make_subplots(rows=5, cols=2, y_title="Coupling factor (g)",
                    row_titles=["Frequency", "Power", "max-min", "intra_std", "inter_std"],
                    column_titles=modes)

for j, mode in enumerate(modes):

    sl = True if j == 0 else False

    sub = df.loc[df["mode"]==mode]

    fig.add_trace(go.Heatmap(z=sub["freq_peaks"], x=sub.taue, y=sub.taui, showscale=sl,
                             zmin=0, zmax=max(df["freq_peaks"]),
                             colorbar=dict(thickness=10, len=0.15, y=0.9, title="Hz")),
                  row=1, col=j + 1)

    fig.add_trace(go.Heatmap(z=sub["band_modules"], x=sub.taue, y=sub.taui, colorscale='Viridis', showscale=sl,
                             zmin=0, zmax=max(df["band_modules"]),
                             colorbar=dict(thickness=10, len=0.15, y=0.70, title="dB")),
                  row=2, col=j + 1)

    fig.add_trace(go.Heatmap(z=sub["Smax"] - sub["Smin"], x=sub.taue, y=sub.taui, colorscale='Inferno', showscale=sl,
                             zmin=0, zmax= max(df["Smax"]-df["Smin"]),
                             colorbar=dict(thickness=10, len=0.15, y=0.5, title="mV")),
                  row=3, col=j + 1)

    fig.add_trace(go.Heatmap(z=sub["intraS_std"], x=sub.taue, y=sub.taui, colorscale='Inferno', showscale=sl,
                             zmin=0, zmax=max(df["intraS_std"]),
                             colorbar=dict(thickness=10, len=0.15, y=0.30, title="std")),
                  row=4, col=j + 1)

    fig.add_trace(go.Heatmap(z=sub["interS_std"], x=sub.taue, y=sub.taui, colorscale='Inferno', showscale=sl,
                             zmin=0, zmax=max(df["interS_std"]),
                             colorbar=dict(thickness=10, len=0.15, y=0.1, title="std")),
                  row=5, col=j + 1)

    # fig.add_trace(go.Scatter(x=10, y=20), row=1, col=j+1)


fig.update_layout(title=title, template="plotly_white",
                  # yaxis1=dict(title="Coupling factor"), yaxis6=dict(title="Coupling factor"),
                  # yaxis11=dict(title="Coupling factor"), yaxis16=dict(title="Coupling factor ("), yaxis21=dict(title="FB"),
                  xaxis21=dict(title="He"), xaxis22=dict(title="Hi"), xaxis23=dict(title="taue"),
                  xaxis24=dict(title="taui"), xaxis25=dict(title="speed"))

pio.write_html(fig, file=main_folder + simulations_tag + "/" + title + ".html", auto_open=True)










#
#
# # Define PSE folder
# main_folder = 'E:\\LCCN_Local\PycharmProjects\\AlphaOrigins\\2bnm-parametrization\\PSE\\'
# simulations_tag = "PSEmpi_bnm-params-m04d19y2024-t12h.14m.20s"  # Tag cluster job
# df = pd.read_csv(main_folder + simulations_tag + "/results.csv")
#
#
# ratio_range = np.arange(0.30, 0.71, 0.1)
# ratio_vals = [(round(val, 2), round(1 - val, 2)) for val in ratio_range]
#
# modes = ["classical", "classical-reparam", "classical-reparam-cs"]
#
# sigma_vals = [0, 2.2e-4, 2.2e-3, 2.2e-2]  #, 2.2e-1, 2.2e-2, 2.2e-3, 2.2e-4]
#
# rois = list(set(df.roi))
#
#
# # Color-code Frequencies
# df["hover"] = ["Frequency peak - %0.2f Hz" % (row.freq_peaks) for i, row in df.iterrows()]
#
# ## 1a) BIFURCATIONS
# fig = make_subplots(rows=len(ratio_vals) + len(modes), cols=len(sigma_vals), shared_xaxes=True, shared_yaxes=True,
#                     column_titles=["sigma==%0.1e" % vals for vals in sigma_vals],
#                     row_titles=["%0.2f/%0.2f" % vals for vals in ratio_vals] + modes)
#
# cmap = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
# for j, sigma in enumerate(sigma_vals):
#     for c, roi in enumerate(rois):
#         for i, (ff, fb) in enumerate(ratio_vals):
#
#             sub = df.loc[(df["FF"] == ff) & (df["FB"] == fb) & (df["sigma"] == sigma) & (df["roi"] == roi)]
#
#             sl = True if (j == 0) & (i == 0) else False
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smin, name=roi, legendgroup=roi, showlegend=sl,
#                                      line=dict(color=cmap[c])), row=i+1, col=j+1)
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smax, name=roi, legendgroup=roi, showlegend=False,
#                                      line=dict(color=cmap[c])), row=i+1, col=j+1)
#
#         for i, mode in enumerate(modes):
#
#             sub = df.loc[(df["mode"] == mode) & (df["sigma"] == sigma) & (df["roi"] == roi)]
#
#             sl = True if (j == 0) & (i == 0) else False
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smin, name=roi, legendgroup=roi, showlegend=False,
#                                      line=dict(color=cmap[c])), row=len(ratio_vals) + i + 1, col=j + 1)
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smax, name=roi, legendgroup=roi, showlegend=False,
#                                      line=dict(color=cmap[c])), row=len(ratio_vals) + i + 1, col=j + 1)
#
# fig.update_layout(template="plotly_white", title="Bifurcations per ratio (ff/fb)",
#                   legend=dict(orientation="h"), height=1000)
#
# pio.write_html(fig, file=main_folder + simulations_tag + "/bnm-bifurcations_g-sigma-roi.html", auto_open=True)
#
#
#
# ## 1a) FREQUENCY
#
# # Color code the frequency
# fig = make_subplots(rows=len(ratio_vals) + len(modes), cols=len(sigma_vals), shared_xaxes=True, shared_yaxes=True,
#                     column_titles=["sigma==%0.1e" % vals for vals in sigma_vals],
#                     row_titles=["%0.2f/%0.2f" % vals for vals in ratio_vals] + modes)
#
# size, op = 5, 0.7
#
# for j, sigma in enumerate(sigma_vals):
#     for c, roi in enumerate(rois):
#         for i, (ff, fb) in enumerate(ratio_vals):
#
#             sub = df.loc[(df["FF"] == ff) & (df["FB"] == fb) & (df["sigma"] == sigma) & (df["roi"] == roi)]
#
#             sl = True if (j == 0) & (i == 0) else False
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smin, name=roi, legendgroup=roi, showlegend=sl, mode="markers",
#                                      hovertext=sub.hover, marker=dict(color=sub.freq_peaks, size=size, opacity=op,
#                                                  colorbar=dict(thickness=10), cmin=0, cmax=12)),
#                           row=i+1, col=j+1)
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smax, name=roi, legendgroup=roi, showlegend=False, mode="markers",
#                                      hovertext=sub.hover, marker=dict(color=sub.freq_peaks, size=size, opacity=op, cmin=0, cmax=12)),
#                           row=i+1, col=j+1)
#
#         for i, mode in enumerate(modes):
#
#             sub = df.loc[(df["mode"] == mode) & (df["sigma"] == sigma) & (df["roi"] == roi)]
#
#             sl = True if (j == 0) & (i == 0) else False
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smin, name=roi, legendgroup=roi, showlegend=False, mode="markers",
#                                      hovertext=sub.hover, marker=dict(color=sub.freq_peaks, size=size, opacity=op,
#                                                  cmin=0, cmax=12)),
#                           row=len(ratio_vals) + i + 1, col=j + 1)
#
#             fig.add_trace(go.Scatter(x=sub.g, y=sub.Smax, name=roi, legendgroup=roi, showlegend=False, mode="markers",
#                                      hovertext=sub.hover,
#                                      marker=dict(color=sub.freq_peaks, size=size, opacity=op, cmin=0, cmax=12)),
#                           row=len(ratio_vals) + i + 1, col=j + 1)
#
# fig.update_layout(template="plotly_white", title="Bifurcations per ratio (ff/fb)",
#                   legend=dict(orientation="h"), height=1000)
#
# pio.write_html(fig, file=main_folder + simulations_tag + "/bnm-bif-freq_g-sigma-roi.html", auto_open=True)
