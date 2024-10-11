
import os

import pandas as pd
import numpy as np
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\BrainRhythms\\1Hierarchies\\2NetProgressive\PSE\\'
simulations_tag = "PSEmpi_hierarchiesProgressive-m04d13y2024-t22h.47m.45s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")


# Decide what mode to explore
title = "Hierarchical-r436tract-5L"
l = 5

# ff/fb ratios to explore
ratio_vals = [(0.25, 1), (0.5, 1), (0.75, 1), (1, 1), (1, 0.75), (1, 0.5), (1, 0.25)]

g_vals = np.arange(0, 100, 1)

ratio_data = pd.DataFrame(
    np.array(
        [(ff/fb, g, g*ff, g*fb) for g in g_vals for (ff, fb) in ratio_vals]
    ), columns=["ratio", "g", "ff", "fb"])


# Filter out combinations for which you don't have data
ratio_data = ratio_data.loc[(ratio_data["ff"] < 100) & (ratio_data["fb"] < 100)]
ratio_data["ff"] = ratio_data["ff"].astype(int)
ratio_data["fb"] = ratio_data["fb"].astype(int)


# Gather data about bifurcation per combination
ratio_data["Smin"] = [df["Smin"].loc[(df["FF"] == row["ff"]) & (df["FB"] == row["fb"]) & (df["L"] == l)].values[0] for i, row in ratio_data.iterrows()]
ratio_data["Smax"] = [df["Smax"].loc[(df["FF"] == row["ff"]) & (df["FB"] == row["fb"]) & (df["L"] == l)].values[0] for i, row in ratio_data.iterrows()]



cmap = px.colors.qualitative.Plotly
fig = make_subplots(rows=1, cols=2, column_widths=[0.25, 0.75])

sub = df.loc[(df["L"] == 5)]

fig.add_trace(go.Heatmap(z=sub["peaks_post_r0"], x=sub.FF, y=sub.FB, showscale=False,
                         zmin=0, zmax=max(df["peaks_post_r0"])), row=1, col=1)

for i, (ff, fb) in enumerate(ratio_vals):

    ratio = ff/fb

    temp = ratio_data.loc[ratio_data["ratio"] == ratio]

    fig.add_trace(go.Scatter(x=temp.ff, y=temp.fb, mode="markers", showlegend=False,
                             line=dict(color=cmap[i]), legendgroup=str(ratio)
                             ), row=1, col=1)

    fig.add_trace(go.Scatter(x=temp.g, y=temp.Smin, line=dict(color=cmap[i]), name="R" + str(ratio), legendgroup=str(ratio)), row=1, col=2)
    fig.add_trace(go.Scatter(x=temp.g, y=temp.Smax, line=dict(color=cmap[i]), legendgroup=str(ratio), showlegend=False), row=1, col=2)


fig.update_layout(template="plotly_white", height=400, title="Bifurcations per Ratio (ff/fb)",
                  xaxis1=dict(title="FF"), yaxis1=dict(title="FB"),
                  xaxis2=dict(title="bifurcation parameter"), yaxis2=dict(title="mV"),
                  legend=dict(orientation="h", x=0.15, y=-0.25))

pio.write_html(fig, file=main_folder + simulations_tag + "/bifurcations.html", auto_open=True)






