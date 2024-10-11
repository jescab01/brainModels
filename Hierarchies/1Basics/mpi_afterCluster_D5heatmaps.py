
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\IntegrativeRhythms\Models\PSE\\'
simulations_tag = "PSEmpi_4hierarchies-FF_FB-m02d29y2024-t19h.37m.55s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

modes = ["r2_g1", "r3lin_g0.5", "r3all_g0.5", "r5_g0.25"]

h, w = 650, 1250

## 1a) PLOTTING ROI A (the lower in the hierarchy recieving the input)

fig = make_subplots(rows=3, cols=8, column_titles=["(r2, g1)<br>Baseline", "Post-stim", "(r3, g0.5)<br>Baseline ", "Post-stim",
                                                   "(r3, g0.5)<br>Baseline ", "Post-stim", "(r5, g0.25)<br>Baseline ", "Post-stim"],
                    row_titles=["Frequency", "Power", "Duration"])


for j, mode in enumerate(modes):

    sub = df.loc[df["mode"] == mode]

    sl = True if j == 0 else False
    fig.add_trace(go.Heatmap(z=sub.peaks_pre_r0, x=sub.FF, y=sub.FB, showscale=sl, zmin=0, zmax=max(df.peaks_post_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.85, title="Hz")), row=1, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.peaks_post_r0, x=sub.FF, y=sub.FB, showscale=False, zmin=0, zmax=max(df.peaks_post_r0)
                             ), row=1, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.band_modules_pre_r0, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=sl,
                             zmin=0, zmax=max(df.band_modules_post_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.5, title="dB")), row=2, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.band_modules_post_r0, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=False,
                             zmin=0, zmax=max(df.band_modules_post_r0)), row=2, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.duration_r0, x=sub.FF, y=sub.FB, colorscale='PuBu', showscale=sl,
                             zmin=0, zmax=max(df.duration_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.15, title="ms")), row=3, col=j*2 + 2)


fig.update_layout(height=h, width=w,
                  title="ROI A", yaxis1=dict(title="FB"), yaxis9=dict(title="FB"), yaxis18=dict(title="FB"),
                  xaxis9=dict(title="FF"), xaxis18=dict(title="FF"), xaxis20=dict(title="FF"),
                  xaxis22=dict(title="FF"), xaxis24=dict(title="FF"), )

pio.write_html(fig, file=main_folder + simulations_tag + "/4Hierarchies_roiA_full.html", auto_open=True)




## 1b) PLOTTING ROI A :: withouth the data from bifurcated simulations

fig = make_subplots(rows=3, cols=8, column_titles=["(r2, g1)<br>Baseline", "Post-stim", "(r3, g0.5)<br>Baseline ", "Post-stim",
                                                   "(r3, g0.5)<br>Baseline ", "Post-stim", "(r5, g0.25)<br>Baseline ", "Post-stim"],
                    row_titles=["Frequency", "Power", "Duration"])



for j, mode in enumerate(modes):

    sub = df.loc[(df["mode"] == mode) & (df["band_modules_pre_r0"] < 100)]

    sl = True if j == 0 else False
    fig.add_trace(go.Heatmap(z=sub.peaks_pre_r0, x=sub.FF, y=sub.FB, showscale=sl, zmin=0, zmax=max(sub.peaks_post_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.85, title="Hz")), row=1, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.peaks_post_r0, x=sub.FF, y=sub.FB, showscale=False, zmin=0, zmax=max(sub.peaks_post_r0)
                             ), row=1, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.band_modules_pre_r0, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=sl,
                             zmin=0, zmax=max(sub.band_modules_post_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.5, title="dB")), row=2, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.band_modules_post_r0, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=False,
                             zmin=0, zmax=max(sub.band_modules_post_r0)), row=2, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.duration_r0, x=sub.FF, y=sub.FB, colorscale='PuBu', showscale=sl,
                             zmin=0, zmax=max(sub.duration_r0),
                             colorbar=dict(thickness=10, len=0.3, y=0.15, title="ms")), row=3, col=j*2 + 2)


# Define a series of 3 points per mode that would be interesting to look-up
poi = [[(45, 40), (40, 40), (35, 35), (25, 20), (15, 15)],
       [(60, 55), (50, 50), (30, 25)],
       [(40, 98), (40, 85), (40, 60), (40, 35)],
       [(98, 98), (85, 85), (60, 50), (50, 40)]]

for j, mode in enumerate(modes):
    for i in range(3):
        fig.add_trace(go.Scatter(x=[x for (x, y) in poi[j]], y=[y for (x, y) in poi[j]],
                                 showlegend=False, mode="markers", marker=dict(symbol="circle-open", color="red")), row=i+1, col=j*2 + 2)


fig.update_layout(height=h, width=w,
                  template="plotly_white", title="ROI A", yaxis1=dict(title="FB"), yaxis9=dict(title="FB"), yaxis18=dict(title="FB"),
                  xaxis9=dict(title="FF"), xaxis18=dict(title="FF"), xaxis20=dict(title="FF"),
                  xaxis22=dict(title="FF"), xaxis24=dict(title="FF"), )

pio.write_html(fig, file=main_folder + simulations_tag + "/4Hierarchies_roiA_prebif.html", auto_open=True)








## 2a) PLOTTING ROI B  (the Higher in the hierarchy)

fig = make_subplots(rows=3, cols=8, column_titles=["(r2, g1)<br>Baseline", "Post-stim", "(r3, g0.5)<br>Baseline ", "Post-stim",
                                                   "(r3, g0.5)<br>Baseline ", "Post-stim", "(r5, g0.25)<br>Baseline ", "Post-stim"],
                    row_titles=["Frequency", "Power", "Duration"])



for j, mode in enumerate(modes):

    sub = df.loc[df["mode"] == mode]

    sl = True if j == 0 else False
    fig.add_trace(go.Heatmap(z=sub.peaks_pre_r1, x=sub.FF, y=sub.FB, showscale=sl, zmin=0, zmax=max(df.peaks_post_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.85, title="Hz")), row=1, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.peaks_post_r1, x=sub.FF, y=sub.FB, showscale=False, zmin=0, zmax=max(df.peaks_post_r1)
                             ), row=1, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.band_modules_pre_r1, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=sl,
                             zmin=0, zmax=max(df.band_modules_post_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.5, title="dB")), row=2, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.band_modules_post_r1, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=False,
                             zmin=0, zmax=max(df.band_modules_post_r1)), row=2, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.duration_r1, x=sub.FF, y=sub.FB, colorscale='PuBu', showscale=sl,
                             zmin=0, zmax=max(df.duration_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.15, title="ms")), row=3, col=j*2 + 2)


fig.update_layout(height=h, width=w,
                  title="ROI B", yaxis1=dict(title="FB"), yaxis9=dict(title="FB"), yaxis18=dict(title="FB"),
                  xaxis9=dict(title="FF"), xaxis18=dict(title="FF"), xaxis20=dict(title="FF"),
                  xaxis22=dict(title="FF"), xaxis24=dict(title="FF"), )

pio.write_html(fig, file=main_folder + simulations_tag + "/4Hierarchies_roiB_full.html", auto_open=True)


## 2b) PLOTTING ROI B :: without the bifurcated data

fig = make_subplots(rows=3, cols=8, column_titles=["(r2, g1)<br>Baseline", "Post-stim", "(r3, g0.5)<br>Baseline ", "Post-stim",
                                                   "(r3, g0.5)<br>Baseline ", "Post-stim", "(r5, g0.25)<br>Baseline ", "Post-stim"],
                    row_titles=["Frequency", "Power", "Duration"])



for j, mode in enumerate(modes):

    sub = df.loc[(df["mode"] == mode) & (df["band_modules_pre_r1"] < 100)]

    sl = True if j == 0 else False
    fig.add_trace(go.Heatmap(z=sub.peaks_pre_r1, x=sub.FF, y=sub.FB, showscale=sl, zmin=0, zmax=max(sub.peaks_post_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.85, title="Hz")), row=1, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.peaks_post_r1, x=sub.FF, y=sub.FB, showscale=False, zmin=0, zmax=max(sub.peaks_post_r1)
                             ), row=1, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.band_modules_pre_r1, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=sl,
                             zmin=0, zmax=max(sub.band_modules_post_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.5, title="dB")), row=2, col=j*2 + 1)

    fig.add_trace(go.Heatmap(z=sub.band_modules_post_r1, x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=False,
                             zmin=0, zmax=max(sub.band_modules_post_r1)), row=2, col=j*2 + 2)

    fig.add_trace(go.Heatmap(z=sub.duration_r1, x=sub.FF, y=sub.FB, colorscale='PuBu', showscale=sl,
                             zmin=0, zmax=max(sub.duration_r1),
                             colorbar=dict(thickness=10, len=0.3, y=0.15, title="ms")), row=3, col=j*2 + 2)


fig.update_layout(height=h, width=w,
                  template="plotly_white", title="ROI B", yaxis1=dict(title="FB"), yaxis9=dict(title="FB"), yaxis18=dict(title="FB"),
                  xaxis9=dict(title="FF"), xaxis18=dict(title="FF"), xaxis20=dict(title="FF"),
                  xaxis22=dict(title="FF"), xaxis24=dict(title="FF"), )

pio.write_html(fig, file=main_folder + simulations_tag + "/4Hierarchies_roiB_prebif.html", auto_open=True)







