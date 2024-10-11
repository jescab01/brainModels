
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\IntegrativeRhythms\SurfSim\\bifurcations\PSE\\'
simulations_tag = "PSEmpi_surf_pse-m02d15y2024-t21h.25m.19s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")


## PLOT
taue_vals = sorted(set(df.tau_e))
taui_vals = sorted(set(df.tau_i), reverse=True)

size, op = 2, 0.6

fig = make_subplots(rows=len(taue_vals), cols=len(taui_vals), shared_yaxes=True,
                    row_titles=["taue" + str(round(tau)) for tau in taue_vals],
                    column_titles=["taui" + str(round(tau)) for tau in taui_vals])

for i, taue in enumerate(taue_vals):
    for j, taui in enumerate(taui_vals):

        sl = True if i+j == 0 else False

        subset = df.loc[(df["tau_e"] == taue) & (df["tau_i"] == taui)]
        # Bifurcation data
        fig.add_trace(go.Scatter(x=subset.p, y=subset.maxPSP, mode="markers", marker=dict(color="indianred", size=size, opacity=op), name="max",legendgroup="max",showlegend=sl), row=i+1, col=j+1)
        fig.add_trace(go.Scatter(x=subset.p, y=subset.minPSP, mode="markers", marker=dict(color="steelblue", size=size, opacity=op), name="min",legendgroup="min", showlegend=sl), row=i+1, col=j+1)

        # stimulation data (maxs)
        fig.add_trace(go.Scatter(x=subset.p, y=subset.maxt1, mode="markers", marker=dict(color="yellow", size=size, opacity=op), name="t1",legendgroup="t1", showlegend=sl), row=i+1, col=j+1)
        fig.add_trace(go.Scatter(x=subset.p, y=subset.maxt2, mode="markers", marker=dict(color="orange", size=size, opacity=op), name="t2",legendgroup="t2", showlegend=sl), row=i+1, col=j+1)
        fig.add_trace(go.Scatter(x=subset.p, y=subset.maxt3, mode="markers", marker=dict(color="red", size=size, opacity=op), name="t3",legendgroup="t3", showlegend=sl), row=i+1, col=j+1)


fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.1, x=0.05))
pio.write_html(fig, file=main_folder + simulations_tag + "/bifurcations.html", auto_open=True)

