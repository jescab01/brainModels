

import pandas as pd

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


## 0. Data, folders and definitions

main_folder = 'E:\\LCCN_Local\PycharmProjects\\BrainRhythms\LocalConn\PSE\\'
simulations_tag = "PSEmpi_localConn_disc8k-m04d10y2024-t14h.14m.57s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")


poi = [1474, 462, 752, 494, 1327] if "disc4k" in simulations_tag else \
    [3556, 1594, 252, 951, 2549] if "disc8k" in simulations_tag else \
        [3556] if "disc17k" in simulations_tag else []


fig = make_subplots(rows=4, cols=1, x_title="LC strength", shared_xaxes=True)

cmap = px.colors.qualitative.Plotly

for i, v in enumerate(poi):

    sub = df.loc[(df["vertex"] == v)].dropna()

    fig.add_trace(go.Scatter(x=sub.lc, y=sub.maxSignal, name=v, legendgroup=v, showlegend=True,
                             line=dict(color=cmap[i])), row=1, col=1)

    fig.add_trace(go.Scatter(x=sub.lc, y=sub.time_maxSignal.abs(), name=v, legendgroup=v, showlegend=False,
                             line=dict(color=cmap[i])), row=2, col=1)

    fig.add_trace(go.Scatter(x=sub.lc, y=sub.intraS_std, name=v, legendgroup=v, showlegend=False,
                             line=dict(color=cmap[i])), row=3, col=1)

    fig.add_trace(go.Scatter(x=sub.lc, y=sub.interS_std, name=v, legendgroup=v, showlegend=False,
                             line=dict(color=cmap[i], dash="dash")), row=4, col=1)



fig.update_layout(template="plotly_white",yaxis1=dict(title="max mV"),
                  yaxis2=dict(title="time to max"),
                  yaxis3=dict(title="intraSignal std"), yaxis4=dict(title="interSignal std"))# xaxis1=dict(type="log"), xaxis2=dict(type="log"))

pio.write_html(fig, main_folder + simulations_tag + "\\lineplots.html", auto_open=True)


