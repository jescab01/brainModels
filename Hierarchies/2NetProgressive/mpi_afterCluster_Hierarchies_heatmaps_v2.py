
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\AlphaOrigins\\1Hierarchies\\2NetProgressive\PSE\\'
simulations_tag = "PSEmpi_hierProgr_toAlpha-m06d04y2024-t21h.21m.26s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")


# Decide what mode to explore
bif_vals = [""]
L_vals = [0]
nrois_vals = [7, 31, 180, 360, 426]



for bif in bif_vals:

    title = "Hierarchical-5rtract-L0_%s" % (bif)

    fig = make_subplots(rows=5, cols=5,
                        row_titles=["Frequency", "Power", "max-min", "intra_std", "inter_std"],
                        column_titles=["nrois == %i" % n for n in nrois_vals])

    for j, n in enumerate(nrois_vals):

        sub = df.loc[(df["n_rois"] == n)] if bif == "pre" else df.loc[(df["n_rois"] == n)]

        sl = True if j == 0 else False


        fig.add_trace(go.Heatmap(z=sub["peak"], x=sub.FF, y=sub.FB, showscale=sl,
                                 zmin=0, zmax=max(df["peak"]),
                                 colorbar=dict(thickness=10, len=0.15, y=0.9, title="Hz")),
                      row=1, col=j + 1)

        fig.add_trace(go.Heatmap(z=sub["band_module"], x=sub.FF, y=sub.FB, colorscale='Viridis', showscale=sl,
                                 zmin=0, zmax=100, #max(df["band_module"]),
                                 colorbar=dict(thickness=10, len=0.15, y=0.70, title="dB")),
                      row=2, col=j + 1)

        fig.add_trace(go.Heatmap(z=sub["Smax"]-sub["Smin"], x=sub.FF, y=sub.FB, colorscale='Inferno', showscale=sl,
                                 zmin=0, zmax=1, #max(df["Smax"]-df["Smin"]),
                                 colorbar=dict(thickness=10, len=0.15, y=0.5, title="mV")),
                      row=3, col=j + 1)

        fig.add_trace(go.Heatmap(z=sub["intraS_std"], x=sub.FF, y=sub.FB, colorscale='Inferno', showscale=sl,
                                 zmin=0, zmax=0.25, #max(df["intraS_std"]),
                                 colorbar=dict(thickness=10, len=0.15, y=0.30, title="std")),
                      row=4, col=j + 1)

        fig.add_trace(go.Heatmap(z=sub["interS_std"], x=sub.FF, y=sub.FB, colorscale='Inferno', showscale=sl,
                                 zmin=0, zmax=0.05, #max(df["interS_std"]),
                                 colorbar=dict(thickness=10, len=0.15, y=0.1, title="std")),
                      row=5, col=j + 1)


    fig.update_layout(title=title, template="plotly_white",
                      yaxis1=dict(title="FB"), yaxis6=dict(title="FB"), yaxis11=dict(title="FB"),
                      yaxis16=dict(title="FB"), yaxis21=dict(title="FB"),
                      xaxis21=dict(title="FF"), xaxis22=dict(title="FF"), xaxis23=dict(title="FF"),
                      xaxis24=dict(title="FF"), xaxis25=dict(title="FF"))

    pio.write_html(fig, file=main_folder + simulations_tag + "/" + title + ".html", auto_open=True)





