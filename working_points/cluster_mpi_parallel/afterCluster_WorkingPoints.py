
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots


def WPplot(df, z=None, title=None, type="linear", folder="figures", auto_open="True"):

    if "g&s" in title:
        xtitle="Conduction speed (m/s)"
    elif "g&p" in title:
        xtitle="p (input)"

    fig_fc = make_subplots(rows=1, cols=6, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma", "Power"),
                        specs=[[{}, {}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title=xtitle, y_title="Coupling factor")

    df_sub = df.loc[df["band"]=="1-delta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.v2, y=df_sub.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                             reversescale=True, zmin=-z, zmax=z), row=1, col=1)

    df_sub = df.loc[df["band"] == "2-theta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.v2, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=2)
    df_sub = df.loc[df["band"] == "3-alpha"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.v2, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=3)
    df_sub = df.loc[df["band"] == "4-beta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.v2, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=4)
    df_sub = df.loc[df["band"] == "5-gamma"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.v2, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=5)

    fig_fc.add_trace(go.Heatmap(z=df_sub.bModule, x=df_sub.v2, y=df_sub.G, colorscale='Viridis',
                             reversescale=True), row=1, col=6)

    fig_fc.update_layout(yaxis1_type=type, yaxis2_type=type, yaxis3_type=type, yaxis4_type=type, yaxis5_type=type,
                         title_text='FC correlation (empirical - simulated data) || %s' % title)
    pio.write_html(fig_fc, file=folder + "/paramSpace_%s.html" % title, auto_open=auto_open)


simulations_tag = "mpi17June22_jr_gianluca"  # Tag cluster job

# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\brainModels\working_points\cluster_mpi_parallel\PSE\\' + simulations_tag
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

# Load the data
df = pd.read_csv(main_folder + "/results.csv")
n_reps = df["rep"].max() + 1

# Loop over Modes and Subjects
#for mode in ["jr", "jrd_def", "jrd",  "jrd_pTh", "cb", "jrdcb"]:
for mode in ["jr_gianluca"]:
    for test_params in list(set(df.test_params)):
        for out in list(set(df.out)):

            df_temp = df.loc[(df["Mode"] == mode) & (df["out"] == out) & (df["test_params"] == test_params)]

            df_temp = df_temp.groupby(["G", "v2"]).mean().reset_index()
            (g, v2) = df_temp.groupby(["G", "v2"]).mean().idxmax(axis=0).rPLV

            specific_folder = main_folder + "\\PSE_allWPs-AVGg" + str(g) + "s" + str(v2) + "_" + mode + "_" + out + "_" + test_params + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
            os.mkdir(specific_folder)

            for subj in list(set(df.Subject)):

                # subset data per mode and subject
                df_temp = df.loc[(df["Subject"] == subj) & (df["Mode"] == mode) & (df["out"] == out) & (df["test_params"] == test_params)]

                # Avg repetitions
                df_temp = df_temp.groupby(["G", "v2", "band"]).mean().reset_index()
                df_temp.drop("rep", inplace=True, axis=1)

                # Calculate WP
                (g, v2) = df_temp.groupby(["G", "v2"]).mean().nlargest(10, 'rPLV').idxmax(axis=0).bModule
                # (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV

                name = subj + "_" + mode + "_" + out + "_" + test_params + "-g" + str(g) + "s" + str(v2)

                # save data
                df_temp.to_csv(specific_folder + "/" + name + "-" + str(n_reps) +"reps.csv")

                # plot paramspace
                WPplot(df_temp, z=0.5, title=name, type="linear", folder=specific_folder, auto_open=False)
