
import os
import pandas as pd
import time
import numpy as np
import pickle

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots


# Define PSE folder
folder = 'E:\LCCN_Local\PycharmProjects\\brainModels\CORTICON\PSE\PSEmpi_dynSys_bigNet_plv_WP-m04d01y2022-t20h.19m.37s\\'
space = "g-speed_bigNet_plv"  # Tag cluster job


# Load the data
file = folder + space + "_results.pkl"
with open(file, "rb") as f:
    results = pickle.load(f)
    f.close()

n_reps = max(results.rep) + 1

## calculate freqs With simtime
lowcut, highcut = 0, 60

freqs = np.arange(results["simtime"].iloc[0] / 2)
freqs = freqs / ((results["simtime"].iloc[0]) / 1000)  # simLength (ms) / 1000 -> segs

freqs = freqs[(freqs > lowcut) & (freqs < highcut)]  # remove undesired frequencies

# Loop over Modes and Subjects
wp = []
for model in ["jr_def", "jrd_def", "jr_mix", "jrd_mix"]:
    for subj in list(set(results.subject)):

        # subset data per mode and subject
        df_temp = results.loc[(results["model"] == model) & (results["subject"] == subj)]

        # Avg repetitions
        df_temp = df_temp.groupby(["g", "speed"]).mean().reset_index()
        df_temp.drop("rep", inplace=True, axis=1)

        # Calculate WP
        (g, s) = df_temp.groupby(["g", "speed"]).mean().idxmax(axis=0).rPLV
        max_r_plv = df_temp["rPLV"].loc[(df_temp["g"] == g) & (df_temp["speed"]==s)].values[0]

        name = subj + "_" + model + "-g" + str(g) + "s" + str(s)

        wp.append([model, subj, max_r_plv, g, s])

        ### Average and Plot FFTs for best point
        ffts = results["fft"].loc[
            (results["model"] == model) & (results["subject"] == subj) & (results["g"] == g) & (results["speed"] == s)].values

        th_fft = np.average(np.asarray([pair_fft[0] for pair_fft in ffts]), axis=0)
        cx_fft = np.average(np.asarray([pair_fft[1] for pair_fft in ffts]), axis=0)

        fig_fc = make_subplots(rows=1, cols=2, subplot_titles=("rPLV - Alpha band", "FFTs @ g"+str(g)+" s"+str(s)),
                               specs=[[{}, {}]])
                               # x_title="Conduction speed (m/s)", y_title="Coupling factor")

        fig_fc.add_trace(go.Heatmap(z=df_temp.rPLV, x=df_temp.speed, y=df_temp.g, colorscale='RdBu',
                                    colorbar=dict(title="Pearson's r"),
                                    reversescale=True, zmin=-0.5, zmax=0.5), row=1, col=1)

        fig_fc.add_trace(go.Scatter(x=freqs, y=th_fft, name="Thalamus"), row=1, col=2)
        fig_fc.add_trace(go.Scatter(x=freqs, y=cx_fft, name="Cortex"), row=1, col=2)

        fig_fc.update_layout(title_text='FC correlation (empirical - simulated data) || %s' % name)
        pio.write_html(fig_fc, file=folder + "/paramSpace_%s.html" % name, auto_open=False)

wp_df = pd.DataFrame(wp, columns=["model", "subj", "max_r_plv", "g", "speed"])
wp_df.to_csv(folder + "dynSys-Working_Points.csv")
