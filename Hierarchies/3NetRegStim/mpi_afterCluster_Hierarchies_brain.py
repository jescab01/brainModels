
import os
import time
import pandas as pd
import numpy as np
from tvb.simulator.lab import cortex, connectivity

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


## 0. Data, folders and definitions

main_folder = 'E:\\LCCN_Local\PycharmProjects\\BrainRhythms\\1Hierarchies\\3NetRegStim\PSE\\'
simulations_tag = "PSEmpi_hierarchiesBrain-m04d15y2024-t14h.24m.20s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")


data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\.Data\\"
surfpack = "HCPex-r426-surfdisc4k_pack\\"

conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")

cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                             region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                             local_connectivity_file=None)  # cutoff=40mm (gaussian)


n_rois = 426
hier_vals = [(90, 50), (60, 60), (50, 90),  # L = 1/2 * min (FF, FB)
             (75, 20), (40, 40), (20, 75),
             (65, 15), (25, 25), (10, 70),
             (18, 18)]

params = [(ff, fb, 5) for ff, fb in hier_vals]


## 1. Plot in 3D
vertices, triangles = cx.surface.vertices, cx.surface.triangles

# associate each vertex with a value using region mapping
rm = cx.region_mapping_data.array_data
v_set = set(rm)

r_ids = [i for i, centre in enumerate(conn.centres) if i not in v_set]
r_centres = conn.centres[r_ids]


for ff, fb, l in params:

    title = "FF%iFB%iL%i" % (ff, fb, l)

    ## 1.1 Prepare data to plot
    sub = df.loc[(df["FF"] == ff) & (df["FB"] == fb) & (df["L"] == l)]

    # plot frequency
    v_peaks_pre = [sub.peaks_pre_r0.loc[sub["r0"] == roi].values[0] for roi in rm]
    v_peaks_post = [sub.peaks_post_r0.loc[sub["r0"] == roi].values[0] for roi in rm]

    r_peaks_pre = [sub.peaks_pre_r0.loc[sub["r0"] == roi].values[0] for roi in r_ids]
    r_peaks_post = [sub.peaks_post_r0.loc[sub["r0"] == roi].values[0] for roi in r_ids]

    # plot power
    v_pow_pre = [sub.band_modules_pre_r0.loc[sub["r0"] == roi].values[0] for roi in rm]
    v_pow_post = [sub.band_modules_post_r0.loc[sub["r0"] == roi].values[0] for roi in rm]

    r_pow_pre = [sub.band_modules_pre_r0.loc[sub["r0"] == roi].values[0] for roi in r_ids]
    r_pow_post = [sub.band_modules_post_r0.loc[sub["r0"] == roi].values[0] for roi in r_ids]

    # plot duration
    v_dur_post = [sub.duration_r0.loc[sub["r0"] == roi].values[0] for roi in rm]
    r_dur_post = [sub.duration_r0.loc[sub["r0"] == roi].values[0] for roi in r_ids]


    data = [[(v_peaks_pre, r_peaks_pre), (v_peaks_post, r_peaks_post), "thermal", "Hz", 0.825, 10],
            [(v_pow_pre,   r_pow_pre),   (v_pow_post,   r_pow_post),   "viridis", "dB", 0.5, 100],
            [([],        []),        (v_dur_post,   r_dur_post),       "pubu",    "ms", 0.175, 924]]


    # 1.2 Prepare labels
    v_labels = [conn.region_labels[roi] + "-v" + str(pre) + "-r" + str(roi) for pre, roi in enumerate(rm)]
    v_hovertext = [lab + "<br>PRE :: " + str(v_peaks_pre[id]) + "Hz; " + str(round(v_pow_pre[id], 4)) + "dB"
                   "<br>POST :: " + str(v_peaks_post[id]) + "Hz; " + str(round(v_pow_post[id], 4)) + "dB; " + str(v_dur_post[id]) + "ms"
                   for id, lab in enumerate(v_labels)]

    r_labels = [conn.region_labels[roi] + "-r" + str(roi) for pre, roi in enumerate(r_ids)]
    r_hovertext = [lab + "<br>PRE :: " + str(r_peaks_pre[id]) + "Hz; " + str(round(v_pow_pre[id], 4)) + "dB"
                   "<br>POST :: " + str(r_peaks_post[id]) + "Hz; " + str(round(r_pow_post[id], 4)) + "dB; " + str(r_dur_post[id]) + "ms"
                   for id, lab in enumerate(r_labels)]


    # 1.3 mount figure
    fig = make_subplots(rows=3, cols=2, horizontal_spacing=0.05, vertical_spacing=0.05, column_titles=["PRE", "POST"],
                        specs=[[{"type": "surface"}, {"type": "surface"}], [{"type": "surface"}, {"type": "surface"}],
                                               [{"type": "surface"}, {"type": "surface"}]])

    for i, row in enumerate(data):

        row_data, cs = row[:2], row[2:]
        # cmax = max([val for pair in row_data for v_r in pair for val in v_r if val != np.nan])  # Global maximum

        for j, (v_vals, r_vals) in enumerate(row_data):

            ss = True if j == 1 else False

            if v_vals:

                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=v_hovertext, hoverinfo="text",

                    cmax=cs[-1], cmin=0, showscale=ss, showlegend=False,
                    colorscale=cs[0], colorbar=dict(title=cs[1], x=1, y=cs[2], thickness=7, len=0.25),
                    intensity=v_vals,  # Intensity of each vertex, which will be interpolated and color-coded

                    i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2]), row=i+1, col=j+1)


                # Add subcortical regions not included in the cortical mesh
                fig.add_trace(go.Scatter3d(
                    x=r_centres[:, 0], y=r_centres[:, 1], z=r_centres[:, 2], hovertext=r_hovertext, hoverinfo="text", showlegend=False,
                    mode="markers", marker=dict(size=4, color=r_vals, cmax=cs[-1], cmin=0, colorscale=cs[0], showscale=False)), row=i+1, col=j+1)

    cam_dist = 1
    fig.update_layout(template="plotly_white", height=1100, title=title,
                      scene1=dict(camera=dict(eye=dict(x=cam_dist, y=cam_dist, z=cam_dist))),
                      scene2=dict(camera=dict(eye=dict(x=cam_dist, y=cam_dist, z=cam_dist))),
                      scene3=dict(camera=dict(eye=dict(x=cam_dist, y=cam_dist, z=cam_dist))),
                      scene4=dict(camera=dict(eye=dict(x=cam_dist, y=cam_dist, z=cam_dist))),
                      scene6=dict(camera=dict(eye=dict(x=cam_dist, y=cam_dist, z=cam_dist))))

    pio.write_html(fig, file=main_folder + simulations_tag + '/' + title + "_brains.html", auto_open=False)


