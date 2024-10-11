
"""
Testing the surface-based simulations
 - w/ David model it takes TODO x min to simulate x sec
 - and implementing hierarchical connections on the HCP atlas
"""

import os
import time
import pandas as pd
import scipy
import numpy as np
from tvb.simulator.lab import *
import matplotlib.pylab as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995, JansenRitDavid2005

## Folder structure - Local
import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra


surfpack = "HCPex-r426-surf4k_pack\\"


data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\.Data\\"
main_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\Hierarchies\\"


simname = "LC_orig"

# viz options :: signals_raw | signals_avg | timespectra_raw | timespectra_avg | 3D (vnorm) (B/W)
viz_types = ["timespectra", "3Dbrain"]
viz_params = dict(timespectra=dict(modes=["raw"], show_rois=[1000, 2000]),
                  brain3D=dict(modes=["norm", "raw"], lowcolor='lightcyan'))

recomp_localconn = True

tic = time.time()

## Simulation parameters
simLength = 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 0  # ms to exclude from timeseries due to initial transient


# NEURAL MASS MODEL    #########################################################
m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                       tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                       gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                       gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                       p=np.array([0]), sigma=np.array([0]),
                       e0=np.array([0.0025]), r=np.array([0.56]))

m.stvar = np.array([1])  # Define where to input the stimulation


# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.weights = conn.scaled_weights(mode="tract")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022


# COUPLING :: Hierarchical      #########################################################
aV = dict(F=0.3, B=0.6, L=0.15)  # Scaling factor for each type of hierarchical connection

aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy.txt', dtype=str)

aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([100]), aF=aF, aB=aB, aL=aL)


# CORTICAL SURFACE        #########################################
# look for local connectivity file, in case created previously
local_conn = data_dir + surfpack + "local_connectivity.mat" \
    if ("local_connectivity.mat" in os.listdir(data_dir + surfpack)) and (not recomp_localconn) else None

cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                             region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                             local_connectivity_file=local_conn)  # cutoff=40mm (gaussian)

cx.region_mapping_data.connectivity = conn
cx.coupling_strength = np.array([1])

if not local_conn or recomp_localconn:
    cx.local_connectivity = (
        local_connectivity.LocalConnectivity(
            cutoff=40.0, surface=cx.region_mapping_data.surface,
            equation=equations.Gaussian(parameters=dict(amp=1, sigma=1, midpoint=0, offset=0))))


##### STIMULUS
# To stimulate specific area in the surface follow the region mapping
# np.argwhere(cx.region_mapping == 0)

# # Mixed stimulus _Regional and Surface
stim = patterns.StimuliSurfaceRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=0.1, t_start=50.0, t_end=75.0)),
    spatial=equations.Gaussian(parameters=dict(amp=1, sigma=1, midpoint=0, offset=0)),
    surface=cx.surface,
    # focal_points_surface=np.array([4310, 3111, 4089, 4290], dtype=int),  # (8k V1) || In the order of vertices
    focal_points_surface=np.array([1273, 1799, 1667, 1882, 2078], dtype=int),  # 4k V1
    focal_regions=np.array([], dtype=int),)  # In the order of conn.region_labels

## Slow Spontaneous Oscillations
# stim = patterns.SpontaneousActivity(
#     # temporal=equations.SO(parameters=dict(up_time=1.08, up_std=0.38, down_time=3.44, down_std=1.37)),  # in seconds
#     temporal=equations.SO(parameters=dict(up_time=0.02, up_std=0.005, down_time=3.44, down_std=1.37,  # in seconds
#                                           ref_time=0.1, ref_slope=20, ref_threshold=0.00001)),
#     spatial=equations.Gaussian(parameters=dict(amp=0.0001, sigma=0.001, midpoint=0, offset=0)),
#     surface=cx.surface)


# INITIAL CONDITIONS   ###
# DavidFriston2005 use to have equilibrium in prebif at 0;
# therefore using 0-init.
n_subcx = len(set(range(len(conn.region_labels))).difference(cx.region_mapping_data.array_data))
if "SpontaneousActivity" in stim.title:
    init_pad = int(stim.temporal.parameters["ref_time"] * 1000)  # timepoints for 0-init
else:
    init_pad = 100

init = np.zeros((init_pad, 8, len(cx.region_mapping_data.array_data) + n_subcx, 1))


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(), monitors.SpatialAverage(),)


# SIMULATE  ####
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, surface=cx,
                          integrator=integrator, monitors=mon, stimulus=stim, initial_conditions=init)
sim.configure()


prep_time = time.time() - tic


output = sim.run(simulation_length=simLength)

print("\n(SURF) Simulating %s for %0.1fs ended :: time consumed %0.2fs (preparation %0.2fs)\n "
      "(g%i, F%0.2f, B%0.2f, L%0.2f, lc%i)\n"
      " Now PLOTTING. Takes time, be patient. " %
      (surfpack, simLength/1000, (time.time()-tic), prep_time, coup.a, aV["F"], aV["B"], aV["L"], cx.coupling_strength[0]), end="\r")


# Save local_
if not local_conn or recomp_localconn:
    localCoup = cx.local_connectivity.matrix
    scipy.io.savemat(data_dir + surfpack + 'local_connectivity.mat', {"LocalCoupling": localCoup})
    # plt.spy(localCoup)



title = "g%iF%0.2fB%0.2fL%0.2f-lc%i_%s" % (coup.a, aV["F"], aV["B"], aV["L"], cx.coupling_strength[0], simname)

tic = time.time()

for viz in viz_types:

    if "timespectra" in viz:

        for mode in viz_params["timespectra"]["modes"]:

            if "avg" in mode:
                data = output[1][1][transient:, 1, :, 0].T - output[1][1][transient:, 2, :, 0].T
                labels = [str(i) + "-" + roi for i, roi in enumerate(conn.region_labels)]

            elif "raw" in mode:
                l_show, t_show = viz_params["timespectra"]["show_rois"]
                data = output[0][1][transient:, 1, l_show:t_show, 0].T - output[0][1][transient:, 2, l_show:t_show, 0].T
                labels = [str(pre) + "-" + conn.region_labels[id] for pre, id in enumerate(cx.region_mapping)]

            timeseries_spectra(data, simLength, transient, labels, title="timespectra" + mode + title, width=1050, height=750)

    if "3D" in viz:

        for mode in viz_params["brain3D"]["modes"]:

            labels = [str(pre) + "-" + conn.region_labels[id] for pre, id in enumerate(cx.region_mapping)]

            vertices = cx.surface.vertices
            triangles = cx.surface.triangles

            data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
            time_ = output[0][0][transient:] - init_pad

            if "raw" in mode:
                bar_title = 'mV'
            else:
                data = np.array([(signal - 0) / np.std(signal) for signal in data])
                bar_title = 'vertex-wise<br>standardized<br>activation'

            cmax = np.max(data[~np.isnan(data) & ~np.isinf(data)])
            cmin = np.min(data[~np.isnan(data) & ~np.isinf(data)])

            zeroc = np.abs(cmin / (cmax - cmin))
            fiftyc = zeroc + ((1 - zeroc) / 2)

            # What regions are not in contained vertices?
            v_set = set(cx.region_mapping_data.array_data)
            rois_centres = np.array([centre for i, centre in enumerate(conn.centres) if i not in v_set])

            subcorticals = True if len(v_set) < len(conn.region_labels) else False

            colorscale = [[0, viz_params["brain3D"]["lowcolor"]], [zeroc, 'whitesmoke'], [fiftyc, 'gold'], [1, 'crimson']]

            fig = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=labels[:len(vertices)],
                      colorbar=dict(title=bar_title , x=0.9, thickness=15), colorscale=colorscale, cmax=cmax, cmin=cmin,
                      # Intensity of each vertex, which will be interpolated and color-coded
                      intensity=data[:len(vertices), 0],
                      # i, j and k give the vertices of triangles
                      # here we represent the 4 triangles of the tetrahedron surface
                      i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], showscale=True))

            if subcorticals:
                # Add subcortical regions not included in the cortical mesh
                fig.add_trace(go.Scatter3d(x=rois_centres[:, 0], y=rois_centres[:, 1], z=rois_centres[:, 2], hovertext=labels[len(vertices):],
                                           mode="markers", marker=dict(size=5, color=data[len(vertices):, 0],
                                           colorscale=colorscale, cmax=cmax, cmin=cmin)))

                fig.update(frames=[go.Frame(data=[
                    go.Mesh3d(intensity=data[:len(vertices), i]),
                    go.Scatter3d(marker=dict(color=data[len(vertices):, i]))],
                                   traces=[0, 1], name=str(t)) for i, t in enumerate(time_)])

            else:
                fig.update(frames=[go.Frame(data=[
                    go.Mesh3d(intensity=data[:len(vertices), i])],
                                   traces=[0], name=str(t)) for i, t in enumerate(time_)])


            fig.update_layout(template="plotly_white", title=title,
                              sliders=[
                                  dict(steps=[
                                      dict(method='animate',
                                           args=[[str(t)], dict(mode="immediate",
                                                                frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                                transition=dict(duration=100))], label=str(t))
                                      for i, t in enumerate(time_)],

                                      transition=dict(duration=0), x=0.3, xanchor="left", y=-0.05,
                                      currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                                      len=0.52, tickcolor="white")],
                              updatemenus=[dict(type="buttons", showactive=False, y=-0.1, x=0.2, xanchor="left",
                                      buttons=[dict(label="Play", method="animate",
                                               args=[None,
                                                     dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                          transition=dict(duration=100),
                                                          fromcurrent=True, mode='immediate')]),
                                          dict(label="Pause", method="animate",
                                               args=[[None],
                                                     dict(frame=dict(duration=0, redraw=False, easing="cubic-in-out"),
                                                          transition=dict(duration=0),
                                                          mode="immediate")])])])

            pio.write_html(fig, file=main_dir + "figures/activation3D" + mode + "_" + title + ".html", auto_open=True, auto_play=False)

print("  elapsed %0.2fs" % (time.time() - tic))

