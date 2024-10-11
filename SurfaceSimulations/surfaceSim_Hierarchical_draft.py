
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


data_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\.DataTemp\\"
main_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\SurfSim\\"

# viz options :: signals_raw | signals_avg | timespectra_raw | timespectra_avg | 3D
viz = "timespectra_avg-3D"


# TODO initial conditions in zero?

tic = time.time()

## Simulation parameters
simLength = 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 500  # ms to exclude from timeseries due to initial transient



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

aV = dict(F=0.1, B=0.2, L=0.05)  # Scaling factor for each type of hierarchical connection

aM = np.loadtxt(data_dir + 'HCPex_hier_full_proxy.txt', dtype=str)
# aM = np.array(   # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
#             # FROM
#     [[0,   "B", 0, 0, 0, 0, 0],
#      ["F", 0,  0, 0, 0, 0, 0],
#      [0,   0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0],  # TO
#      [0,    0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0],
#      [0,    0,  0, 0, 0, 0, 0]])

aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([10]), aF=aF, aB=aB, aL=aL)


# CORTICAL SURFACE        #########################################
# look for local connectivity file, in case created previously
local_conn = data_dir + surfpack + "local_connectivity.mat" \
    if "local_connectivity.mat" in os.listdir(data_dir + surfpack) else None

cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                             region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                             local_connectivity_file=local_conn)  # cutoff=40mm (gaussian)

cx.region_mapping_data.connectivity = conn
cx.coupling_strength = np.array([0])

if not local_conn:
    cx.local_connectivity = local_connectivity.LocalConnectivity(cutoff=30.0,
                                             surface=cx.region_mapping_data.surface,
                                             equation=equations.Gaussian())

# Print ratio (g/l)
# ratio = l / np.average(conn.weights[np.triu_indices(len(conn.weights), 1)]) * 100

##### STIMULUS
# To stimulate specific area in the surface follow the region mapping
# np.argwhere(cx.region_mapping == 0)

# Mixed stimulus _Regional and Surface
stim = patterns.StimuliSurfaceRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=1, t_start=550.0, t_end=575.0)),
    spatial=equations.Gaussian(parameters=dict(amp=1, sigma=0.001, midpoint=0, offset=0)),
    surface=cx.surface,
    focal_points_surface=np.array([1273, 1799, 1882, 2078, 2163], dtype=int),  # In the order of vertices
    focal_regions=np.array([], dtype=int),)  # In the order of unmapped regions

# ## REGIONAL
# weighting = np.zeros((len(conn.region_labels), ))
# weighting[[0, 3, 5]] = 0.1
#
# stimulus_regional = patterns.StimuliRegion(
#     connectivity=conn,
#     weight=weighting)
#
# # # Configure space and time
# # stimulus_regional.configure_space()
# # stimulus_regional.configure_time(np.arange(0, simLength, 1))
#
#
# ## SURFACE
# weighting = np.zeros((len(cx.region_mapping_data.array_data), ))
# weighting[[0, 5]] = 1
#
# # cx.surface.number_of_vertices = len(cx.region_mapping)
# # cx.surface.number_of_triangles = len(cx.triangles)
#
# stimulus_surface = patterns.StimuliSurface(
#     temporal=eqn_t,
#
#     spatial=equations.Gaussian())
# #
# # Configure space and time
# # stimulus_surface.configure_space()
# # stimulus_surface.configure_time(np.arange(0, simLength, 1))
#
# #And take a look
# # plot_pattern(stimulus_surface)


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(), monitors.SpatialAverage(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, surface=cx,
                          integrator=integrator, monitors=mon, stimulus=stim)
sim.configure()


prep_time = time.time() - tic


output = sim.run(simulation_length=simLength)

print("\nSimulating %s for %is ended :: time consumed %0.2fm (preparation %0.2fs)\n (g%i, F%0.2f, B%0.2f, L%0.2f)\n"
      " Now PLOTTING. Takes time, be patient. " %
      (surfpack, simLength/1000, (time.time()-tic)/60, prep_time, coup.a, aV["F"], aV["B"], aV["L"]))


# Save local_
if not local_conn:
    localCoup = cx.local_connectivity.matrix
    scipy.io.savemat(data_dir + surfpack + 'local_connectivity.mat', {"LocalCoupling": localCoup})
    # plt.spy(localCoup)

# Plot signals
if "avg" in viz:

    data = output[1][1][transient:, 1, :, 0].T - output[1][1][transient:, 2, :, 0].T
    time = output[1][0][transient:]

    labels = conn.region_labels

    if "signals_avg" in viz:
        fig = go.Figure()
        for i, signal in enumerate(data):
            fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
        fig.update_layout(template="plotly_white")
        pio.write_html(fig, file=main_dir + "figures/signals_avg.html", auto_open=True)

    if "timespectra_avg" in viz:
        timeseries_spectra(data, simLength, transient, labels, title="timespectra_avg", width=1300)


if "raw" in viz:
    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    time = output[0][0][transient:]

    if len(data) < len(cx.region_mapping):
        labels = conn.region_labels
    else:
        labels = [conn.region_labels[id] + "-" + str(pre) for pre, id in enumerate(cx.region_mapping)]

    if "signals_raw" in viz:
        fig = go.Figure()
        for i, signal in enumerate(data):
            fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
        fig.update_layout(template="plotly_white")
        pio.write_html(fig, file=main_dir + "figures/signals_raw.html", auto_open=True)

    if "timespectra_raw" in viz:
        timeseries_spectra(data, simLength, transient, labels, title="timespectra_raw", width=1300)

# Plot in 3D
if "3D" in viz:

    labels = [conn.region_labels[id] + "-" + str(pre) for pre, id in enumerate(cx.region_mapping)]

    vertices = cx.surface.vertices
    triangles = cx.surface.triangles

    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    time = output[0][0][transient:]

    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

    # What regions are not in contained vertices?
    v_set = set(cx.region_mapping_data.array_data)
    rois_centres = np.array([centre for i, centre in enumerate(conn.centres) if i not in v_set])


    subcorticals = True if len(v_set) < len(conn.region_labels) else False

    colorscale = [[0, 'whitesmoke'], [0.5, 'gold'], [1, 'crimson']]

    fig = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=labels[:len(vertices)],
              colorbar=dict(title='mV', x=0.9, thickness=15), colorscale=colorscale, cmax=1, cmin=0,
              # Intensity of each vertex, which will be interpolated and color-coded
              intensity=data_norm[:len(vertices), 0],
              # i, j and k give the vertices of triangles
              # here we represent the 4 triangles of the tetrahedron surface
              i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], showscale=True))

    if subcorticals:
        # Add subcortical regions not included in the cortical mesh
        fig.add_trace(go.Scatter3d(x=rois_centres[:, 0], y=rois_centres[:, 1], z=rois_centres[:, 2], hovertext=labels[len(vertices):],
                                   mode="markers", marker=dict(size=5, color=data_norm[len(vertices):, 0],
                                   colorscale=colorscale, cmax=1, cmin=0)))

        fig.update(frames=[go.Frame(data=[
            go.Mesh3d(intensity=data_norm[:len(vertices), i]),
            go.Scatter3d(marker=dict(color=data_norm[len(vertices):, i]))],
                           traces=[0, 1], name=str(t)) for i, t in enumerate(time)])

    else:
        fig.update(frames=[go.Frame(data=[
            go.Mesh3d(intensity=data_norm[:len(vertices), i])],
                           traces=[0], name=str(t)) for i, t in enumerate(time)])


    fig.update_layout(template="plotly_white",
                      sliders=[
                          dict(steps=[
                              dict(method='animate',
                                   args=[[str(t)], dict(mode="immediate",
                                                        frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=100))], label=str(t))
                              for i, t in enumerate(time)],

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

    pio.write_html(fig, file=main_dir + "figures/activation3D.html", auto_open=True, auto_play=False)




