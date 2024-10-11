

import os
import scipy
from scipy.special import comb
import numpy as np
from tvb.simulator.lab import *
import matplotlib.pylab as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995

## Folder structure - Local
import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra
from toolbox.dfc import kuramoto_order, kuramoto_polar


"""
Testing the surface-based simulations
 - w/ JansenRit it takes 1 min to simulate 1 sec
 - The problem will be the scaling in RAM (1 sim.sec=2Gb)
"""


# surfpack = "HCPex-r426-surfdisc4k_pack\\"
# surfpack = "defsub-r7-surf408_pack\\"
surfpack = "subHCPex-r2R-surf499_pack\\"
local_conn = "local_connectivity-amp1.0sig3.67cut5.51.mat"

data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.DataTemp\\"
main_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\SurfSim\\"

# viz options :: signals_raw | signals_avg | timespectra_raw | timespectra_avg | 3D | KO
viz = "timespectra_avg_raw_ko"


## Simulation parameters
simLength = 9000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient



# NEURAL MASS MODEL    #########################################################
m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                  tau_e=np.array([10]), tau_i=np.array([20]),
                  c=np.array([125]), c_pyr2exc=np.array([1]), c_exc2pyr=np.array([0.8]),
                  c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                  p=np.array([0.22]), sigma=np.array([0]),
                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([5.52]))

m.stvar = np.array([1])  # Define where to input the stimulation


# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.weights = conn.scaled_weights(mode="tract")

conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022


coup = coupling.SigmoidalJansenRit(a=np.array([40]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))


# CORTICAL SURFACE        #########################################
# look for local connectivity file, in case created previously
cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                             region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                             local_connectivity_file=data_dir + surfpack + "local_conn/" + local_conn)  # cutoff=40mm (gaussian)

cx.region_mapping_data.connectivity = conn
cx.coupling_strength = np.array([0.001])


##### STIMULUS

## Prepare the independent stimuli per node
# Activation probability. Selecting nodes from a pool with replacement
rm = cx.region_mapping_data.array_data
nodes_pool = np.argwhere(rm == 0).squeeze()  # Activating only region 0

nAct, reAct = 400, True

# Given the number of extractions and if reactivation, the probability of a node to activate at least one is:
pAct = 1 - ((len(rm) - 1) / len(rm)) ** nAct if reAct else 1 - comb((len(rm) - 1), nAct) / comb(len(rm), nAct)  # w/ wo/ replacement (re-activation)s

print("Probability of activation per subnode: %0.2f (nAct %i; reAct - %s)" % (pAct, nAct, reAct))

samples = np.random.choice(nodes_pool, size=nAct, replace=reAct)

stim_set = []
for sample in samples:

    # # Activation amplitude
    # wn_amp = abs(np.random.normal(0.5, 1))
    #
    # # Activation duration and start time
    # wn_start = np.random.choice(np.arange(transient, simLength, 1), size=1)[0]
    # wn_dur = abs(np.random.normal(500, 250))
    #
    # onset = wn_start
    # offset = wn_start + wn_dur if wn_start + wn_dur < simLength else simLength

    # General approach :: Mixed stimulus _Regional and Surface. Both or any.
    stim_ind = patterns.StimuliSurfaceRegion(
        # temporal=equations.PulseTrain(parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
        temporal=equations.WhiteNoise(parameters=dict(mean=0, std=1, onset=500, offset=2500)),
        # temporal=equations.PinkNoise(parameters=dict(gain=10, n_sources=50, onset=500, offset=2500)),
        spatial=equations.Gaussian(parameters=dict(amp=1, sigma=0.001, midpoint=0, offset=0)),
        surface=cx.surface, focal_points_surface=np.array([sample]),  # In the order of vertices
        focal_regions=np.array([], dtype=np.int64),)  # In the order of unmapped regions


    stim_set.append(stim_ind)

stim = patterns.MultiStimuli(stim_set)


# stim = patterns.StimuliSurface(
#     temporal=equations.WhiteNoise(parameters=dict(mean=0, std=1, onset=1500, offset=2000)),
#     spatial=equations.Gaussian(parameters=dict(amp=1, sigma=0.001, midpoint=0, offset=0)),
#     surface=cx.surface, focal_points_surface=np.array([113, 265, 275, 267, 393]) )



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

output = sim.run(simulation_length=simLength)


# print("Ration l/g = %0.4f" % ratio)


# Plot signals
if "avg" in viz:

    data = output[1][1][transient:, 1, :, 0].T - output[1][1][transient:, 2, :, 0].T
    time = output[1][0][transient:]

    labels = conn.region_labels

    if "signals" in viz:
        fig = go.Figure()
        for i, signal in enumerate(data):
            fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
        fig.update_layout(template="plotly_white")
        pio.write_html(fig, file=main_dir + "figures/signals_avg.html", auto_open=True)

    if "timespectra" in viz:
        timeseries_spectra(data, simLength, transient, labels, title="timespectra_avg", width=1000, height=550)


if "raw" in viz:
    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    time = output[0][0][transient:]

    if len(data) < len(cx.region_mapping):
        labels = conn.region_labels
    else:
        labels = [str(pre) + "-" + conn.region_labels[id] for pre, id in enumerate(cx.region_mapping)]

    if "signals" in viz:
        fig = go.Figure()
        for i, signal in enumerate(data):
            fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
        fig.update_layout(template="plotly_white")
        pio.write_html(fig, file=main_dir + "figures/signals_raw.html", auto_open=True)

    if "timespectra" in viz:
        timeseries_spectra(data, simLength, transient, labels, title="timespectra_raw", width=1000, height=750)


# Plot in 3D
if "3D" in viz:

    labels = [conn.region_labels[id] + "-" + str(pre) for pre, id in enumerate(cx.region_mapping)]

    vertices = cx.surface.vertices
    triangles = cx.surface.triangles

    raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    raw_data_norm = (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))

    # que regiones no estan en vertices?
    v_set = set(cx.region_mapping_data.array_data)
    rois_centres = np.array([centre for i, centre in enumerate(conn.centres) if i not in v_set])


    subcorticals = True if len(v_set) < len(conn.region_labels) else False


    fig = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=labels[:len(vertices)],
              colorbar=dict(title='mV', x=0.9, thickness=15),
              colorscale=[[0, 'mediumturquoise'], [0.5, 'gold'], [1, 'magenta']],
                        cmax=1, cmin=0,
              # Intensity of each vertex, which will be interpolated and color-coded
              intensity=raw_data_norm[:len(vertices), 0],
              # i, j and k give the vertices of triangles
              # here we represent the 4 triangles of the tetrahedron surface
              i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], showscale=True))

    if subcorticals:
        # Add subcortical regions not included in the cortical mesh
        fig.add_trace(go.Scatter3d(x=rois_centres[:, 0], y=rois_centres[:, 1], z=rois_centres[:, 2], hovertext=labels[len(vertices):],
                                   mode="markers", marker=dict(color=raw_data_norm[len(vertices):, 0],
                                   colorscale=[[0, 'mediumturquoise'], [0.5, 'gold'], [1, 'magenta']], cmax=1, cmin=0)))

        fig.update(frames=[go.Frame(data=[
            go.Mesh3d(intensity=raw_data_norm[:len(vertices), i]),
            go.Scatter3d(marker=dict(color=raw_data_norm[len(vertices):, i]))],
                           traces=[0, 1], name=str(t)) for i, t in enumerate(raw_time)])

    else:
        fig.update(frames=[go.Frame(data=[
            go.Mesh3d(intensity=raw_data_norm[:len(vertices), i])],
                           traces=[0], name=str(t)) for i, t in enumerate(raw_time)])


    fig.update_layout(template="plotly_white",
                      sliders=[
                          dict(steps=[
                              dict(method='animate',
                                   args=[[str(t)], dict(mode="immediate",
                                                        frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=100))], label=str(t))
                              for i, t in enumerate(raw_time)],

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


## Kuramotos
if "ko" in viz:
    data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    time = output[0][0][transient:]

    rm = cx.region_mapping_data.array_data

    KOarray, KOstd, KOavg = kuramoto_order(data[rm == 0], samplingFreq)

    kuramoto_polar(data[rm == 0], time, samplingFreq, 10)

    #
    # KOarray, KOstd, KOavg = kuramoto_order(data[:4], samplingFreq)
    #
    # kuramoto_polar(data[:4], time, samplingFreq, 10)
    #
    #
    # ## TEMP
    #
    # fig=go.Figure()
    # fig.add_trace(go.Scatter(x=time, y=data[0]))
    # fig.add_trace(go.Scatter(x=time, y=filterSignals[0]))
    # fig.add_trace(go.Scatter(x=time, y=efPhase[0]))
    # fig.show("browser")