
import numpy as np
import scipy
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

surfpack = "subset-r7-surf408_pack\\"
# surfpack = "default-r76-surf16k_pack\\"

main_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\SurfSim\\"

# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"

## Simulation parameters
simLength = 2000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# NEURAL MASS MODEL    #########################################################
m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                  tau_e=np.array([10]), tau_i=np.array([20]),
                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                  p=np.array([0.08]), sigma=np.array([0]),
                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

m.stvar = np.array([1])  # Define where to input the stimulation

# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(main_dir + surfpack + "connectivity.zip")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

                                # FROM
conn.weights = np.array([[0,  5, 5, 0, 0, 0, 0],
                         [10, 0, 4, 0, 0, 0, 0],
                         [10, 4, 0, 0, 0, 0, 0],
                         [0,  0, 0, 0, 0, 0, 0],  # TO
                         [0,  0, 0, 0, 0, 0, 0],
                         [0,  0, 0, 0, 0, 0, 0],
                         [0,  0, 0, 0, 0, 0, 0]])

coup = coupling.SigmoidalJansenRit(a=np.array([5]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))

##### STIMULUS
weighting = np.zeros((len(conn.region_labels),))
weighting[[0, 3, 5]] = 0.1

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=1, t_start=1500.0, t_end=1525.0)),
    weight=weighting, connectivity=conn)  # In the order of unmapped regions

# ## REGIONAL
#
# stimulus_regional = patterns.StimuliRegion(
#     connectivity=conn,
#     weight=weighting)
#
# # # Configure space and time
# # stimulus_regional.configure_space()
# # stimulus_regional.configure_time(np.arange(0, simLength, 1))

# #And take a look
# # plot_pattern(stimulus_surface)


# OTHER PARAMETERS   ###
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          stimulus=stim)

sim.configure()

output = sim.run(simulation_length=simLength)

data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
time = output[0][0][transient:]

labels = conn.region_labels

if "signals_raw" in viz:
    fig = go.Figure()
    for i, signal in enumerate(data):
        fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file=main_dir + "figures/signals_raw.html", auto_open=True)

if "timespectra_raw" in viz:
    timeseries_spectra(data, simLength, transient, labels, title="timespectra_raw")
