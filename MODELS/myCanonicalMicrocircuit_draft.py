import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995, JansenRitDavid2005, myCanonicalMicroCircuit

## Folder structure - Local
import sys

sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.mixes import timeseries_spectra

surfpack = "HCPex-r426-surfdisc17k_pack\\"
# surfpack = "default-r76-surf16k_pack\\"
data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\.DataTemp\\"



# viz options :: signals_raw | timespectra_raw
viz = "timespectra_raw"



## Simulation parameters
simLength = 500  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 10000  # Hz
transient = 0  # ms to exclude from timeseries due to initial transient

nrois = 2

# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

conn.weights = conn.weights[:, :nrois][:nrois]
conn.tract_lengths = conn.tract_lengths[:, :nrois][:nrois]
conn.centres = conn.centres[:nrois]
conn.region_labels = conn.region_labels[:nrois]
conn.cortical = conn.cortical[:nrois]

conn.weights = conn.scaled_weights(mode="tract")

# conn.weights = np.array(
#     # FROM
#     [[0, 1, 0, 0, 0, 0, 0],
#      [1, 0, 1, 0, 0, 0, 0],
#      [0, 1, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],  # TO
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0]])

# c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
#                   c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
# c_pyr2exc=np.array([50]), c_exc2pyr=np.array([40]),
#                   c_pyr2inh=np.array([12]), c_inh2pyr=np.array([12]),
# NEURAL MASS MODEL    #########################################################
c = 135
m = myCanonicalMicroCircuit(He=np.array([3.25]), Hi_slow=np.array([22]), Hi_fast=np.array([10]),
                          taue=np.array([10]), taui_slow=np.array([16]), taui_fast=np.array([2]),
                          c_exc2pyrsup=np.array([c]), c_pyrsup2pyrinf=np.array([0.8*c]),
                          c_pyrsup2inhslow=np.array([0.25*c]), c_exc2inhfast=np.array([0.25*c]),
                          c_inhfast2pyrsup=np.array([0.25*c]), c_inhslow2pyrinf=np.array([0.25*c]),
                          p=np.array([0]), sigma=np.array([0]),
                          e0=np.array([0.0025]), r=np.array([0.56]), v0=np.array([6]))

m.stvar = np.array([1])  # Define where to input the stimulation

# COUPLING    #########################################################
aV = dict(FF=40, FB=50)  # Scaling factor for each type of hierarchical connection


aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)
aM = aM[:, :nrois][:nrois]

aF = np.array([[aV["FF"] if val == "F" else 0 for val in row] for row in aM])
aB = np.array([[aV["FB"] if val == "B" else 0 for val in row] for row in aM])
coup = coupling.SigmoidalCanonicalMicroCircuit(a=np.array([50]), aF=aF, aB=aB)



# STIMULUS    #########################################################
weighting = np.zeros((len(conn.region_labels),))
weighting[[0]] = 0.1

stim = patterns.StimuliRegion(
    # temporal=equations.PulseTrain(
    #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
    temporal=equations.DC(parameters=dict(dc_offset=0.1, t_start=2000.0, t_end=2025.0)),
    weight=weighting, connectivity=conn)  # In the order of unmapped regions

# ## REGIONAL
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

data = output[0][1][transient:, 0, :, 0].T + output[0][1][transient:, 2, :, 0].T
time = output[0][0][transient:]

labels = conn.region_labels

data = output[0][1][transient:, :, 0, 0].T
labels = ["vPyr_l2l3", "vExc_l4", "vPyr_l5l6", "vInh_fast", "vInh_slow"]

if "signals_raw" in viz:
    fig = go.Figure()
    for i, signal in enumerate(data):
        fig.add_trace(go.Scatter(x=time, y=signal, name=labels[i]))
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file="figures/David2005-signals_raw.html", auto_open=True)

if "timespectra_raw" in viz:
    timeseries_spectra(data, simLength, transient, labels, title="CMC2012-timespectra_raw", width=1000, height=750)
