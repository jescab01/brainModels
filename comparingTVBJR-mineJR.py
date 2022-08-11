
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

import sys

sys.path.append("E:\\LCCN_Local\PycharmProjects\\")  # temporal append
from toolbox.signals import timeseriesPlot
from toolbox.fft import FFTplot, FFTpeaks

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
wd = os.getcwd()
ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"

emp_subj = "NEMOS_035"
g, s = 17, 12.5

tic0 = time.time()

samplingFreq = 1000  # Hz
simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
transient = 0  # seconds to exclude from timeseries due to initial transient

##### GET INITIAL CONDITIONS with tvbJR
bp=False
conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2.zip")
conn.weights = conn.scaled_weights(mode="tract")

m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.05]),
                     mu=np.array([0.09]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]),
                     variables_of_interest=(["y0", "y1", "y2", "y3", "y4", "y5"]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

# Coupling function
coup = coupling.SigmoidalJansenRit(a=np.array([37]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))
conn.speed = np.array([22.5])

mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
sim.configure()

output_initial = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))

## Save initial conditions:
# np.savetxt("ic_jrd", )
# import pickle
# with open('jrd_ic','wb') as f: pickle.dump(np.concatenate((output_initial[0][1], output_initial[0][1]), axis=1), f)
# with open('jr_ic','wb') as f: pickle.dump(output_initial[0][1], f)


####
# ### Simulate with tvbJR
m_tvb = models.JansenRit(A=np.array([3.25]), B=np.array([22]), a=np.array([0.1]), b=np.array([0.05]),
                     J=np.array([1]), a_1=np.array([135]), a_2=np.array([108]), a_3=np.array([33.75]), a_4=np.array([33.75]),
                     mu=np.array([0.09]), p_max=np.array([0]), p_min=np.array([0]),
                     nu_max=np.array([0.0025]), r=np.array([0.56]), v0=np.array([6]))


# Run simulation
sim = simulator.Simulator(model=m_tvb, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          initial_conditions=output_initial[0][1])
sim.configure()

output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))
# Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")


#### Simulate with myJR
m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]), tau_e=np.array([1/m_tvb.a[0]]), tau_i=np.array([1/m_tvb.b[0]]),
                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]), c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                  p=np.array([0.09]), sigma=np.array([0]),
                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          initial_conditions=output_initial[0][1])
sim.configure()

output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))
# Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")



### Simulate with JRD and w=1
stage="JRD"
# Parameters edited from David and Friston (2003).
m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                       tau_e1=np.array([1/m_tvb.a[0]]), tau_i1=np.array([1/m_tvb.b[0]]),

                       He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                       tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                       w=np.array([1.0]), c=np.array([1]),
                       c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                       c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                       v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                       p=np.array([0.09]), sigma=np.array([0]))

# Remember to hold tau*H constant.
# m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
# m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

coup = coupling.SigmoidalJansenRitDavid(a=np.array([37]), w=m.w, e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]))


# Run simulation
art_ic = np.concatenate((output_initial[0][1], np.zeros(output_initial[0][1].shape)), axis=1)
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          initial_conditions=art_ic)
sim.configure()

output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))

raw_data = m.w * output[0][1][transient:, 0, :, 0].T + (1 - m.w) * output[0][1][transient:, 4, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")
