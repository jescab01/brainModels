import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N


# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".2003JansenRitDavid_N"
emp_subj = "NEMOS_035"
test = "sigma0_g&p"

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

main_folder = os.getcwd() + "\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

specific_folder = main_folder+"\\""PSE"+test+subjectid+"-"+emp_subj+"-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

# Prepare simulation parameters
simLength = 5 * 1000 # ms
samplingFreq = 1024 #Hz
transient = 1000 #ms
n_rep = 5

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)


conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")

mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 200, 1)
input_vals = np.arange(0, 0.22, 0.01)


results_amp = list()

for g in coupling_vals:
    for p in input_vals:
        for w in [0.8, 1]:

            # Parameters from Stefanovski 2019.
            m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                     tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                     He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                     tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
                                     w=np.array([w]), c=np.array([135.0]),

                                     c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                     c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                     v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                     p=np.array([p]), sigma=np.array([0]))

            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
            conn.speed = np.array([15])

            tic = time.time()
            print("Simulation for g = %i, input = %0.2f and w = %0.2f" % (g, p, w))

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)

            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            raw_data = output[0][1][transient:, 0, :, 0].T
            # raw_time = output[0][0][transient:]

            mean = np.average(np.average(raw_data, axis=1))
            sd = np.average(np.std(raw_data, axis=1))

            # timeseriesPlot(raw_data, raw_time, conn.region_labels)

            # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
            results_amp.append((g, p, w, mean, sd))

            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))



## GATHER RESULTS
simname = test+subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
# Working on FFT peak results
df = pd.DataFrame(results_amp, columns=["g", "p", "w", "amp_mean", "amp_sd"])
df.to_csv(specific_folder+"/PSE_"+simname+".csv", index=False)


fig = make_subplots(rows=1, cols=4, subplot_titles=("Signal std [w=0.8]", "Signal mean [w=0.8]", "Signal std [w=1]", "Signal mean [w=1]"),
                    specs=[[{}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                    x_title="Input (p)", y_title="Coupling factor")

df_subset = df.loc[df["w"] == 0.8]
fig.add_trace(go.Heatmap(z=df_subset.amp_sd, x=df_subset.p, y=df_subset.g, colorscale='Viridis', colorbar=dict(title="mV", thickness=10, x=1.02/4.8*1)), row=1, col=1)
fig.add_trace(go.Heatmap(z=df_subset.amp_mean, x=df_subset.p, y=df_subset.g, colorscale='Viridis', colorbar=dict(title="mV", thickness=10, x=1.02/4.3*2)), row=1, col=2)

df_subset = df.loc[df["w"] == 1]
fig.add_trace(go.Heatmap(z=df_subset.amp_sd, x=df_subset.p, y=df_subset.g, colorscale='Viridis', colorbar=dict(title="mV", thickness=10, x=1.02/4.1*3)), row=1, col=3)
fig.add_trace(go.Heatmap(z=df_subset.amp_mean, x=df_subset.p, y=df_subset.g, colorscale='Viridis', colorbar=dict(title="mV", thickness=10, x=1.0/4*4)), row=1, col=4)

fig.update_layout(
    title_text='Signals amplitude by coupling factor (g) and intrinsic input (p)|| %s' % emp_subj)
pio.write_html(fig, file=specific_folder + "/paramSpace-%s.html" % test, auto_open=True)
