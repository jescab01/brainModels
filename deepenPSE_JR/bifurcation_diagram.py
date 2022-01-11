import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots

def bifurcationPlot(data, mode="all", folder='figures', region="Precentral_L", title="NSNC"):

    if mode=="all":
        fig = make_subplots(rows=6, cols=8, subplot_titles=[region for region in conn.region_labels[0:-1:2]],
                            shared_yaxes=True, shared_xaxes=True,
                            x_title="Coupling factor", y_title="Voltage (mV)")

        col, row= 1, 1

        for region in conn.region_labels[0:-1:2]:

            subset = data.loc[df['region'] == region]

            fig.add_scatter(x=subset.G, y=subset.minV, row=row, col=col)
            fig.add_scatter(x=subset.G, y=subset.maxV, row=row, col=col)

            if col<8:
                col=col+1
            else:
                col=1
                row=row+1

        # fig.update_yaxes(range=[0.08, 0.16])
        fig.update_layout(title="Min-max voltage by coupling factor - %s" % title)
        pio.write_html(fig, file=folder+"/bifurcation_diagrams-all_"+simname+".html", auto_open=True)

    elif mode!="all":

        fig=go.Figure()

        subset = data.loc[df['region'] == region]

        fig.add_scatter(x=subset.G, y=subset.minV)
        fig.add_scatter(x=subset.G, y=subset.maxV)

        fig.update_xaxes(title_text="Coupling factor")
        fig.update_yaxes(title_text="Voltage (mV)")
        fig.update_layout(title="Min-max voltage by coupling factor")

        pio.write_html(fig, file=folder+"/bifurcation_diagram-single_"+simname+".html", auto_open=True)




wd=os.getcwd()
main_folder = wd+"\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\CTB_data\output\\"

specific_folder = main_folder+"\\bifurcationsPSE_allNEMOS-"+time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)



# Prepare simulation parameters
simLength = 10*1000 # ms
samplingFreq = 1024 #Hz
transient = 2000 #ms
n_rep=1

# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))



# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

mon = (monitors.Raw(),)


# Define the name of the NMM to test
# and the connectome to use from the available subjects


for i in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]:

    subjectid = ".1995JansenRit"
    emp_subj = "NEMOS_0"+str(i)

    conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    coupling_vals = np.arange(0, 120, 1)
    # speed_vals = np.arange(0.5, 25, 1)
    # coupling_vals = np.arange(0, 2, 1)
    # speed_vals = np.arange(0.5, 1, 1)

    results_mV=list()

    for g in coupling_vals:

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
        conn.speed = np.array([15])
        tic = time.time()
        print("Simulating for Coupling factor = %i " % g)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        raw_data = output[0][1][transient:, 0, :, 0].T
        raw_time = output[0][0][transient:]


        for i, label in enumerate(conn.region_labels):
            results_mV.append([g, label, np.min(raw_data[i]), np.max(raw_data[i])])
        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
        # results_fft_peak.append((g, s, r, FFTpeaks(data, simLength-transient)[0][0][0],
        #                          FFTpeaks(data, simLength-transient)[1][0][0],
        #                          FFTpeaks(data, simLength-transient)))


        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))



    ## GATHER RESULTS
    simname = 'bifurcation_diagram'+subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
    # Working on FFT peak results
    df = pd.DataFrame(results_mV, columns=["G", "region", "minV", "maxV"])

    df.to_csv(specific_folder+"/bifurcation"+simname+".csv", index=False)

    bifurcationPlot(df, "single", specific_folder, title=emp_subj)
    bifurcationPlot(df, "all", specific_folder, title=emp_subj)
