
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.signals import timeseriesPlot
from toolbox.fft import FFTplot, FFTpeaks

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
main_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\\"
ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
emp_subj = "NEMOS_035"


tic0 = time.time()

samplingFreq = 1000  #Hz
simLength = 5000  # ms - relatively long simulation to be able to check for power distribution
transient = 1000  # seconds to exclude from timeseries due to initial transient

m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                  tau_e=np.array([10]), tau_i=np.array([20]),
                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                  p=np.array([0.22]), sigma=np.array([0]),
                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.EulerDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Subset of 2 nodes is enough
conn.weights = conn.weights[:2][:, :2]
conn.tract_lengths = conn.tract_lengths[:2][:, :2]
conn.region_labels = conn.region_labels[:2]

# Coupling function
coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))

mon = (monitors.Raw(),)

for mode in ["fixed"]:
    Results = list()
    for He in np.arange(1.5, 10, 0.25):
        for Hi in np.arange(17, 35, 1):

            print(He)
            print(Hi)

            m.He = np.array([He])
            m.Hi = np.array([Hi])

            if mode == "balance":
                m.tau_e = np.array([32.5/He])
                m.tau_i = np.array([440/Hi])

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
            sim.configure()

            output = sim.run(simulation_length=simLength)
            print("Simulation time: %0.2f sec" % (time.time() - tic0,))
            # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
            raw_data = output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T
            raw_time = output[0][0][transient:]
            regionLabels = conn.region_labels

            # Check initial transient and cut data
            #timeseriesPlot(raw_data, raw_time, regionLabels, main_folder, mode="png", title="tau_e = " + str(tau_e) + " |  tau_i = " + str(tau_i))

            # Fourier Analysis plot
            #FFTplot(raw_data, simLength - transient, regionLabels, main_folder, mode="png", title="tau_e = " + str(tau_e) + " |  tau_i = " + str(tau_i))

            Results.append([mode, m.tau_e, m.tau_i, He,  Hi] +
                           list(FFTpeaks(raw_data, simLength - transient)[0]) +
                           list(FFTpeaks(raw_data, simLength - transient)[2]) +
                           [np.average(raw_data[0])])

    Results_df = pd.DataFrame(Results, columns=["mode", "tau_e", "tau_i", "He", "Hi", "roi1_Hz", "roi2_Hz", "roi1_auc", "roi2_auc", "roi1_meanS"])
    Results_df.to_csv(main_folder + "\\data\FrequencyChart_H-" + mode + ".csv")

    # Plot results
    Results_df["roi1_Hz"].loc[Results_df["roi1_auc"] < 1e-6] = 0
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Frequency", "Power", "meanSignal"], horizontal_spacing=0.12)

    fig.add_trace(go.Heatmap(z=Results_df.roi1_Hz, x=Results_df.He, y=Results_df.Hi, colorbar_x=0.26, colorbar=dict(thickness=10, title="Hz")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=Results_df.roi1_auc, x=Results_df.He, y=Results_df.Hi, colorbar_x=0.64, colorbar=dict(thickness=10, title="dB"), colorscale="Viridis"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=Results_df.roi1_meanS, x=Results_df.He, y=Results_df.Hi, colorbar=dict(thickness=10, title="mV"), colorscale="Cividis"), row=1, col=3)

    fig.update_layout(xaxis1=dict(title="He (mV)"), yaxis1=dict(title="Hi (mV)"),
                      xaxis2=dict(title="He (mV)"), yaxis2=dict(title="Hi (mV)"),
                      xaxis3=dict(title="He (mV)"), yaxis3=dict(title="Hi (mV)"),

                      title="He-Hi Chart   _" + mode)

    pio.write_html(fig, file=main_folder+"\\figures\\HeHi_"+mode+".html", auto_open=True)


