import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N1
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

from toolbox import timeseriesPlot, FFTplot, FFTpeaks

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = ".2003JansenRitDavid"
wd = os.getcwd()
main_folder=wd+"\\"+subjectid
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

emp_subj = "NEMOS_035"


tic0 = time.time()

samplingFreq = 1000 #Hz
simLength = 2000 # ms - relatively long simulation to be able to check for power distribution
transient = 1000 # seconds to exclude from timeseries due to initial transient

m = JansenRitDavid2003_N1(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10.0]), tau_i=np.array([20.0]),
                          v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                          c=np.array([135.0]),
                          c_11=np.array([1.0]), c_12=np.array([0.8]),
                          c_21=np.array([0.25]), c_22=np.array([0.25]),
                          p=np.array([0.22]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.EulerDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Coupling function
coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))
conn.speed = np.array([6])

mon = (monitors.Raw(),)

Results = list()
for tau_e in np.arange(2, 60, 2):
    for tau_i in np.arange(2, 60, 2):

        m.tau_e = np.array([tau_e])
        m.tau_i = np.array([tau_i])
        m.He = np.array([32.5/tau_e])
        m.Hi = np.array([440/tau_i])

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

        Results.append([tau_e] + [tau_i] + list(FFTpeaks(raw_data, simLength - transient)[0])+list(FFTpeaks(raw_data, simLength - transient)[1]))

Results_df = pd.DataFrame(Results, columns=["tau_e", "tau_i", "ACCl_Hz", "ACCr_Hz", "Prl_Hz", "Prr_Hz", "ACCl_auc", "ACCr_auc", "Prl_auc", "Prr_auc"])
Results_df.to_csv(main_folder + "\\FrequencyChart_tau.csv")

# por los labels de los ejes alma de cántaro.
fig = go.Figure(go.Heatmap(z=Results_df.ACCl_Hz, x=Results_df.tau_e, y=Results_df.tau_i))
pio.write_html(fig, file=main_folder+"\\FrequencyChart_tau.html", auto_open=True)

fig = go.Figure(go.Heatmap(z=Results_df.ACCl_auc, x=Results_df.tau_e, y=Results_df.tau_i))
pio.write_html(fig, file=main_folder+"\\ModuleChart_tau.html", auto_open=True)

