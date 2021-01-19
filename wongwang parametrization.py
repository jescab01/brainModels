import time

import numpy as np

from tvb.simulator.lab import *
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace

subjectid = ".2006WongWang"
wd=os.getcwd()
main_folder=wd+"\\"+subjectid
ctb_folder=wd+"\\CTB_data\\output\\"

emp_subj = "subj04"

tic0=time.time()
tic1=time.time()

simLength = 2000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000 #Hz
transient=100 # ms to exclude from timeseries due to initial transient

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1, noise=noise.Additive(nsig=np.array([0.01])))
integrator = integrators.HeunDeterministic(dt=1)


conn = connectivity.Connectivity.from_file(ctb_folder+"CTB_connx66_"+emp_subj+".zip")
conn.weights = conn.scaled_weights(mode="tract")
coup = coupling.Linear(a=np.array([0]))

mon = (monitors.Raw(),)

# m = models.ReducedWongWang(I_o=np.array([0.33]), J_N=np.array([0.2609]),
#                            a=np.array([0.27]), b=np.array([0.108]), d=np.array([154]),
#                            gamma=np.array([0.641]), sigma_noise=np.array([0.000000001]),
#                            tau_s=np.array([100]), w=np.array([0.6]))


#########
# Parameter space exploration
########

# W_i=[0.7, 0.9]
W_i=np.arange(0.7,0.9,0.01)
# J_N=[0.15, 0.25]
J_N=np.arange(0.15,0.25, 0.005)
# I_o=[0.382, 0.42]
I_o=np.arange(0.35,0.42,0.002)
#J_i=[1]

Peaks=list()
Modules=list()
ii=0
for wi in W_i:
    for jn in J_N:
        ii += 1
        print(str(ii) + "/" + str(int(len(W_i) * len(J_N))), end=" - ")
        print("Exploration time: %0.2f sec" % (time.time() - tic0,), end=" - ")
        print("Added time: %0.2f sec" % (time.time()- tic1,), end=" - ")
        print(time.asctime())
        tic1 = time.time()
        for io in I_o:

            m = models.ReducedWongWangExcInh(G=np.array([0]), I_o=np.array([0.3]),
                                             J_N=np.array([0.15]), J_i=np.array([1]),
                                             W_e=np.array([1]), W_i=np.array([0.7]),
                                             a_e=np.array([310]), a_i=np.array([615]),
                                             b_e=np.array([125]), b_i=np.array([177]),
                                             d_e=np.array([0.16]), d_i=np.array([0.087]),
                                             gamma_e=np.array([0.000641]), gamma_i=np.array([0.001]),
                                             lamda=np.array([0]),
                                             tau_e=np.array([100]), tau_i=np.array([10]),
                                             w_p=np.array([1.4]))

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
            sim.configure()

            output = sim.run(simulation_length=simLength)

            # Extract data cutting initial transient
            raw_data = output[0][1][:, 0, :, 0].T
            raw_time = output[0][0][:]
            regionLabels = conn.region_labels
            regionLabels=list(regionLabels)
            regionLabels.insert(0,"AVG")

            # average signals to obtain mean signal frequency peak
            data = np.asarray([np.average(raw_data, axis=0)])
            data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]
            # # Check initial transient and cut data
            # t="W_I="+str(wi)+"  |  J_N="+str(jn)+"  |  I_o="+str(io)
            # timeseriesPlot(data[1:5], raw_time, regionLabels[1:], main_folder)#, title=t, mode="png")
            # # Inihibitory subpopulation
            # raw_Si = output[0][1][:, 1, :, 0].T
            # timeseriesPlot(raw_Si[0:4], raw_time, regionLabels[1:], main_folder, title=t, mode="png")
            # # Fourier Analysis plot
            #  FFTplot(data[1:5], simLength, regionLabels[1:], main_folder,  mode="html")

            ### FFT analysis
            peaks = list()
            modules = list()
            signals=data[1:8]
            for i in range(len(signals)):
                fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
                fft = fft[range(np.int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
                freqs = np.arange(len(signals[i]) / 2)
                freqs = freqs / (simLength / 1000)  # simLength (ms) / 1000 -> segs

                fft = fft[freqs > 0.5]  # remove undesired frequencies from peak analisis
                freqs = freqs[freqs > 0.5]

                modules.append(fft[np.where(fft == max(fft))])
                peaks.append(freqs[np.where(fft == max(fft))])

            Modules.append(np.concatenate((np.asarray([wi,jn,io]),np.asarray(modules)[:,0])))
            Peaks.append(np.concatenate((np.asarray([wi,jn,io]),np.asarray(peaks)[:,0])))

M=np.asarray(Modules)
P=np.asarray(Peaks)

import pandas as pd
colnames=["wi","jn","io","s1","s2","s3","s4","s5","s6","s7"]
Mdf=pd.DataFrame(M)
Mdf.columns=colnames
Mdf.to_csv("modulesdf1.csv")
Pdf=pd.DataFrame(P)
Pdf.columns=colnames
Pdf.to_csv("peaksdf1.csv")

## Load
M=pd.read_csv("modulesdf1.csv", index_col=0)
P=pd.read_csv("peaksdf1.csv", index_col=0)

Ps=P.loc[(P["s1"]>1) | (P["s2"]>1)| (P["s3"]>1)| (P["s4"]>1)| (P["s5"]>1)| (P["s6"]>1)| (P["s7"]>1)]

import plotly.express as px
import plotly.io as pio
fig=px.scatter(Ps, x="wi", y="jn")
pio.write_html(fig, file="Ti.html", auto_open=True)



import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()

import plotly.graph_objects as go
import numpy as np

N = 1000
t = np.linspace(0, 10, 100)
y = np.sin(t)

fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))

fig.show()