import os
import time
import subprocess

import numpy as np
from scipy import signal, stats, cluster
import pandas as pd

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

import sys
sys.path.append("E:/LCCN_Local/PycharmProjects")
from toolbox.signals import timeseriesPlot, epochingTool
from toolbox.fc import PLV

from plotly.subplots import make_subplots
from communities.algorithms import louvain_method
from sklearn.linear_model import LinearRegression


# Define the name of the NMM to test
# and the connectome to use from the available subjects
subjectid = ".1995JansenRit"
emp_subj = "NEMOS_035"

wd = os.getcwd()
main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"


specific_folder = main_folder + "\\dynFC" + subjectid + "-" + emp_subj + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1024  # Hz
transient = 500  # ms

# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig = np.array([1e-9]))) # between -8 and -9
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2.zip")
conn.weights = conn.scaled_weights(mode="tract")
regionLabels = conn.region_labels

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                 'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                 'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                 'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Calcarine_L',
                 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                 'Temporal_Inf_R']


# load text with FC rois; check if match SC
FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in
                 cortical_rois]  # find indexes in FClabs that matches cortical_rois

SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


# Load empirical FC data to make simple comparisons
plv_emp = np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "\\3-alpha_plv_rms.txt", delimiter=",")[:, FC_cortex_idx][FC_cortex_idx]
# aec_emp = np.loadtxt(wd + "\\CTB_data\\output\\FC_" + emp_subj + "\\3-alpha_aec.txt")

# # Clustering for better visualization
communs, frames = louvain_method(plv_emp)
indexes = [item for c in communs for item in c]
PLVemp = plv_emp[:, indexes][indexes]
# AECemp = aec_emp[:, indexes][indexes]


mon = (monitors.Raw(),)

coupling_vals = np.arange(0, 120, 1)
# coupling_vals = np.arange(0, 120, 1)

PLVsim = list()
PLVr = list()

# AECsim=list()
# AECr=list()

for g in coupling_vals:

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([15])

    tic = time.time()
    print("Simulating for Coupling factor = %i" % g)

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_data = output[0][1][transient:, 0, :, 0].T
    raw_data = raw_data[SC_cortex_idx, :]  # Filter cortical rois
    raw_data = raw_data[indexes, :]  # Match clustering order

    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels[SC_cortex_idx]

    # average signals to obtain mean signal frequency peak
    # data = np.asarray([np.average(raw_data, axis=0)])
    # data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

    # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
    # results_fft_peak.append((g, FFTpeaks(data, simLength-transient)[0][0][0],
    #                          FFTpeaks(data, simLength-transient)[1][0][0],
    #                          FFTpeaks(data, simLength-transient)))

    newRow = [g]
    bands = [["3-alpha"], [(8, 12)]]
    # [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]
    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.unwrap(np.angle(analyticalSignal)))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], regionLabels, 1, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV
        plv_sim = PLV(efPhase)
        PLVsim.append(plv_sim)
        # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"plv.txt"
        # np.savetxt(fname, plv)

        ## AEC
        # aec_sim = AEC(efEnvelope)
        # AECsim.append(aec_sim)
        # fname = wd+"\\CTB_data\\output\\"+subjectid+"\\"+bands[0][b]+"corramp.txt"
        # np.savetxt(fname, aec)
        # Comparisons


        t1 = np.zeros(shape=(2, len(plv_sim) ** 2 // 2 - len(plv_sim) // 2))
        t1[0, :] = plv_sim[np.triu_indices(len(plv_sim), 1)]
        t1[1, :] = PLVemp[np.triu_indices(len(plv_sim), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        # t3 = np.zeros(shape=(2, 2145))
        # t3[0, :] = aec_sim[np.triu_indices(66, 1)]
        # t3[1, :] = AECemp[np.triu_indices(66, 1)]
        # aec_r = np.corrcoef(t3)[0, 1]
        #
        PLVr.append(plv_r)
        # AECr.append(aec_r)

    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic,))

# Save all this info somehow


## ANIMATION
# ############################# try subplotting
# https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/
# https://community.plotly.com/t/single-subplot-animations/35235

# Prepare colors per community for scatter plot
def clusterColor(emp_matrix, communs):
    colors_array = [i for i, c in enumerate(communs) for item in c] # item just to repeat
    colors_m = np.ndarray((len(emp_matrix), len(emp_matrix)))
    for i, c1 in enumerate(colors_array):
        for j, c2 in enumerate(colors_array):
            if c1 == c2:
                colors_m[:, j][i] = c1 + 1
            else:
                colors_m[:, j][i] = 0

    return colors_m[np.triu_indices(len(emp_matrix), 1)]
colors=clusterColor(PLVemp, communs)
# Prepare regression lines
def regLines(emp_matrix, sim_matrix):
    regressor = LinearRegression()

    X = np.reshape(emp_matrix[np.triu_indices(len(emp_matrix), 1)],
                   (len(emp_matrix[np.triu_indices(len(emp_matrix), 1)]), 1))
    x_range = np.linspace(0, 1, 100)

    reglines = list()
    for i in range(len(sim_matrix)):
        y = sim_matrix[i][np.triu_indices(len(emp_matrix), 1)]
        model = regressor.fit(X, y)
        y_range = model.predict(x_range.reshape(-1, 1))
        reglines.append(y_range)

    return reglines

## TRIPLOT
# def PLV_triplot(PLVemp, PLVsim, PLVr, PLV_rl, colors):
#     fig = make_subplots(rows=1, cols=3, subplot_titles=('PLV empirical','PLV simulated', 'rPLV emp-sim'))
#
#     fig.add_heatmap(z=PLVemp, y=regionLabels, x=regionLabels, colorscale='Viridis', zmin=0, zmax=1, row=1, col=1, showscale=False)
#     fig.add_heatmap(z=PLVsim[0], colorscale='Viridis', zmin=0, zmax=1, row=1, col=2, showscale=False)
#     fig.add_scatter(x=PLVemp[np.triu_indices(len(conn.region_labels), 1)],
#                     y=PLVsim[0][np.triu_indices(len(conn.region_labels), 1)], mode='markers', marker=dict(color=colors),
#                     row=1, col=3, showlegend=False)
#     fig.add_scatter(x=[1], y=[1], text=["r = " + str(np.round(PLVr[0], 4))], mode='text', row=1, col=3, showlegend=False)
#     fig.add_scatter(x=np.linspace(0, 1, 100), y=PLV_rl[0], mode='lines', marker=dict(color='red'), row=1, col=3, showlegend=False)
#
#
#     fig.update_xaxes(range=[-0.1, 1.1], row=1, col=3)
#     fig.update_yaxes(range=[-0.1, 1.1], row=1, col=3)
#
#
#     number_frames = len(coupling_vals)
#     frames = [dict(
#         name=k,
#         data=[go.Heatmap(z=PLVsim[k]),
#               go.Scatter(x=PLVemp[np.triu_indices(len(conn.region_labels), 1)],
#                          y=PLVsim[k][np.triu_indices(len(conn.region_labels), 1)], mode='markers',
#                          marker=dict(color=colors)),  # update the second trace in (1,2) scatter plot
#               go.Scatter(x=[1], y=[1], text=["r = " + str(np.round(PLVr[k], 4))], mode='text'),
#               # update the trace in (1,2) Text - r= 0.3
#               go.Scatter(x=np.linspace(0, 1, 100), y=PLV_rl[k], mode='lines', marker=dict(color='red')),
#               ],
#         traces=[1, 2, 3, 4]  # the elements of the list [0,1,2] give info on the traces in fig.data
#         # that are updated by the above three go.Scatter instances
#         ) for k in range(number_frames)]
#
#
#     updatemenus = [dict(type='buttons',
#                         buttons=[dict(label='Play',
#                                       method="animate",
#                                       args=[[f'{k}' for k in range(number_frames)],
#                                              dict(frame=dict(duration=1500, redraw=False),
#                                                   transition=dict(duration=0),
#                                                   easing='quadratic-in-out',
#                                                   fromcurrent=True,
#                                                   mode='immediate')]),
#                                  dict(label='Pause',
#                                       method="animate",
#                                       args=[[None],
#                                              dict(frame=dict(duration=0, redraw=False),
#                                                   transition=dict(duration=0),
#                                                   mode='immediate')])],
#                         direction= 'left',
#                         pad=dict(r= 10, t=85),
#                         showactive =False, x= 0.4, y= 0, xanchor= 'right', yanchor= 'top')
#                 ]
#
#
#     sliders = [{'yanchor': 'top',
#                 'xanchor': 'left',
#                 'currentvalue': {'font': {'size': 11}, 'prefix': 'Coupling factor id: ', 'visible': False, 'xanchor': 'right'},
#                 'transition': {'duration': 300.0, 'easing': 'cubic-in-out'},
#                 'pad': {'b': 10, 't': 50},
#                 'len': 0.65, 'x': 0.4, 'y': 0,
#                 'steps': [{'args': [[k], {'frame': {'duration': 50.0, 'easing': 'quadratic-in-out', 'redraw':True},
#                                           'transition': {'duration': 0, 'easing': 'quadratic-in-out'}}],
#                            'label': k, 'method': "animate"} for k in range(number_frames)
#                         ]}]
#
#     fig.update(frames=frames)
#
#     fig.update_layout(updatemenus=updatemenus,
#                       sliders=sliders);
#
#     pio.write_html(fig, file="figures/triPLVemp-sim.html", auto_open=True)

## DUOPLOT
def duoplot(emp_matrix, sim_matrix, matrix_r, matrix_rl, regionLabels, colors, title=None, folder='figures'):

    # Create matrixduo: half empirical half simulated
    matrixduo = sim_matrix.copy()
    for i in range(len(sim_matrix)):
        matrixduo[i][np.tril_indices(len(emp_matrix), 0)]=emp_matrix[np.tril_indices(len(emp_matrix), 0)]

    # Plot it
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title+' emp-sim', 'r' + title + ' emp-sim'))

    fig.add_heatmap(z=matrixduo[0], y=regionLabels, x=regionLabels, colorscale='Viridis', zmin=0, zmax=1, row=1, col=1, showscale=False)
    fig.add_scatter(x=emp_matrix[np.triu_indices(len(emp_matrix), 1)],
                    y=sim_matrix[0][np.triu_indices(len(emp_matrix), 1)], mode='markers', marker=dict(color=colors),
                    row=1, col=2, showlegend=False)
    fig.add_scatter(x=[1], y=[1], text=["r = " + str(np.round(matrix_r[0], 4))], mode='text', row=1, col=2, showlegend=False)
    fig.add_scatter(x=np.linspace(0, 1, 100), y=matrix_rl[0], mode='lines', marker=dict(color='red'), row=1, col=2, showlegend=False)


    fig.update_xaxes(range=[-0.1, 1.1], row=1, col=2)
    fig.update_yaxes(range=[-0.1, 1.1], row=1, col=2)


    number_frames = len(coupling_vals)
    frames = [dict(
        name=k,
        data=[go.Heatmap(z=matrixduo[k]),
              go.Scatter(x=emp_matrix[np.triu_indices(len(emp_matrix), 1)],
                         y=sim_matrix[k][np.triu_indices(len(emp_matrix), 1)], mode='markers',
                         marker=dict(color=colors)),  # update the second trace in (1,2) scatter plot
              go.Scatter(x=[1], y=[1], text=["r = " + str(np.round(matrix_r[k], 4)) + "<br>g = " + str(coupling_vals[k])], mode='text'),
              # update the trace in (1,2) Text - r= 0.3
              go.Scatter(x=np.linspace(0, 1, 100), y=matrix_rl[k], mode='lines', marker=dict(color='red')),
              ],
        traces=[0, 1, 2, 3]  # the elements of the list [0,1,2] give info on the traces in fig.data
        # that are updated by the above three go.Scatter instances
        ) for k in range(number_frames)]


    updatemenus = [dict(type='buttons',
                        buttons=[dict(label='Play',
                                      method="animate",
                                      args=[[f'{k}' for k in range(number_frames)],
                                             dict(frame=dict(duration=1500, redraw=True),
                                                  transition=dict(duration=0),
                                                  easing='elastic-in-out',
                                                  fromcurrent=True,
                                                  mode='immediate')]),
                                 dict(label='Pause',
                                      method="animate",
                                      args=[[None],
                                             dict(frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0),
                                                  mode='immediate')])],
                        direction='left',
                        pad=dict(r=10, t=85),
                        showactive=False, x=0.6, y=0, xanchor='right', yanchor='top')
                ]


    sliders = [{'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {'font': {'size': 11}, 'prefix': 'Coupling factor id: ', 'visible': False, 'xanchor': 'right'},
                'transition': {'duration': 1000.0, 'easing': 'elastic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.45, 'x': 0.6, 'y': 0,
                'steps': [{'args': [[k], {'frame': {'duration': 50.0, 'easing': 'elastic-in-out', 'redraw': True},
                                          'transition': {'duration': 0, 'easing': 'elastic-in-out'}}],
                           'label': k, 'method': "animate"} for k in range(number_frames)
                        ]}]

    fig.update(frames=frames)

    fig.update_layout(updatemenus=updatemenus,
                      sliders=sliders);

    pio.write_html(fig, file=folder+"/duo"+title+"_emp-sim.html", auto_open=True)

duoplot(PLVemp,PLVsim,PLVr,regLines(PLVemp, PLVsim), regionLabels, colors,"PLV (alpha band)", specific_folder)
# duoplot(AECemp,AECsim,AECr,regLines(AECemp, AECsim), colors, "AEC (alpha band)", specific_folder)

# PLV_triplot(PLVemp,PLVsim,PLVr,PLV_rl)






# # Plot
# fig = go.Figure(data=go.Heatmap(z=PLVemp, x=regionLabels[indexes], y=regionLabels[indexes], colorscale='Viridis'))
# fig.update_layout(title='Phase Locking Value')
# pio.write_html(fig, file="figures/PLV_" + subjectid + ".html", auto_open=True)
#
# #### dynamical plv sim
# fig = go.Figure(data=go.Heatmap(z=PLVsim[0][:, indexes][indexes], x=regionLabels[indexes], y=regionLabels[indexes],
#                                 colorscale='Viridis', zmin=0, zmax=1),
#                 frames=[go.Frame(data=go.Heatmap(z=PLVsim[i][:, indexes][indexes])) for i in range(len(PLVsim))])
# fig.update_layout(
#     updatemenus=[
#         dict(type="buttons", visible=True,
#              buttons=[dict(label="Play", method="animate", args=[None])]
#              )])
# fig.show(renderer='browser')
#
# ### dynamical scatterplot
# # mount the dataframe from PLVr_df a df
#
# gs = list()
# sim = list()
# emp = list()
#
# for i, g in enumerate(coupling_vals):
#     gs += [g] * len(PLVr_df[0][0])
#     sim += list(PLVr_df[i][0])
#     emp += list(PLVr_df[i][1])
#
# temp1 = np.asarray([gs, sim, emp]).transpose()
# import pandas as pd
# import plotly.express as px
#
# PLVdf = pd.DataFrame(temp1, columns=["g", "sim", "emp"])
#
# fig = px.scatter(df, x="emp", y="sim", animation_frame="g", trendline="ols")
# fig.show(renderer="browser")




### Trying to rearrange FC matrix - by now not useful
# import scipy.cluster
# # a = cluster.hierarchy.linkage(conn.weights, method='average')
# # scipy.cluster.hierarchy.dendrogram(c)
# plot_matrix(plv_emp)
# new_order = scipy.cluster.hierarchy.fclusterdata(plv_emp, 1)
# # new_order=np.concatenate((np.arange(0, 92, 2), np.arange(1, 92, 2)))
# M=plv_emp[:, new_order][new_order]
# rl=regionLabels[new_order]
# fig = go.Figure(data=go.Heatmap(z=M, x=regionLabels, y=regionLabels, colorscale='Viridis'))
# fig.update_layout(title='Phase Locking Value')
# pio.write_html(fig, file="figures/PLV_" + subjectid + ".html", auto_open=True)
# import sklearn.cluster
# b=sklearn.cluster.AgglomerativeClustering(a, linkage='ward')
# for i in set(new_order):
#     temp = np.where(new_order == i)
# [np.where(new_order==i) for i in set(new_order)]


## try with px - doesnt work
# import plotly.express as px
# # you'll need to create a df with 1 col=g; 2col=img
# ddf=list()
# for i, g in enumerate(coupling_vals):
#     ddf.append([g, PLVsim[i]])
# ddf1=pd.DataFrame(ddf, columns=["g","m"])
# f=px.imshow(ddf1, z="m", animation_frame='g')
# f.show(renderer="browser")


# ## GATHER RESULTS
# simname = subjectid+"-"+emp_subj+"-t"+str(simLength)+"-"+time.strftime("m%md%dy%Y")
# # Working on FFT peak results
# df1 = pd.DataFrame(results_fft_peak, columns=["G", "speed",  "mS_peak", "mS_module", "allSignals"])
#
# df1.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)
#
# dfFFT_m = df1.groupby(["G", "speed"])[["G", "speed", "mS_peak"]].mean()
# # Load previously gathered data
# # df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")
# # df1a=df1[df1.G<33]
# paramSpace(df1, title=simname, folder=specific_folder)
#
# # Working on FC results
# df = pd.DataFrame(results_fc, columns=["G", "speed", "round", "plvD_r",  "plvT_r", "plvA_r", "plvB_r", "plvG_r"])
#                             # columns=["G", "speed", "plvD_r", "pliD_r", "aecD_r", "plvT_r","pliT_r", "aecT_r","plvA_r",
#                                   #"pliA_r","aecA_r", "plvB_r","pliB_r","aecB_r", "plvG_r","pliG_r", "aecG_r"])
#
# df.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)
#
#
# dfPLV_m = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
# dfPLV_sd = df.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].std().reset_index()
#
# dfPLV_m.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# dfPLV_sd.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
#
# # dfPLI = df[["G", "speed", "pliD_r", "pliT_r", "pliA_r", "pliB_r", "pliG_r"]]
# # dfPLI.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
# # dfAEC = df[["G", "speed", "aecD_r", "aecT_r", "aecA_r", "aecB_r", "aecG_r"]]
# # dfAEC.columns=["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]
#
#
# paramSpace(dfPLV_m, 0.5, emp_subj+subjectid+"_PLV", folder=specific_folder)
# paramSpace(dfPLV_sd, 0.5, emp_subj+subjectid+"_PLV_sd", folder=specific_folder)
#
# # paramSpace(dfPLI, 0.5, subjectid+"_PLI", folder=specific_folder)
# # paramSpace(dfAEC, 0.5, subjectid+"_AEC", folder=specific_folder)
#
# # df.to_csv("loop0-2m.csv", index=False)
# # b=df.sort_values(by=["noise", "G"])
# # fil=df[["G", "noise", "plv_avg","aec_avg","Tavg"]]
# # fil=fil.sort_values(by=["noise", "G"])
#
#
#
