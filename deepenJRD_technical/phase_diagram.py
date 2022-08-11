
import time
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append("E:\\LCCN_Local\PycharmProjects\\")
from toolbox.signals import timeseriesPlot, epochingTool
from toolbox.fc import PLV
from toolbox.fft import FFTpeaks
# "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\NEMOS_035_AAL2.zip"
# "E:\LCCN_Local\PycharmProjects\brainModels\deepenJRD_technical\Precentrals_connected.zip"

import warnings
warnings.filterwarnings("ignore")


def simulate(params, init_cond=None):

    # Prepare simulation parameters
    simLength = params["simLength"] * 1000  # ms
    samplingFreq = params["samplingFreq"]  # Hz

    model, g, s, out = params["model"], params["g"], params["s"], params["out"]
    w, p, sigma = params["w"], params["p"], params["sigma"]


    # STRUCTURAL CONNECTIVITY      #########################################

    conn = connectivity.Connectivity.from_file("E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\NEMOS_035_AAL2.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # NEURAL MASS MODEL    &    COUPLING FUNCTION         #########################################

    if "jrd" == model:  # JANSEN-RIT-DAVID
        # Parameters edited from David and Friston (2003).
        m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                               tau_e1=np.array([10]), tau_i1=np.array([20]),
                               He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                               tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                               w=np.array([w]), c=np.array([135.0]),
                               c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                               c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                               v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                               p=np.array([p]), sigma=np.array([sigma]),

                               variables_of_interest=["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                                                      "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"])

        coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

        # # Remember to hold tau*H constant: Spiegler (2010) pp.1045;
        m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    elif model == "jr":  # JANSEN-RIT
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),

                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                          c=np.array([135.0]), p=np.array([p]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),

                          variables_of_interest=["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"])

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
        # Remember to hold tau*H constant.
        m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])

    conn.speed = np.array([s])

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    tic = time.time()
    print("Simulating %s (%is)  || structure: %s \nPARAMS: g%i s%i w%0.2f p%0.4f sigma%0.4f" %
          (params["model"], params["simLength"], params["structure"][30:],
           params["g"], params["s"], params["w"], params["p"], params["sigma"]))

    # Run simulation
    if init_cond is not None:
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                  initial_conditions=init_cond)
    else:
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)

    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][time,state_vars,channels,modes].T" where: a=monitorIndex, b=(data:1,time:0) and
    # state_vars are 6 in JR: X0 (pyr mV), X1 (exc mV), X2  (inh mV), x3 (pyr deriv), x4 (exc deriv), x5 (inh deriv).
    roi = 0
    if params["model"] == "jr":
        psp_t = output[0][1][params["transient"]:, 1, roi, 0].T - output[0][1][params["transient"]:, 2, roi, 0].T
        psp_dt = output[0][1][params["transient"]:, 4, roi, 0].T - output[0][1][params["transient"]:, 5, roi, 0].T

    elif params["model"] == "jrd":
        psp_t = params["w"] * (
                    output[0][1][params["transient"]:, 1, roi, 0].T - output[0][1][params["transient"]:, 2, roi, 0].T) + \
                (1 - params["w"]) * (
                            output[0][1][params["transient"]:, 7, roi, 0].T - output[0][1][params["transient"]:, 8, roi,
                                                                              0].T)
        psp_dt = params["w"] * (
                    output[0][1][params["transient"]:, 4, roi, 0].T - output[0][1][params["transient"]:, 5, roi, 0].T) + \
                 (1 - params["w"]) * (
                             output[0][1][params["transient"]:, 10, roi, 0].T - output[0][1][params["transient"]:, 11,
                                                                                roi, 0].T)

    print("Required time %0.3f seconds.\n\n" % (time.time() - tic,))

    return output, conn.region_labels, psp_t, psp_dt


params = {"g": 20, "s": 10, "structure": "E:\LCCN_Local\PycharmProjects\\brainModels\deepenJRD_technical\Precentrals_connected.zip",
          "simLength": 10, "samplingFreq": 1000, "transient": 2000, "out": "psp",
          "model": "jr", "w": 1, "p": 0.1085, "sigma": 0}


# Simulate once to fix INITIAL CONDITIONS
print("INITIAL CONDITIONS _ long enough to remove initial transients")
params["model"] = "jr"
params["simLength"] = 10  # s
output, labels, _, _ = simulate(params)
init_cond = output[0][1]


## PHASE DIAGRAMS
print("PHASE DIAGRAMS _ ")

## JR
params["model"] = "jr"
params["p"] = 0.2
params["g"] = 0
params["simLength"] = 2  # s
params["transient"] = 0

output, labels, psp_t, psp_dt = simulate(params, init_cond)

fig = px.scatter_3d(x=output[0][0][params["transient"]:], y=psp_dt, z=psp_t)
fig.update_traces(marker_size=3)
fig.show(renderer="browser")

params["sigma"] = 0.022

output, labels, psp_t, psp_dt = simulate(params, init_cond)

fig = px.scatter_3d(x=output[0][0][params["transient"]:], y=psp_dt, z=psp_t)
fig.update_traces(marker_size=3)
fig.show(renderer="browser")

# df = pd.DataFrame(np.asarray([psp_t, psp_dt, output[0][0][params["transient"]:]]).T, columns=["psp_t", "psp_dt", "time"])
#
# fig = px.scatter(df, x="psp_dt", y="psp_t", animation_frame="time", opacity=1)
# fig.update_traces(marker_size=15)
# fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 5
# fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
# fig.add_trace(go.Scatter(x=psp_dt, y=psp_t, marker_color="slategray", line=dict(width=4), opacity=0.9))
# fig.add_trace(go.Scatter(x=psp_dt[-1:], y=psp_t[-1:], mode="markers", marker=dict(size=10, color="black")))
# fig.add_trace(go.Scatter(x=psp_dt[:1], y=psp_t[:1], mode="markers", marker=dict(size=10, color="steelblue")))
# fig.show(renderer="browser")




## JRD
params["model"] = "jrd"
params["w"] = 0.8
params["p"] = 0.2
params["sigma"] = 0.022
params["g"] = 0
params["simLength"] = 2  # s
params["transient"] = 0

output, labels, psp_t, psp_dt = simulate(params, np.concatenate((init_cond, np.zeros(init_cond.shape)), axis=1))

fig = px.scatter_3d(x=output[0][0][params["transient"]:], y=psp_dt, z=psp_t)
fig.update_traces(marker_size=3)
fig.show(renderer="browser")


# df = pd.DataFrame(np.asarray([psp_t, psp_dt, output[0][0][params["transient"]:]]).T, columns=["psp_t", "psp_dt", "time"])
# fig = px.scatter(df, x="psp_dt", y="psp_t", animation_frame="time")
# fig.update_traces(marker_size=15)
# fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1
# fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1
# fig.add_trace(go.Scatter(x=psp_dt, y=psp_t, marker_color="slategray", line=dict(width=4)))
# fig.add_trace(go.Scatter(x=psp_dt[-1:], y=psp_t[-1:], mode="markers", marker=dict(size=10, color="black")))
# fig.add_trace(go.Scatter(x=psp_dt[:1], y=psp_t[:1], mode="markers", marker=dict(size=10, color="steelblue")))
# fig.show(renderer="browser")


### BIFURCATION DIAGRAMS


## JR
params["model"] = "jr"
params["p"] = 0.2
params["sigma"] = 0
params["g"] = 0
params["simLength"] = 5  # s
params["transient"] = 4000  # ms

p_vals = np.arange(0, 0.5, 0.01)

bifurcation = []
bifurcation3d = []

for i, p in enumerate(p_vals):

        print("Simulating p%0.4f  -  %i/%i" % (p, i, len(p_vals)))

        params["p"] = p

        _, _, psp_t, psp_dt = simulate(params)

        # 2D
        bifurcation.append([p, np.max(psp_t),  np.min(psp_t)])

        # 3D
        bifurcation3d.append([[p]*len(psp_dt), psp_dt, psp_t])



bifurcation = np.asarray(bifurcation)

fig = px.line(x=bifurcation[:, 0], y=bifurcation[:, 1])
fig.add_trace(go.Scatter(x=bifurcation[:, 0], y=bifurcation[:, 2], mode="lines"))
fig.show(renderer="browser")


ps, psp_t, psp_dt = [], [], []
for id in range(len(bifurcation3d)):
    ps.append(bifurcation3d[id][0])
    psp_t.append(bifurcation3d[id][1])
    psp_dt.append(bifurcation3d[id][2])

ps = np.asarray(ps).flatten()
psp_t = np.asarray(psp_t).flatten()
psp_dt = np.asarray(psp_dt).flatten()

fig = px.scatter_3d(x=ps, y=psp_t, z=psp_dt)
fig.update_traces(marker_size=2)
fig.show(renderer="browser")





## Trying to extract just one circle: done but not averaged.

# from scipy.signal import argrelextrema
# id_start = argrelextrema(np.abs(psp_dt), np.less)[0][-3]
# id_end = argrelextrema(np.abs(psp_dt), np.less)[0][-1]
#
# id_start1 = argrelextrema(np.abs(psp_dt), np.less)[0][-5]
# id_end1 = argrelextrema(np.abs(psp_dt), np.less)[0][-3]
#
# fig = px.scatter(x=psp_dt[id_start:id_end], y=psp_t[id_start:id_end])
# fig.add_trace(go.Scatter(x=psp_dt[id_start1:id_end1], y=psp_t[id_start1:id_end1], mode="markers"))
# fig.show(renderer="browser")














# # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
# IAF, module, band_module = FFTpeaks(data, simLength - transient)
#
# # bands = [["3-alpha"], [(8, 12)]]
# bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]
#
# for b in range(len(bands[0])):
#     (lowcut, highcut) = bands[1][b]
#
#     # Band-pass filtering
#     filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)
#
#     # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
#     efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")
#
#     # Obtain Analytical signal
#     efPhase = list()
#     efEnvelope = list()
#     for i in range(len(efSignals)):
#         analyticalSignal = scipy.signal.hilbert(efSignals[i])
#         # Get instantaneous phase and amplitude envelope by channel
#         efPhase.append(np.angle(analyticalSignal))
#         efEnvelope.append(np.abs(analyticalSignal))
#
#     # Check point
#     # from toolbox import timeseriesPlot, plotConversions
#     # regionLabels = conn.region_labels
#     # timeseriesPlot(raw_data, raw_time, regionLabels)
#     # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)
#
#     # CONNECTIVITY MEASURES
#     ## PLV
#     plv = PLV(efPhase)
#
#     # ## PLE - Phase Lag Entropy
#     # ## PLE parameters - Phase Lag Entropy
#     # tau_ = 25  # ms
#     # m_ = 3  # pattern size
#     # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)
#
#     # Load empirical data to make simple comparisons
#     if "cb" in mode:
#         plv_emp = \
#             np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
#             FC_cb_idx][
#                 FC_cb_idx]
#     else:
#         plv_emp = \
#             np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
#             FC_cortex_idx][
#                 FC_cortex_idx]
#
#     # Comparisons
#     t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
#     t1[0, :] = plv[np.triu_indices(len(plv), 1)]
#     t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
#     plv_r = np.corrcoef(t1)[0, 1]
#
#     ## Gather results
#     result.append(
#         (emp_subj, mode, g, s, r, out, test_params, IAF[0], module[0], band_module[0], bands[0][b], plv_r))
#
# print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))




# mesh and apply simulation function for all that mesh


# quiver plots


# 3d cone plot
