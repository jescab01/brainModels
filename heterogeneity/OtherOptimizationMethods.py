import time
import numpy as np
import pandas as pd
from mne import filter
import scipy

from fooof import FOOOF
import plotly.express as px
import plotly.offline
from plotly.subplots import make_subplots
import plotly.graph_objects as go  # for data visualisation

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N

from toolbox import PLV, epochingTool, multitapper, timeseriesPlot


##

def signalsJRD(params, individual, verbose=True):

    ctb_folder = params["ctb_folder"]
    emp_subj = params["emp_subj"]
    samplingFreq = params["samplingFreq"]  #Hz
    simLength = params["simLength"]  # ms - relatively long simulation to be able to check for power distribution
    transient = params["transient"]  # seconds to exclude from timeseries due to initial transient
    structure = params["structure"]

    # Unpack individual
    g, s, w = individual[0], individual[1], individual[2]

    conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+structure+".zip")
    conn.weights = conn.scaled_weights(mode="tract")

    p_array=np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L' ), 0.22, 0)
    sigma_array=np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L' ), 0.022, 0)

    m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),   # SLOW population
                              tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                              He2=np.array([3.25]), Hi2=np.array([22]),   # FAST population
                              tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
                              w=w, c=np.array([135.0]),

                              c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                              c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                              v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                              p=np.array([p_array]), sigma=np.array([sigma_array]))

    ## Remember to hold tau*H constant.
    m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
    m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    #integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

    if params["sc_subset"]:
        sc_rois = params["sc_subset"]
        conn.weights = conn.weights[:, sc_rois][sc_rois]
        conn.tract_lengths = conn.tract_lengths[:, sc_rois][sc_rois]
        conn.region_labels = conn.region_labels[sc_rois]

    tic0 = time.time()
    if verbose:
        print("Simulating for g=%i - s=%0.2f" % (g, s))

    # Coupling function
    coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
    conn.speed = np.array([s])

    mon = (monitors.Raw(),)

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=simLength)
    if verbose:
        print("Simulation time: %0.2f sec" % (time.time() - tic0,))
    # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
    # Mount PSP output as: w * (vExc1 - vInh1) + (1 - w) * (vExc2 - vInh2)
    raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
               (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)

    return raw_data


def fc_fit(params, raw_data):
    ctb_folder = params["ctb_folder"]
    emp_subj = params["emp_subj"]
    structure = params["structure"]
    sc_subset = params["sc_subset"]
    fc_subset = params["fc_subset"]

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + structure + ".zip")
    if sc_subset:
        conn.region_labels = conn.region_labels[sc_subset]

    results = list()
    bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, params["samplingFreq"], lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, params["samplingFreq"], "signals", verbose=False)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # CONNECTIVITY MEASURES
        ## PLV
        plv = PLV(efPhase, verbose=False)
        # aec = AEC(efEnvelope)

        # Load empirical data to make simple comparisons
        if fc_subset:
            plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "\\" + bands[0][b] + "_plv.txt")[:, fc_subset][
                fc_subset]
        else:
            plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "\\" + bands[0][b] + "_plv.txt")

        # Comparisons
        t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
        t1[0, :] = plv[np.triu_indices(len(conn.region_labels), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        # t2 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
        # t2[0, :] = aec[np.triu_indices(len(conn.region_labels), 1)]
        # t2[1, :] = aec_emp[np.triu_indices(len(conn.region_labels), 1)]
        # aec_r = np.corrcoef(t2)[0, 1]

        results.append(plv_r)

    return results


def cost_function(params, signals, rule="alpha_vs_all", save_curves=False):

    ctb_folder = params["ctb_folder"]
    emp_subj = params["emp_subj"]
    samplingFreq = params["samplingFreq"]  #Hz
    fc_subset = params["fc_subset"]

    # Calculate simulated spectra
    sim_freqs, sim_spectra = multitapper(signals, samplingFreq, epoch_length=4, ntapper=4, smoothing=0.5, plot=False)

    try:  # Load empirical spectra
        if fc_subset:
            emp_spectra = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatSpectra.txt')[fc_subset, :]
            gaussian_means = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_gaussianMean.txt')[fc_subset]
        else:
            emp_spectra = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatSpectra.txt')
            gaussian_means_emp = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_gaussianMean.txt')

        emp_freqs = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatFreqs.txt')

    except:
        print('Did you calculate spectra for the current subject?')

    # Pointwise cost
    emp_full_integral = np.trapz(emp_spectra, emp_freqs)
    sim_full_integral = np.trapz(sim_spectra, sim_freqs)
    # Use the differences point to point between spectra to have another measure of cost: use it as stop criteria.
    # these indexes cut simulated spectra to the length of empirical one
    low_idx = int(np.where(emp_freqs[0] == sim_freqs)[0])
    high_idx = int(np.where(emp_freqs[-1] == sim_freqs)[0])

    norm_emp_spectra = emp_spectra / emp_full_integral[:, np.newaxis]
    norm_sim_spectra = sim_spectra[:, low_idx:high_idx + 1] / sim_full_integral[:, np.newaxis]

    pointwise_cost = np.sum(abs(norm_sim_spectra - norm_emp_spectra), axis=1)

    if rule == "corrcoef":

        emp_curve = emp_spectra
        emp_curve = scipy.signal.savgol_filter(emp_curve, 7, 3)
        emp_curve[emp_curve < 0] = 0  # truncado

        sim_curve = sim_spectra
        sim_curve = scipy.signal.savgol_filter(sim_curve, 7, 3)

        cost = 1 - np.array([np.corrcoef(emp_curve[roi], sim_curve[roi, low_idx:high_idx + 1].T)[1, 0] for roi in range(len(sim_curve))])
        if save_curves:
            return np.average(cost), pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
        else:
            return np.average(cost), pointwise_cost

    elif rule == "alpha_vs_all":
        # Now, calculate integrals: all curve vs until alpha curve. Used until 28 Oct.
        emp_2alpha_integral = np.trapz(emp_spectra[:, (2 <= emp_freqs) & (emp_freqs <= 12)], emp_freqs[(2 <= emp_freqs) & (emp_freqs <= 12)])
        sim_2alpha_integral = np.trapz(sim_spectra[:, (2 <= sim_freqs) & (sim_freqs <= 12)], sim_freqs[(2 <= sim_freqs) & (sim_freqs <= 12)])

        # COST: Calculates fit between pre-alpha, post-alpha proportions in empirical and simulated spectra.
        # A positive cost means, frequency and "w" must raise.
        # This value can be directly used in the gradient descent as a positive value implies
        # the simulation need to raise its frequency (lowering down "w").
        fft_cost = sim_2alpha_integral / sim_full_integral - emp_2alpha_integral / emp_full_integral

        if save_curves:
            return fft_cost, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
        else:
            return fft_cost, pointwise_cost


    elif rule == "pointwise_comparison":
        # New approach (28 oct). Use the mean of the gaussian model as reference and check the differences point to point
        # between the spectra to see where the error is: above or below the mean, then move w consequently.
        mean_idx = [np.argmin(np.abs(emp_freqs - m)) for m in gaussian_means_emp]
        norm_spectra_diff = norm_sim_spectra - norm_emp_spectra

        sum_preMean = np.array([sum(norm_spectra_diff[i, :mean_i]) for i, mean_i in enumerate(mean_idx)])
        sum_postMean = np.array([sum(norm_spectra_diff[i, mean_i:]) for i, mean_i in enumerate(mean_idx)])

        # np.vstack((sum_preMean, um_postMean)).T  # Visualization purposes

        diff = sum_preMean - sum_postMean  # if positive -> move to higher frequencies (i.e. lower w)

        if save_curves:
            return diff, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
        else:
            return diff, pointwise_cost

    elif rule == "fooof":
        ## Use foof to model the simulated spectra and define whether it should move up or down.
        fm = FOOOF(max_n_peaks=1)  # peak_width_limits=[2.0, 10.0]
        # Define frequency range across which to model the spectrum
        freq_range = [5, 40]  # Consider that multitapper uses the range [2, 45]

        gaussian_means_sim = []
        for roi in range(len(sim_spectra)):
            # Model the power spectrum with FOOOF, and print out a report
            fm.fit(sim_freqs, sim_spectra[roi], freq_range)
            # Calculate gaussian mean
            if np.isnan(fm.get_params('peak_params', 'CF')):
                gaussian_means_sim.append(0)
            else:
                gaussian_means_sim.append(fm.get_params('peak_params', 'CF'))


        # positive cost implies lowering down 'w'
        cost = gaussian_means_emp - np.array(gaussian_means_sim)

        if save_curves:
            return cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
        else:
            return cost


def JRDfit(w):

    history["theta"].append(w)
    n_rep = 3

    print(w)
    g, s = 112, 15

    # w[w <= 0] = 0
    # w[w >= 1] = 1

    individual = list((g, s, w))


    signals = [signalsJRD(params, individual, False) for i in range(n_rep)]
    post_rFC = np.average([fc_fit(params, signals_i) for signals_i in signals], axis=0)
    print("FC fit: %0.5f " % np.average(post_rFC))
    history["fc"].append(post_rFC)

    cost_fft, cost_pw, curves_emp, curves_sim_fft, curves_sim_freq = [], [], [], [], []

    for signals_i in signals:
        cost = cost_function(params, signals_i, rule="corrcoef", save_curves=True)
        cost_fft.append(cost[0])
        cost_pw.append(cost[1])
        curves_sim_fft.append(cost[2][0])
        curves_sim_freq.append(cost[2][1])
        # curves_emp = cost[2][2:]

    cost_fft = np.average(cost_fft, axis=0)
    cost_pw = np.average(cost_pw, axis=0)
    history["cost"].append([cost_fft, cost_pw])

    curves_sim_fft = np.average(curves_sim_fft, axis=0)
    curves_sim_freq = np.average(curves_sim_freq, axis=0)


    print("Global fft cost: %0.5f" % np.average(cost_fft))
    print("Global pw cost: %0.5f" % np.average(cost_pw))

    history["cost"].append([cost_fft, cost_pw])

    # curves_sim_fft = np.average(curves_sim_fft, axis=0)
    # curves_sim_freq = np.average(curves_sim_freq, axis=0)
    # history["curves"].append(
    #     [curves_sim_fft, curves_sim_freq])  # Curves[0]=sim_spectra; [1]sim_freqs; [2]emp_spectra; [3]emp_freqs
    # if it == 0:  # in the first iteration save empirical spectra and freqs
    #     history["curves_emp"] = curves_emp

    return np.average(cost_fft)


history = {"theta": list(), "cost": list(), "curves": list(), "curves_emp": list(), "fc": list()}


# Indexes ROI for DMN in AAL2red structure: Frontal_Sup; ACC; PCC; Parietal_inf; Precuneus; Temporal_Mid.
AAL2red_rois_dmn = [2, 3, 34, 35, 38, 39, 64, 65, 70, 71, 84, 85]

## Optimizee global parameters
params = {
    "mode": "FC",  # FC, FFT. Whenever FFT gradient descent will be applied.

    "simLength": 10000, #44000
    "repetitions": 3,
    "transient": 2000,
    "samplingFreq": 1000,

    "emp_subj": "NEMOS_035",
    "structure": "_AAL2red",
    "sc_subset": None,  # AAL2red_rois_dmn,  # Subsets over structure
    "fc_subset": None,  # AAL2red_rois_dmn,  # Subsets over AAL2red (main FC output shape from brainstorm).
    "ctb_folder": "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\",
    "verbose": False}

# Calculate number of rois
conn = connectivity.Connectivity.from_file(params["ctb_folder"] + params["emp_subj"] + params["structure"]+".zip")
if params["sc_subset"]:
    n_rois = len(params["sc_subset"])
else:
    n_rois = len(conn.region_labels)

# Define region labels
if params["sc_subset"]:
    regionLabels = conn.region_labels[params["sc_subset"]]
else:
    regionLabels = conn.region_labels

# ## SET UP SIMULATION
# iterations = 200
# fit_method = "alpha_vs_all"
# learning_rate = {"alpha_vs_all": 5, "pointwise_comparison": 1, "fooof": 0.1}
# show_dynFFTs = True

w = np.asarray([0.8]*n_rois)

bounds = [(0, 1) for i in range(n_rois)]

res = scipy.optimize.minimize(JRDfit, w, bounds=bounds)


# tic0 = time.time()
# print("Initializing Gradient Descent Algortihm...")
# individual = list((g, s, w))
# history = {"theta": list(), "cost": list(), "curves": list(), "curves_emp":list(), "fc": list()}
# theta = w
#
# for it in range(iterations):
#     tic = time.time()
#
#     print('\n\n Iteration %i  -  ' % it, end="")
#     history["theta"].append(theta.T)
#     signals = [signalsJRD(params, individual, verbose=True) for i in range(params["repetitions"])]
#
#     if it == 0:
#         pre_rFC = np.average([fc_fit(params, signals_i) for signals_i in signals])
#         history["fc"].append(pre_rFC)
#         print("A priori rFC: ", end="")
#         print(pre_rFC)
#         print("pre-avg: %0.4f" % np.average(pre_rFC))
#
#     else:
#         post_rFC = np.average([fc_fit(params, signals_i) for signals_i in signals])
#         history["fc"].append(post_rFC)
#         print("A priori rFC: ", end="")
#         print(pre_rFC)
#         print("Post-fit rFC: ", end="")
#         print(post_rFC)
#         print("pre-avg: %0.4f | post-avg: %0.4f" % (np.average(pre_rFC), np.average(post_rFC)))
#
#     ### Calculate a relative difference (low frequencies vs all) in emp-sim spectra per roi
#     # "alpha_vs_all"; "pointwise_comparison"; "foof"
#     cost_fft = []
#     cost_pw = []
#     curves_emp = []
#     curves_sim_fft = []
#     curves_sim_freq = []
#     for signals_i in signals:
#         cost = cost_function(params, signals_i, rule=fit_method, save_curves=True)
#         cost_fft.append(cost[0])
#         cost_pw.append(cost[1])
#         curves_sim_fft.append(cost[2][0])
#         curves_sim_freq.append(cost[2][1])
#         curves_emp = cost[2][2:]
#
#     cost_fft = np.average(cost_fft, axis=0)
#     cost_pw = np.average(cost_pw, axis=0)
#     history["cost"].append([cost_fft, cost_pw])
#
#     curves_sim_fft = np.average(curves_sim_fft, axis=0)
#     curves_sim_freq = np.average(curves_sim_freq, axis=0)
#     history["curves"].append([curves_sim_fft, curves_sim_freq])  # Curves[0]=sim_spectra; [1]sim_freqs; [2]emp_spectra; [3]emp_freqs
#     if it == 0:   # in the first iteration save empirical spectra and freqs
#         history["curves_emp"] = curves_emp
#
#     d_cost, d_pw = cost_derivative(history)  # Stop mechanism when max pw fit is achieved; after that undesired gamma peak appears. Avoid this.
#     cost_fft[np.where((d_cost > 0) & (d_pw <= 0))] = - cost_fft[np.where((d_cost > 0) & (d_pw <= 0))]
#
#     theta = theta - (1/len(w)) * learning_rate[fit_method] * cost_fft  # len(w) as "m"
#
#
#     theta[theta <= 0] = 0
#     theta[theta >= 1] = 1
#
#     individual[2] = theta
#
#     print(' cost %0.2f  -  time: %0.2f/%0.2f' % (np.average(np.abs(cost_fft)), time.time()-tic, time.time()-tic0, ))
#
#     if (it > 1) & (it % dyn_check != 0) & (it % check_point == 0):
#         show_evolution(history, regionLabels)
#     elif (it > 1) & (it % dyn_check == 0) & (it % check_point == 0):
#         show_evolution(history, regionLabels, True)
#
#
#
#     ###### MORE plots
#         # fig = px.line(x=range(len(cost_history)), y=np.average(cost_history, axis=1))
#         # fig.show(renderer="browser")
#         # # plotly.offline.iplot(fig)
#         #
#         #
#         # if sc_subset:
#         #     fig = px.scatter(x=conn.region_labels[sc_subset], y=cost_fft)
#         #     fig.add_scatter(x=conn.region_labels[sc_subset], y=theta, name=w)
#         # else:
#         #     fig = px.scatter(x=conn.region_labels, y=cost_fft)
#         #     fig.add_scatter(x=conn.region_labels, y=theta, name=w)
#         # fig.update_yaxes(range=[-0.2, 1])
#         # fig.show(renderer="browser")
#         # # plotly.offline.iplot(fig)
#         # print(cost_fft)
#         #
#         # print("Plotting FFTs...")
#         # sim_spectra, sim_freqs = curves[0], curves[1]
#         # emp_spectra, emp_freqs = curves_emp[0], curves_emp[1]
#         #
#         # fig = go.Figure()
#         # for specific_roi in range(5):
#         #     fig.add_scatter(x=sim_freqs, y=sim_spectra[specific_roi] / np.trapz(sim_spectra, sim_freqs)[specific_roi],
#         #                     legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " sim")
#         #     fig.add_scatter(x=emp_freqs, y=emp_spectra[specific_roi] / np.trapz(emp_spectra, emp_freqs)[specific_roi],
#         #                     legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " emp")
#         # fig.show(renderer="browser")
#         # plotly.offline.iplot(fig)
#         #
#         # print("Plotting signals...")
#         # timeseriesPlot(signals - np.average(signals, axis=1)[:, None], np.arange(transient, simLength, 1), regionLabels,
#         #                mode="html")
#
#
