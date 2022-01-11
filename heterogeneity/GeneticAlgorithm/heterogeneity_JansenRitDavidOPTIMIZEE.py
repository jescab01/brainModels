import time
import numpy as np
import pandas as pd
from mne import filter
import scipy

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N

from toolbox import PLV, epochingTool, multitapper


def fitness_JRD(individual, optimizee_params, variables, verbose):

    if optimizee_params["mode"] == "FC":

        # Simulate n_sim times and gather signals
        signals = signalsJRD(optimizee_params, individual, variables, verbose)
        signals_amp_sd = [np.std(signals[i]) for i in range(len(signals))]
        if np.average(signals_amp_sd) < 0.01:
            rfc = 0
        else:
            rfc = fc_fit(optimizee_params, signals)
        return np.average(rfc)

    elif optimizee_params["mode"] == "FFT":
        # Call gradient descent algorithm that will call signals JRD and the fit function desired.
        w_updated, cost_fft, pre_rfc, post_rfc = gradient_descent(optimizee_params, individual, variables, verbose)

        return w_updated, np.average(pre_rfc), np.average(post_rfc)


def signalsJRD(optimizee_params, individual, variables, verbose=True):

    ctb_folder = optimizee_params["ctb_folder"]
    emp_subj = optimizee_params["emp_subj"]
    samplingFreq = optimizee_params["samplingFreq"]  #Hz
    simLength = optimizee_params["simLength"]  # ms - relatively long simulation to be able to check for power distribution
    transient = optimizee_params["transient"]  # seconds to exclude from timeseries due to initial transient
    structure = optimizee_params["structure"]

    # Unpack individual
    g, s, p, sigma = individual[0], individual[1], individual[2], individual[3]
    if "w" in variables.keys():
        w = individual[4:]
    else:
        w = np.array([0.8])

    conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+structure+".zip")
    conn.weights = conn.scaled_weights(mode="tract")

    m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),   # SLOW population
                              tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                              He2=np.array([3.25]), Hi2=np.array([22]),   # FAST population
                              tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
                              w=w, c=np.array([135.0]),

                              c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                              c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                              v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                              p=np.array([p]), sigma=np.array([sigma]))


    ## Remember to hold tau*H constant.
    m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
    m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    #integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

    if optimizee_params["sc_subset"]:
        sc_rois = optimizee_params["sc_subset"]
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


def fc_fit(optimizee_params, raw_data):
    ctb_folder = optimizee_params["ctb_folder"]
    emp_subj = optimizee_params["emp_subj"]
    structure = optimizee_params["structure"]
    sc_subset = optimizee_params["sc_subset"]
    fc_subset = optimizee_params["fc_subset"]

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + structure + ".zip")
    if sc_subset:
        conn.region_labels = conn.region_labels[sc_subset]

    results = list()
    bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, optimizee_params["samplingFreq"], lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, optimizee_params["samplingFreq"], "signals", verbose=False)

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


def gradient_descent(optimizee_params, individual, variables, verbose):
    """

    :param optimizee_params:
    :param g:
    :param s:
    :param p:
    :param sigma:
    :param w:
    :param verbose:
    :return:
    """

    # Unpack individual
    g, s, p, sigma, w = individual[0], individual[1], individual[2], individual[3], individual[4:]

    tic0 = time.time()
    if verbose:
        print("Initializing Gradient Descent Algortihm...")

    iterations = optimizee_params["GD_iterations"]
    learning_rate = optimizee_params["GD_learning_rate"]

    theta_history = np.zeros((iterations, len(w)))  # Theta will be the array of "w" to adjust.
    cost_history = np.zeros((iterations, len(w)))

    theta = w

    for it in range(iterations):

        tic = time.time()
        if verbose:
            print('Iteration %i  -  ' % it, end="")

        theta_history[it, :] = theta.T
        signals = signalsJRD(optimizee_params, individual, variables, verbose)
        if it == 0:
            signals_amp_sd = [np.std(signals[i]) for i in range(len(signals))]
            if np.average(signals_amp_sd) < 0.01:
                pre_rFC = 0
            else:
                pre_rFC = fc_fit(optimizee_params, signals)

        # Calculate a relative difference (low frequencies vs all) in emp-sim spectra per roi
        cost_fft = cost_fft_fit(optimizee_params, signals)
        cost_history[it, :] = cost_fft

        theta = theta - (1/len(w)) * learning_rate * cost_fft  # len(w) as "m"
        theta[theta <= 0] = 0
        theta[theta >= 1] = 1

        individual[4:] = theta

        if verbose:
            print(' cost %0.2f  -  time: %0.2f/%0.2f' % (np.average(np.abs(cost_fft)), time.time()-tic, time.time()-tic0, ))

    # Check there is any decent amount of signal | oscillation
    signals = signalsJRD(optimizee_params, individual, variables, verbose)
    signals_amp_sd = [np.std(signals[i]) for i in range(len(signals))]
    if np.average(signals_amp_sd) < 0.01:
        post_rFC = 0
    else:
        post_rFC = fc_fit(optimizee_params, signals)


    if verbose:
        print("w adjusted: ", end="")
        print(theta)
        print("pre | post . rFC")
        print(pre_rFC, "  |  ", post_rFC)
        print(np.average(pre_rFC), "  |  ", np.average(post_rFC))


    return theta, cost_fft, pre_rFC, post_rFC


def cost_fft_fit(optimizee_params, signals):

    ctb_folder = optimizee_params["ctb_folder"]
    emp_subj = optimizee_params["emp_subj"]
    samplingFreq = optimizee_params["samplingFreq"]  #Hz
    fc_subset = optimizee_params["fc_subset"]

    # Calculate simulated spectra
    sim_freqs, sim_spectra = multitapper(signals, samplingFreq, epoch_length=4, ntapper=4, smoothing=0.5, plot=False)

    try:  # Load empirical spectra
        if fc_subset:
            emp_spectra = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatSpectra.txt')[fc_subset, :]
        else:
            emp_spectra = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatSpectra.txt')

        emp_freqs = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_flatFreqs.txt')

    except:
        print('Did you calculate spectra for the current subject?')

    ## Now, calculate integrals: all curve vs until alpha curve.
    emp_full_integral = np.trapz(emp_spectra, emp_freqs)
    emp_2alpha_integral = np.trapz(emp_spectra[:, (2 <= emp_freqs) & (emp_freqs <= 12)], emp_freqs[(2 <= emp_freqs) & (emp_freqs <= 12)])

    sim_full_integral = np.trapz(sim_spectra, sim_freqs)
    sim_2alpha_integral = np.trapz(sim_spectra[:, (2 <= sim_freqs) & (sim_freqs <= 12)], sim_freqs[(2 <= sim_freqs) & (sim_freqs <= 12)])

    ## COST: Calculates fit between pre-alpha, post-alpha proportions in empirical and simulated spectra.
    # A positive cost means, frequency and "w" must raise.
    # This value can be directly used in the gradient descent as a positive value implies
    # the simulation need to raise its frequency (lowering down "w").
    fft_fit = sim_2alpha_integral / sim_full_integral - emp_2alpha_integral / emp_full_integral

    return fft_fit







