import pickle
import time
import numpy as np
import pandas as pd
from mne import filter
import scipy

from fooof import FOOOF
import plotly.express as px
import plotly.offline
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go  # for data visualisation

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N

from toolbox import PLV, epochingTool, multitapper, timeseriesPlot


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

    p_array=np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.22, 0)
    sigma_array=np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.022, 0)

    # Parameters from Stefanovski 2019.
    m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                             tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                             He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
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
            emp_spectra[emp_spectra<0]=0
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

    if rule == "peak_balance":
        # Now, calculate integrals: all curve vs until alpha curve. Used until 28 Oct.
        emp_2alpha_integral = np.trapz(emp_spectra[:, (2 <= emp_freqs) & (emp_freqs <= 12)], emp_freqs[(2 <= emp_freqs) & (emp_freqs <= 12)])
        sim_2alpha_integral = np.trapz(sim_spectra[:, (2 <= sim_freqs) & (sim_freqs <= 12)], sim_freqs[(2 <= sim_freqs) & (sim_freqs <= 12)])

        # Now, calculate integrals: upper gamma. Used until 28 Oct.
        emp_2gamma_integral = np.trapz(emp_spectra[:, 30 <= emp_freqs], emp_freqs[30 <= emp_freqs])
        sim_2gamma_integral = np.trapz(sim_spectra[:, 30 <= sim_freqs], sim_freqs[30 <= sim_freqs])

        # COST: Calculates fit between pre-alpha, post-alpha proportions in empirical and simulated spectra.
        # A positive cost means, frequency and "w" must raise.
        # This value can be directly used in the gradient descent as a positive value implies
        # the simulation need to raise its frequency (lowering down "w").
        alpha_cost = sim_2alpha_integral / sim_full_integral - emp_2alpha_integral / emp_full_integral
        # Now, calculate the fit between gamma ratio in empirical and simulated spectra. This allows to check and
        # balance "w" when the algorithm tends to keep rising spectra in gamma band: what's not a usual finding.
        gamma_cost = sim_2gamma_integral / sim_full_integral - emp_2gamma_integral / emp_full_integral

        if save_curves:
            return alpha_cost, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs), gamma_cost
        else:
            return alpha_cost, pointwise_cost, gamma_cost

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


def cost_derivative(history):

    history_costfft = np.asarray(history["cost"][-5:])[:, 0, :]
    history_costpw = np.asarray(history["cost"][-5:])[:, 1, :]
    history_theta = np.asarray(history["theta"][-5:])

    # derivative of Cost fft based on last 5 iterations
    deriv_estim_costfft = np.average(np.asarray([(history_costfft[i+1] - history_costfft[i])/np.sign(history_theta[i+1] - history_theta[i]) for i in range(len(history_theta)-1)]), axis=0)
    deriv_estim_costfft = np.nan_to_num(deriv_estim_costfft, nan=0.0, posinf=0, neginf=0)

    # Cost pointwise derivative based on last 5 iterations
    deriv_estim_pw = np.average(np.asarray([(history_costpw[i+1] - history_costpw[i])/np.sign(history_theta[i+1] - history_theta[i]) for i in range(len(history_theta)-1)]), axis=0)
    deriv_estim_pw = np.nan_to_num(deriv_estim_pw, nan=0.0, posinf=0, neginf=0)

    return deriv_estim_costfft, deriv_estim_pw


def show_evolution(history, regionLabels, report_rois, title="some", output_folder="figures", show=["report"], auto_open=True):

    tic = time.time()

    # unpack Spectra
    sim_spectra, sim_freqs = curves_sim_fft, curves_sim_freq
    emp_spectra, emp_freqs = history["curves_emp"][0], history["curves_emp"][1]
    # unpack History
    fc_history = np.asarray(history["fc"])
    costfft_history = np.asarray(history["cost"])[:, 0, :]
    costpw_history = np.asarray(history["cost"])[:, 1, :]
    if np.asarray(history["cost"])[:, 2, :].any():
        costfftgamma_history = np.asarray(history["cost"])[:, 2, :]
    theta_history = np.asarray(history["theta"])
    curves_history = np.asarray(history["curves"])

    if "report" in show:
        print("CHECK POINT:")
        print("Plotting report _ must be fast")
        fig = make_subplots(rows=2, cols=3, subplot_titles=("ROI history _it", "ROI history _w", "FFTs", "Current values", "Avg. History"),
                            specs=[[{"secondary_y": True}, {"secondary_y": True}, {}],
                                   [{"colspan": 2, "secondary_y": True}, None, {"secondary_y": True}]])

        for iii, specific_roi in enumerate(report_rois):
            if iii >= 1:  # just show a curve in the beggining
                visible = "legendonly"
            else:
                visible = True

            # Spectra
            fig.add_trace(
                go.Scatter(x=sim_freqs, y=sim_spectra[specific_roi] / np.trapz(sim_spectra, sim_freqs)[specific_roi],
                           legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " sim", xaxis="x2",
                           yaxis="y3", visible=visible), row=1, col=3)

            fig.add_trace(
                go.Scatter(x=emp_freqs, y=emp_spectra[specific_roi] / np.trapz(emp_spectra, emp_freqs)[specific_roi],
                           legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " emp", xaxis="x2",
                           yaxis="y3", visible=visible), row=1, col=3)

            # Cost by w
            fig.add_trace(
                go.Scatter(x=theta_history[:, specific_roi], y=abs(costfft_history[:, specific_roi]), marker_color="red",
                           name="cost_fft", legendgroup=regionLabels[specific_roi], showlegend=False, mode="markers",
                           xaxis="x5", yaxis="y9", visible=visible), row=1, col=2)

            fig.add_trace(
                go.Scatter(x=theta_history[:, specific_roi], y=costpw_history[:, specific_roi], marker_color="gray",
                           name="theta", legendgroup=regionLabels[specific_roi], showlegend=False, mode="markers",
                           xaxis="x5", yaxis="y9", visible=visible), row=1, col=2, secondary_y=True)


            # Historic cost for some ROI
            fig.add_trace(
                go.Scatter(x=np.arange(len(costfft_history)), y=costfft_history[:, specific_roi], marker_color="red",
                           name="cost_fft", legendgroup=regionLabels[specific_roi], showlegend=False, xaxis="x1",
                           yaxis="y1", visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costpw_history)), y=costfftgamma_history[:, specific_roi], marker_color="violet",
                           name="cost_fft_gamma", legendgroup=regionLabels[specific_roi], showlegend=False, xaxis="x1", yaxis="y1",
                           visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costfft_history)), y=theta_history[:, specific_roi], marker_color="blue",
                           name="theta", legendgroup=regionLabels[specific_roi], showlegend=False, xaxis="x1", yaxis="y1",
                           visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costpw_history)), y=costpw_history[:, specific_roi], marker_color="darkgray",
                           name="cost_pw", legendgroup=regionLabels[specific_roi], showlegend=False, xaxis="x1", yaxis="y2",
                           visible=visible), row=1, col=1, secondary_y=True)

        # Current values of cost and w
        fig.add_trace(go.Scatter(x=regionLabels, y=theta, name="w", xaxis="x3", yaxis="y4", marker_color="blue"), row=2,
                      col=1)
        fig.add_trace(
            go.Scatter(x=regionLabels, y=cost_fft, name="cost", mode="markers", xaxis="x3", yaxis="y5", marker_color="red"),
            row=2, col=1, secondary_y=True)

        # History values of cost and fc
        fig.add_trace(
            go.Scatter(x=np.arange(len(fc_history)), y=np.average(fc_history, axis=1), marker_color="mediumaquamarine",
                       name="avgFC", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(go.Scatter(x=np.arange(len(fc_history)), y=np.average(costfft_history, axis=1), marker_color="red",
                                 name="avgCost_fft", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(go.Scatter(x=np.arange(len(fc_history)), y=np.average(costfftgamma_history, axis=1), marker_color="violet",
                                 name="avgCost_fftgamma", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(go.Scatter(x=np.arange(len(fc_history)), y=np.average(costpw_history, axis=1), marker_color="gray",
                                 name="avgCost_pw", xaxis="x4", yaxis="y7"), row=2, col=3, secondary_y=True)

        fig.update_layout(yaxis1=dict(title="fft_cost", titlefont=dict(color="red")),
                          yaxis2=dict(title="pw_cost", titlefont=dict(color="gray")),
                          yaxis3=dict(title="fft_cost", titlefont=dict(color="red")),
                          yaxis4=dict(title="pw_cost", titlefont=dict(color="gray")),
                          yaxis5=dict(title="Normalized module (db)"),
                          yaxis6=dict(title="w param", titlefont=dict(color="blue"), range=[0, 1]),
                          yaxis7=dict(title="cost", titlefont=dict(color="red")),
                          yaxis8=dict(title="rFC emp-sim", titlefont=dict(color="mediumaquamarine")),
                          yaxis9=dict(title="avg cost", titlefont=dict(color="red")),

                          xaxis1=dict(title="iteration"),
                          xaxis2=dict(title="w"),
                          xaxis3=dict(title="Frequency (Hz)"),
                          xaxis4=dict(),  # region labels
                          xaxis5=dict(title="iteration"),


                          title_text="Gradient descent applying '%s' method @ %0.3f lr  |  %0.2f min/it" % (
                          fit_method, learning_rate, (time.time() - tic) / 60,))
        pio.write_html(fig, file=output_folder + "/" + title + "_report.html", auto_open=auto_open)

    # DYNAMICAL FFT
    # mount a df per roi: three columns - name, common_name, it, freq, val
    if "dynFFT" in show:
        tic=time.time()
        print("Plotting dynamical FFTs - may take a while. Wait: ")
        print("      ROIs data reshaping", end="")
        roi_data = np.empty((5))
        for specific_roi in report_rois:
            print(".", end="")
            for it in range(len(curves_history)):
                # Add simulated spectra to roi_data
                it_sim = np.array([it] * len(curves_history[it][1]))
                fft_sim = curves_history[it][0][specific_roi] / np.trapz(curves_history[it][0][specific_roi],
                                                                         curves_history[it][1])
                freqs_sim = curves_history[it][1]
                name_sim = [regionLabels[specific_roi] + "_sim"] * len(curves_history[it][1])
                common_name_sim = [regionLabels[specific_roi]] * len(curves_history[it][1])
                it_chunk = np.vstack((name_sim, common_name_sim, it_sim, fft_sim, freqs_sim)).T
                roi_data = np.vstack((roi_data, it_chunk))
                # Add empirical spectra
                it_emp = np.array([it] * len(emp_spectra[specific_roi]))
                fft_emp = emp_spectra[specific_roi] / np.trapz(emp_spectra[specific_roi], emp_freqs)
                freqs_emp = emp_freqs
                name_emp = [regionLabels[specific_roi] + "_emp"] * len(emp_spectra[specific_roi])
                common_name_emp = [regionLabels[specific_roi]] * len(emp_spectra[specific_roi])
                it_chunk = np.vstack((name_emp, common_name_emp, it_emp, fft_emp, freqs_emp)).T
                roi_data = np.vstack((roi_data, it_chunk))

        print("%0.3fm" % ((time.time() - tic)/60, ))
        tic = time.time()
        print("      Plotting  .  ", end="")
        roi_df = pd.DataFrame(roi_data[1:, :], columns=["name", "common_name", "iteration", "fft", "freqs"])
        roi_df = roi_df.astype({"name": str, "common_name": str, "iteration": int, "fft": float, "freqs": float})

        fig_dyn = px.line(roi_df, x="freqs", y="fft", animation_frame="iteration", animation_group="name",
                          color="name", facet_col="common_name", facet_col_wrap=5, title="Dynamic FFT")
        pio.write_html(fig_dyn, file=output_folder + "/" + title + "_dynFFT.html", auto_open=auto_open)
        print("%0.3fm" % ((time.time() - tic)/60, ))

    # DYNAMICAL W
    # mount a df: three columns - name, it, param=[w, cost], value
    # Current values of cost and w
    if "dynW" in show:
        tic=time.time()
        print("Plotting dynamical Ws - may take a while. Wait: ")
        print("      Data reshaping  .  ", end="")

        roi_data = np.empty((4))
        for it in range(len(theta_history)):
            # Add simulated spectra to roi_data
            it_ = np.array([it] * len(theta_history[0]))
            name_ = regionLabels
            param_ = ["cost"] * len(theta_history[0])
            cost_ = costfft_history[it]
            it_chunk = np.vstack((name_, it_, param_, cost_)).T
            roi_data = np.vstack((roi_data, it_chunk))

            param_ = ["w"] * len(theta_history[0])
            w_ = theta_history[it]
            it_chunk = np.vstack((name_, it_, param_, w_)).T
            roi_data = np.vstack((roi_data, it_chunk))

        print("%0.3fs" % (time.time() - tic, ))
        tic = time.time()
        print("      Plotting  .  ", end="")
        w_df = pd.DataFrame(roi_data[1:, :], columns=["name", "iteration", "param", "value"])
        w_df = w_df.astype({"name": str, "iteration": int, "param": str, "value": float})

        fig_dynw = px.scatter(w_df, x="name", y="value", animation_frame="iteration",
                          color="param", facet_row="param", title="Dynamic W & Cost")
        pio.write_html(fig_dynw, file=output_folder + "/" + title + "_dynW.html", auto_open=auto_open)
        print("%0.3fs" % (time.time() - tic, ))

    return



# Indexes ROI for DMN in AAL2red structure: Frontal_Sup; ACC; PCC; Parietal_inf; Precuneus; Temporal_Mid.
AAL2red_rois_dmn = [2, 3, 34, 35, 38, 39, 64, 65, 70, 71, 84, 85]

## Optimizee global parameters
params = {
    "mode": "FC",  # FC, FFT. Whenever FFT gradient descent will be applied.

    "simLength": 22000, #44000
    "repetitions": 3,
    "transient": 2000,
    "samplingFreq": 1000,

    "emp_subj": "NEMOS_049",
    "structure": "_AAL2red",
    "sc_subset": None,  # AAL2red_rois_dmn,  # Subsets over structure
    "fc_subset": None,  # AAL2red_rois_dmn,  # Subsets over AAL2red (main FC output shape from brainstorm).
    "ctb_folder": "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\",
    "verbose": False}

# Calculate number of rois
conn = connectivity.Connectivity.from_file(params["ctb_folder"] + params["emp_subj"] + params["structure"]+".zip")
n_rois = len(params["sc_subset"]) if params["sc_subset"] else len(conn.region_labels)
# Define region labels
regionLabels = conn.region_labels[params["sc_subset"]] if params["sc_subset"] else conn.region_labels


######
## SET UP SIMULATION
name = "GD_" + time.strftime("d%dm%my%Y-t%Hh%Mm")
output_folder = "Gradient_results/"

iterations = 200
fit_method = "peak_balance"
learning_rate = 5
show_dynFFTs = True

w = np.asarray([0.8]*n_rois)
g, s = 93, 4.5  # For emp_subject NEMOS_049

check_point = 25  # plot results each x iterations
dyn_check = 50
report_rois = [1, 4, 54, 64, 84]
###############


tic0 = time.time()
print("Initializing Gradient Descent Algortihm...")
individual = list((g, s, w))
history = {"theta": list(), "cost": list(), "curves": list(), "curves_emp":list(), "fc": list()}
theta = w

for it in range(iterations):
    tic = time.time()

    print('\n\n Iteration %i  -  Simulating for g=%i - s=%0.1f  -  simulation Time : ' % (it, g, s), end="")
    history["theta"].append(theta.T)
    signals = [signalsJRD(params, individual, verbose=False) for i in range(params["repetitions"])]
    print("%0.4f sec" % (time.time()-tic, ))

    if it == 0:
        pre_rFC = np.average([fc_fit(params, signals_i) for signals_i in signals], axis=0)
        history["fc"].append(pre_rFC)
        print("A priori rFC: ", end="")
        print(pre_rFC)
        print("pre-avg: %0.4f" % np.average(pre_rFC))

    else:
        post_rFC = np.average([fc_fit(params, signals_i) for signals_i in signals], axis=0)
        history["fc"].append(post_rFC)
        print("A priori rFC: ", end="")
        print(pre_rFC)
        print("Post-fit rFC: ", end="")
        print(post_rFC)
        print("pre-avg: %0.4f | post-avg: %0.4f" % (np.average(pre_rFC), np.average(post_rFC)))

    ### Calculate a relative difference (low frequencies vs all) in emp-sim spectra per roi
    # "alpha_vs_all"; "pointwise_comparison"; "foof"
    cost_fft, cost_pw, curves_emp, curves_sim_fft, curves_sim_freq, cost_fft_gamma = [], [], [], [], [], []

    for signals_i in signals:
        cost = cost_function(params, signals_i, rule=fit_method, save_curves=True)
        cost_fft.append(cost[0])
        if fit_method == "peak_balance":
            cost_fft_gamma.append(cost[3])
        cost_pw.append(cost[1])
        curves_sim_fft.append(cost[2][0])
        curves_sim_freq.append(cost[2][1])
        curves_emp = cost[2][2:]

    cost_fft = np.average(cost_fft, axis=0)
    cost_fft_gamma = np.average(cost_fft_gamma, axis=0)
    cost_pw = np.average(cost_pw, axis=0)
    if fit_method == "peak_balance":
        history["cost"].append([cost_fft, cost_pw, cost_fft_gamma])
    else:
        history["cost"].append([cost_fft, cost_pw])

    curves_sim_fft = np.average(curves_sim_fft, axis=0)
    curves_sim_freq = np.average(curves_sim_freq, axis=0)
    history["curves"].append([curves_sim_fft, curves_sim_freq])  # Curves[0]=sim_spectra; [1]sim_freqs; [2]emp_spectra; [3]emp_freqs
    if it == 0:   # in the first iteration save empirical spectra and freqs
        history["curves_emp"] = curves_emp

    # d_cost, d_pw = cost_derivative(history)  # Stop mechanism when max pw fit is achieved; after that undesired gamma peak appears. Avoid this.
    # cost_fft[np.where((d_cost > 0) & (d_pw <= 0))] = - cost_fft[np.where((d_cost > 0) & (d_pw <= 0))]
    if fit_method == "peak_balance":
        cost_fft = cost_fft - cost_fft_gamma
    theta = theta - (1/len(w)) * learning_rate * cost_fft  # len(w) as "m"

    theta[theta <= 0] = 0
    theta[theta >= 1] = 1

    individual[2] = theta

    print(' cost %0.2f  -  time: %0.2f/%0.2f min' % (np.average(np.abs(cost_fft)), (time.time()-tic)/60, (time.time()-tic0)/60, ))

    if (it > 1) & (it % dyn_check != 0) & (it % check_point == 0):
        show_evolution(history, regionLabels, report_rois, name, output_folder)
    elif (it > 1) & (it % dyn_check == 0) & (it % check_point == 0):
        show_evolution(history, regionLabels, report_rois, name, output_folder, show=["report", "dynFFT", "dynW"])

## Save important data  ## To load it back: open(file, 'r') as f; pickle.load()
file_params = open(output_folder + name + "_params.pkl", "wb")
pickle.dump(params, file_params)
file_params.close()

file_history = open(output_folder + name + "_history.pkl", "wb")
pickle.dump(history, file_history)
file_history.close()

## TO LOAD BACK:
# with open("heterogeneity/Gradient_results/GD_d11m11y2021-t13h20m_history.pkl", 'rb') as f:
#     history = pickle.load(f)

## Applying retrospective approach to get best w combination
# Maximizing rFC and minimizing FFT cost.
top_n = 10  # Top results to check
transient = 10  # Initial iterations where fc and cost rapidly shrinks

avgFC_history = np.average(np.asarray(history["fc"])[transient:, :], axis=1)  # 10 iterations to remove initial FC decay and high cost
avgCost_history = np.average(np.asarray(history["cost"])[transient:, 0, :], axis=1)

# Look for iterations where FC is maximized - Extract to 10
top_iterations_short = np.argsort(avgFC_history)[-10:]
top_iterations, top_fc, top_cost, top_w = top_iterations_short + transient, avgFC_history[top_iterations_short], avgCost_history[top_iterations_short], np.asarray(history["theta"])[top_iterations_short]

# Then use minimum cost - as control.
best_fc, best_cost, best_w, best_iteration_short, best_iteration = \
    top_fc[np.argmax(top_fc)], top_cost[np.argmax(top_fc)], top_w[np.argmax(top_fc)], top_iterations_short[np.argmax(top_fc)], top_iterations[np.argmax(top_fc)]

# Left rois full report
report_rois = np.arange(0, len(regionLabels), 4)
show_evolution(history, regionLabels, report_rois, name, output_folder, show=["dynFFT", "dynW"])
# Right rois full report
report_rois = np.arange(1, len(regionLabels), 2)
show_evolution(history, regionLabels, report_rois, name, output_folder, show=["dynFFT", "dynW"])


###### MORE plots
        # fig = px.line(x=range(len(cost_history)), y=np.average(cost_history, axis=1))
        # fig.show(renderer="browser")
        # # plotly.offline.iplot(fig)
        #
        #
        # if sc_subset:
        #     fig = px.scatter(x=conn.region_labels[sc_subset], y=cost_fft)
        #     fig.add_scatter(x=conn.region_labels[sc_subset], y=theta, name=w)
        # else:
        #     fig = px.scatter(x=conn.region_labels, y=cost_fft)
        #     fig.add_scatter(x=conn.region_labels, y=theta, name=w)
        # fig.update_yaxes(range=[-0.2, 1])
        # fig.show(renderer="browser")
        # # plotly.offline.iplot(fig)
        # print(cost_fft)
        #
        # print("Plotting FFTs...")
        # sim_spectra, sim_freqs = curves[0], curves[1]
        # emp_spectra, emp_freqs = curves_emp[0], curves_emp[1]
        #
        # fig = go.Figure()
        # for specific_roi in range(5):
        #     fig.add_scatter(x=sim_freqs, y=sim_spectra[specific_roi] / np.trapz(sim_spectra, sim_freqs)[specific_roi],
        #                     legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " sim")
        #     fig.add_scatter(x=emp_freqs, y=emp_spectra[specific_roi] / np.trapz(emp_spectra, emp_freqs)[specific_roi],
        #                     legendgroup=regionLabels[specific_roi], name=regionLabels[specific_roi] + " emp")
        # fig.show(renderer="browser")
        # plotly.offline.iplot(fig)
        #
        # print("Plotting signals...")
        # timeseriesPlot(signals - np.average(signals, axis=1)[:, None], np.arange(transient, simLength, 1), regionLabels,
        #                mode="html")



