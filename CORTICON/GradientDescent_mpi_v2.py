
import pickle
import time
import numpy as np
import pandas as pd
from mne import filter
import scipy

import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go  # for data visualisation
from mpi4py import MPI

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003

import warnings
warnings.filterwarnings("ignore")

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python GradientDescent_mpi_v2.py
"""

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Hello from rank %i" % rank)

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV

    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
    spectra_folder = "E:\LCCN_Local\PycharmProjects\SpectraAnalysis\\fieldtrip_data_AAL2\\"

## Folder structure - CLUSTER
else:
    from toolbox import multitapper, PLV, epochingTool
    # import sys
    # sys.path.append(wd)
    # from toolbox.fft import multitapper
    # from toolbox.signals import epochingTool
    # from toolbox.fc import PLV

    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data2/"
    spectra_folder = wd + "fieldtrip_data_AAL2/"


def signals_jrd(parameters, individual, verbose=True):

    ## STRUCTURE
    conn_ = connectivity.Connectivity.from_file(parameters["ctb_folder"] + individual[3] + parameters["structure"] + ".zip")
    conn_.weights = conn_.scaled_weights(mode="tract")
    conn_.speed = np.array([individual[1]])

    ## NMM
    p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn_.region_labels])
    sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn_.region_labels])

    # Parameters edited from David and Friston (2003).
    m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                           tau_e1=np.array([10]), tau_i1=np.array([20]),
                           He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                           tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                           w=individual[2], c=np.array([135.0]),
                           c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                           c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                           v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                           p=p_array, sigma=sigma_array)

    # Remember to hold tau*H constant.
    m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
    m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    ## COUPLING FUNCTION
    coup = coupling.SigmoidalJansenRitDavid(a=np.array([individual[0]]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

    ## OTHERS
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / parameters["samplingFreq"])

    mon = (monitors.Raw(),)

    tic0_ = time.time()

    if verbose:
        print("Simulating for g=%i - s=%0.2f" % (individual[0], individual[1]))

    ## RUN SIMULATION
    sim = simulator.Simulator(model=m, connectivity=conn_, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=parameters["simLength"])

    if verbose:
        print("Simulation time: %0.2f sec" % (time.time() - tic0_,))

    # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
    # Mount PSP output as: w * (vExc1 - vInh1) + (1 - w) * (vExc2 - vInh2)
    psp_t = m.w * (output[0][1][parameters["transient"]:, 1, :, 0].T - output[0][1][parameters["transient"]:, 2, :, 0].T) + \
            (1 - m.w) * (output[0][1][parameters["transient"]:, 5, :, 0].T - output[0][1][parameters["transient"]:, 6, :, 0].T)

    return psp_t


def fc_fit(parameters, individual, raw_data, FC_idx):

    results = list()

    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    bands = [["3-alpha"], [(8, 12)]]
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, parameters["samplingFreq"], lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, parameters["samplingFreq"], "signals", verbose=False)

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
        plv_emp = \
            np.loadtxt(parameters["ctb_folder"] + "FCrms_" + individual[3] + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
            FC_idx][FC_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
        t1[0, :] = plv[np.triu_indices(len(plv), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        results.append(plv_r)

    return results


def cost_function(parameters, individual, signals, FFT_idx, rule="alpha_vs_all", save_curves=False):
    """
    Cost length depends on the number of empirical spectra over which we will run the optimization.
    """

    # Calculate simulated spectra
    sim_freqs, sim_spectra = multitapper(signals, parameters["samplingFreq"], cortical_rois, epoch_length=4, ntapper=4, smoothing=0.5,
                                         plot=False)

    try:  # Load empirical spectra

        emp_spectra = np.loadtxt(parameters["spectra_folder"] + individual[3] + "/flat_fft.txt", delimiter=" ")[FFT_idx, :]
        emp_spectra[emp_spectra < 0] = 0
        # gaussian_means_emp = np.loadtxt(ctb_folder+'BSsignals_spectra/'+emp_subj+'_gaussianMean.txt')

        emp_freqs = np.loadtxt(parameters["spectra_folder"] + individual[3] + "/flat_freqs.txt", delimiter=" ")

    except:
        print('Did you calculate spectra for the current subject?')

    ## Pointwise cost
    # Use the differences point to point between spectra to have another measure of cost: use it as stop criteria.
    emp_full_integral = np.trapz(emp_spectra, emp_freqs)
    sim_full_integral = np.trapz(sim_spectra, sim_freqs)

    # these indexes cut simulated spectra to the length of empirical one
    low_idx = int(np.where(sim_freqs[0] == emp_freqs)[0])
    high_idx = int(np.where(sim_freqs[-1] == emp_freqs)[0])  # sim freqs is shorter than emp_freqs

    norm_emp_spectra = emp_spectra[:, low_idx:high_idx + 1] / emp_full_integral[:, np.newaxis]
    norm_sim_spectra = sim_spectra / sim_full_integral[:, np.newaxis]

    pointwise_cost = np.sum(abs(norm_sim_spectra - norm_emp_spectra), axis=1)

    if rule == "peak_balance":
        # Now, calculate integrals: all curve vs until alpha curve. Used until 28 Oct.
        emp_2alpha_integral = np.trapz(emp_spectra[:, (2 <= emp_freqs) & (emp_freqs <= 12)],
                                       emp_freqs[(2 <= emp_freqs) & (emp_freqs <= 12)])
        sim_2alpha_integral = np.trapz(sim_spectra[:, (2 <= sim_freqs) & (sim_freqs <= 12)],
                                       sim_freqs[(2 <= sim_freqs) & (sim_freqs <= 12)])

        # Now, calculate integrals: upper gamma. Used until 28 Oct.
        emp_4gamma_integral = np.trapz(emp_spectra[:, 30 <= emp_freqs], emp_freqs[30 <= emp_freqs])
        sim_4gamma_integral = np.trapz(sim_spectra[:, 30 <= sim_freqs], sim_freqs[30 <= sim_freqs])

        # COST: Calculates fit between pre-alpha (alpha cost), pre-gamma (gamma_cost)
        # proportions in empirical and simulated spectra.
        # A positive alpha cost means: frequency must raise to reach alpha thus "w" must lower down.
        alpha_cost = sim_2alpha_integral / sim_full_integral - emp_2alpha_integral / emp_full_integral

        # Now, compare the difference between gamma ratio in empirical and simulated spectra. Used to stop
        # lowering "w" when This allows to check and
        # balance "w" when the algorithm tends to keep rising spectra in gamma band: what's not a usual finding.
        gamma_cost = sim_4gamma_integral / sim_full_integral - emp_4gamma_integral / emp_full_integral

        if save_curves:
            return alpha_cost, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs), gamma_cost
        else:
            return alpha_cost, pointwise_cost, gamma_cost

    elif rule == "alpha_vs_all":
        # Now, calculate integrals: all curve vs until alpha curve. Used until 28 Oct.
        emp_2alpha_integral = np.trapz(emp_spectra[:, (2 <= emp_freqs) & (emp_freqs <= 12)],
                                       emp_freqs[(2 <= emp_freqs) & (emp_freqs <= 12)])
        sim_2alpha_integral = np.trapz(sim_spectra[:, (2 <= sim_freqs) & (sim_freqs <= 12)],
                                       sim_freqs[(2 <= sim_freqs) & (sim_freqs <= 12)])

        # COST: Calculates fit between pre-alpha, post-alpha proportions in empirical and simulated spectra.
        # A positive cost means, frequency and "w" must raise.
        # This value can be directly used in the gradient descent as a positive value implies
        # the simulation need to raise its frequency (lowering down "w").
        fft_cost = sim_2alpha_integral / sim_full_integral - emp_2alpha_integral / emp_full_integral

        if save_curves:
            return fft_cost, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
        else:
            return fft_cost, pointwise_cost

    # elif rule == "pointwise_comparison":
    #     # New approach (28 oct). Use the mean of the gaussian model as reference and check the differences point to point
    #     # between the spectra to see where the error is: above or below the mean, then move w consequently.
    #     mean_idx = [np.argmin(np.abs(emp_freqs - m)) for m in gaussian_means_emp]
    #     norm_spectra_diff = norm_sim_spectra - norm_emp_spectra
    #
    #     sum_preMean = np.array([sum(norm_spectra_diff[i, :mean_i]) for i, mean_i in enumerate(mean_idx)])
    #     sum_postMean = np.array([sum(norm_spectra_diff[i, mean_i:]) for i, mean_i in enumerate(mean_idx)])
    #
    #     # np.vstack((sum_preMean, um_postMean)).T  # Visualization purposes
    #
    #     diff = sum_preMean - sum_postMean  # if positive -> move to higher frequencies (i.e. lower w)
    #
    #     if save_curves:
    #         return diff, pointwise_cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
    #     else:
    #         return diff, pointwise_cost

    # elif rule == "fooof":
    #     ## Use foof to model the simulated spectra and define whether it should move up or down.
    #     fm = FOOOF(max_n_peaks=1)  # peak_width_limits=[2.0, 10.0]
    #     # Define frequency range across which to model the spectrum
    #     freq_range = [5, 40]  # Consider that multitapper uses the range [2, 45]
    #
    #     gaussian_means_sim = []
    #     for roi in range(len(sim_spectra)):
    #         # Model the power spectrum with FOOOF, and print out a report
    #         fm.fit(sim_freqs, sim_spectra[roi], freq_range)
    #         # Calculate gaussian mean
    #         if np.isnan(fm.get_params('peak_params', 'CF')):
    #             gaussian_means_sim.append(0)
    #         else:
    #             gaussian_means_sim.append(fm.get_params('peak_params', 'CF'))
    #
    #
    #     # positive cost implies lowering down 'w'
    #     cost = gaussian_means_emp - np.array(gaussian_means_sim)
    #
    #     if save_curves:
    #         return cost, (sim_spectra, sim_freqs, emp_spectra, emp_freqs)
    #     else:
    #         return cost


def cost_derivative(history_):
    history_costfft = np.asarray(history_["cost"][-5:])[:, 0, :]
    history_costpw = np.asarray(history_["cost"][-5:])[:, 1, :]
    history_theta = np.asarray(history_["theta"][-5:])

    # derivative of Cost fft based on last 5 iterations
    deriv_estim_costfft = np.average(np.asarray(
        [(history_costfft[i + 1] - history_costfft[i]) / np.sign(history_theta[i + 1] - history_theta[i]) for i in
         range(len(history_theta) - 1)]), axis=0)
    deriv_estim_costfft = np.nan_to_num(deriv_estim_costfft, nan=0.0, posinf=0, neginf=0)

    # Cost pointwise derivative based on last 5 iterations
    deriv_estim_pw = np.average(np.asarray(
        [(history_costpw[i + 1] - history_costpw[i]) / np.sign(history_theta[i + 1] - history_theta[i]) for i in
         range(len(history_theta) - 1)]), axis=0)
    deriv_estim_pw = np.nan_to_num(deriv_estim_pw, nan=0.0, posinf=0, neginf=0)

    return deriv_estim_costfft, deriv_estim_pw


def show_evolution(history_, region_labels, report_rois, title="some", folder="figures", show=["report"],
                   auto_open=True):
    tic = time.time()

    # unpack Spectra
    sim_spectra, sim_freqs = curves_sim_fft, curves_sim_freq
    emp_spectra, emp_freqs = history_["curves_emp"][0], history_["curves_emp"][1]

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
        fig = make_subplots(rows=2, cols=3, subplot_titles=(
            "ROI history _it", "ROI history _w", "FFTs", "Current values", "Avg. History"),
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
                           legendgroup=region_labels[specific_roi], name=region_labels[specific_roi] + " sim", xaxis="x2",
                           yaxis="y3", visible=visible), row=1, col=3)

            fig.add_trace(
                go.Scatter(x=emp_freqs, y=emp_spectra[specific_roi] / np.trapz(emp_spectra, emp_freqs)[specific_roi],
                           legendgroup=region_labels[specific_roi], name=region_labels[specific_roi] + " emp", xaxis="x2",
                           yaxis="y3", visible=visible), row=1, col=3)

            # Cost by w
            fig.add_trace(
                go.Scatter(x=theta_history[:, specific_roi], y=abs(costfft_history[:, specific_roi]),
                           marker_color="red",
                           name="cost_fft", legendgroup=region_labels[specific_roi], showlegend=False, mode="markers",
                           xaxis="x5", yaxis="y9", visible=visible), row=1, col=2)

            fig.add_trace(
                go.Scatter(x=theta_history[:, specific_roi], y=costpw_history[:, specific_roi], marker_color="gray",
                           name="theta", legendgroup=region_labels[specific_roi], showlegend=False, mode="markers",
                           xaxis="x5", yaxis="y9", visible=visible), row=1, col=2, secondary_y=True)

            # Historic cost for some ROI
            fig.add_trace(
                go.Scatter(x=np.arange(len(costfft_history)), y=costfft_history[:, specific_roi], marker_color="red",
                           name="cost_fft", legendgroup=region_labels[specific_roi], showlegend=False, xaxis="x1",
                           yaxis="y1", visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costpw_history)), y=costfftgamma_history[:, specific_roi],
                           marker_color="violet",
                           name="cost_fft_gamma", legendgroup=region_labels[specific_roi], showlegend=False, xaxis="x1",
                           yaxis="y1",
                           visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costfft_history)), y=theta_history[:, specific_roi], marker_color="blue",
                           name="theta", legendgroup=region_labels[specific_roi], showlegend=False, xaxis="x1",
                           yaxis="y1",
                           visible=visible), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=np.arange(len(costpw_history)), y=costpw_history[:, specific_roi], marker_color="darkgray",
                           name="cost_pw", legendgroup=region_labels[specific_roi], showlegend=False, xaxis="x1",
                           yaxis="y2",
                           visible=visible), row=1, col=1, secondary_y=True)

        # Current values of cost and w
        fig.add_trace(go.Scatter(x=region_labels, y=theta, name="w", xaxis="x3", yaxis="y4", marker_color="blue"), row=2,
                      col=1)
        fig.add_trace(
            go.Scatter(x=region_labels, y=cost_fft, name="cost", mode="markers", xaxis="x3", yaxis="y5",
                       marker_color="red"),
            row=2, col=1, secondary_y=True)

        # History values of cost and fc
        fig.add_trace(
            go.Scatter(x=np.arange(len(fc_history)), y=np.average(fc_history, axis=1), marker_color="mediumaquamarine",
                       name="avgFC", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(
            go.Scatter(x=np.arange(len(fc_history)), y=np.average(costfft_history, axis=1), marker_color="red",
                       name="avgCost_fft", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(
            go.Scatter(x=np.arange(len(fc_history)), y=np.average(costfftgamma_history, axis=1), marker_color="violet",
                       name="avgCost_fftgamma", xaxis="x4", yaxis="y6"), row=2, col=3)
        fig.add_trace(
            go.Scatter(x=np.arange(len(fc_history)), y=np.average(costpw_history, axis=1), marker_color="gray",
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
        pio.write_html(fig, file=folder + "/" + title + "_report.html", auto_open=auto_open)

    # DYNAMICAL FFT
    # mount a df per roi: three columns - name, common_name, it, freq, val
    if "dynFFT" in show:
        tic = time.time()
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
                name_sim = [region_labels[specific_roi] + "_sim"] * len(curves_history[it][1])
                common_name_sim = [region_labels[specific_roi]] * len(curves_history[it][1])
                it_chunk = np.vstack((name_sim, common_name_sim, it_sim, fft_sim, freqs_sim)).T
                roi_data = np.vstack((roi_data, it_chunk))
                # Add empirical spectra
                it_emp = np.array([it] * len(emp_spectra[specific_roi]))
                fft_emp = emp_spectra[specific_roi] / np.trapz(emp_spectra[specific_roi], emp_freqs)
                freqs_emp = emp_freqs
                name_emp = [region_labels[specific_roi] + "_emp"] * len(emp_spectra[specific_roi])
                common_name_emp = [region_labels[specific_roi]] * len(emp_spectra[specific_roi])
                it_chunk = np.vstack((name_emp, common_name_emp, it_emp, fft_emp, freqs_emp)).T
                roi_data = np.vstack((roi_data, it_chunk))

        print("%0.3fm" % ((time.time() - tic) / 60,))
        tic = time.time()
        print("      Plotting  .  ", end="")
        roi_df = pd.DataFrame(roi_data[1:, :], columns=["name", "common_name", "iteration", "fft", "freqs"])
        roi_df = roi_df.astype({"name": str, "common_name": str, "iteration": int, "fft": float, "freqs": float})

        fig_dyn = px.line(roi_df, x="freqs", y="fft", animation_frame="iteration", animation_group="name",
                          color="name", facet_col="common_name", facet_col_wrap=5, title="Dynamic FFT")
        pio.write_html(fig_dyn, file=folder + "/" + title + "_dynFFT.html", auto_open=auto_open)
        print("%0.3fm" % ((time.time() - tic) / 60,))

    # DYNAMICAL W
    # mount a df: three columns - name, it, param=[w, cost], value
    # Current values of cost and w
    if "dynW" in show:
        tic = time.time()
        print("Plotting dynamical Ws - may take a while. Wait: ")
        print("      Data reshaping  .  ", end="")

        roi_data = np.empty((4))
        for it in range(len(theta_history)):
            # Add simulated spectra to roi_data
            it_ = np.array([it] * len(theta_history[0]))
            name_ = region_labels
            param_ = ["cost"] * len(theta_history[0])
            cost_ = costfft_history[it]
            it_chunk = np.vstack((name_, it_, param_, cost_)).T
            roi_data = np.vstack((roi_data, it_chunk))

            param_ = ["w"] * len(theta_history[0])
            w_ = theta_history[it]
            it_chunk = np.vstack((name_, it_, param_, w_)).T
            roi_data = np.vstack((roi_data, it_chunk))

        print("%0.3fs" % (time.time() - tic,))
        tic = time.time()
        print("      Plotting  .  ", end="")
        w_df = pd.DataFrame(roi_data[1:, :], columns=["name", "iteration", "param", "value"])
        w_df = w_df.astype({"name": str, "iteration": int, "param": str, "value": float})

        fig_dynw = px.scatter(w_df, x="name", y="value", animation_frame="iteration",
                              color="param", facet_row="param", title="Dynamic W & Cost")
        pio.write_html(fig_dynw, file=folder + "/" + title + "_dynW.html", auto_open=auto_open)
        print("%0.3fs" % (time.time() - tic,))

    return


## Optimizee global parameters
params = {
    "ctb_folder": ctb_folder,
    "spectra_folder": spectra_folder,

    "simLength": 30000,  # 44000
    "transient": 10000,
    "samplingFreq": 1000,
    "repetitions": 3,

    "structure": "_AAL2pTh",

    "verbose": False}


                # model; subj; rPLV; g, s   ### from: CORTICON/PSE/PSEmpi_dynSys_WorkingPoints-m04d01y2022-t20h.19m.37s
working_points = [["jrd_mix", 'NEMOS_075', 0.4561801536874189, 138, 6.5],
                  ["jrd_mix", 'NEMOS_059', 0.4633034792146057, 78,  4.5],
                  ["jrd_mix", 'NEMOS_049', 0.2826762031561815, 50,  4.5],
                  ["jrd_mix", 'NEMOS_035', 0.3655899127321862, 82,  6.5],
                  ["jrd_mix", 'NEMOS_071', 0.3828224797103641, 126, 16.5],
                  ["jrd_mix", 'NEMOS_058', 0.3238695846582934, 146, 6.5],
                  ["jrd_mix", 'NEMOS_050', 0.4568623593946254, 46,  4.5],
                  ["jrd_mix", 'NEMOS_065', 0.3433422282538486, 80,  6.5],
                  ["jrd_mix", 'NEMOS_077', 0.3939552674957418, 146, 4.5],
                  ["jrd_mix", 'NEMOS_064', 0.4236810275102919, 110, 6.5]]


_, emp_subj, r_plv_wp, g, s = working_points[rank]


# Calculate number of rois
conn = connectivity.Connectivity.from_file(params["ctb_folder"] + emp_subj + params["structure"] + ".zip")
n_rois = len(conn.region_labels)

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
FClabs = list(np.loadtxt(params["ctb_folder"] + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois

SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

FFTlabs = list(np.loadtxt(params["spectra_folder"] + emp_subj + "/labels.txt", delimiter=",", dtype=str))
FFT_cortex_idx = [FFTlabs.index(roi) for roi in cortical_rois]

del conn, FClabs, SClabs, FFTlabs



######
## SET UP SIMULATION
name = "GD_" + time.strftime("d%dm%my%Y-t%Hh%Mm")

iterations = 100  # About 50 iterations to fit
fit_method = "peak_balance"
learning_rate = 5

w = np.asarray([0.8] * n_rois)

# check_point = 25  # plot results each x iterations
# dyn_check = 50
# report_rois = [1, 4, 15, 54, 64]
###############



tic0 = time.time()
print("Initializing Gradient Descent Algortihm...")
ind = list((g, s, w, emp_subj))
history = {"theta": list(), "cost": list(), "curves": list(), "curves_emp": list(), "fc": list()}
theta = w

for it in range(iterations):
    tic = time.time()

    print('\n\n Iteration %i  -  Simulating for %s g=%i - s=%0.1f  -  simulation Time : ' % (it, emp_subj, g, s), end="")
    history["theta"].append(list(theta.T))
    signals_rep = [signals_jrd(params, ind, verbose=False) for i in range(params["repetitions"])]
    print("%0.4f sec" % (time.time() - tic,))

    if it == 0:
        pre_rFC = np.average([fc_fit(params, ind, signals_i[SC_cortex_idx], FC_cortex_idx) for signals_i in signals_rep], axis=0)
        history["fc"].append(pre_rFC)
        print("A priori rFC:  %0.4f  __check__ in WP was %0.4f " % (pre_rFC, r_plv_wp))

    else:
        post_rFC = np.average([fc_fit(params, ind, signals_i[SC_cortex_idx], FC_cortex_idx) for signals_i in signals_rep], axis=0)
        history["fc"].append(post_rFC)
        print("pre-avg: %0.4f | post-avg: %0.4f" % (np.average(pre_rFC), np.average(post_rFC)))

    ### Calculate a relative difference (low frequencies vs all) in emp-sim spectra per roi
    # "alpha_vs_all"; "pointwise_comparison"; "fooof"
    cost_fft, cost_pw, curves_emp, curves_sim_fft, curves_sim_freq, cost_fft_gamma = [], [], [], [], [], []

    for signals_i in signals_rep:
        cost = cost_function(params, ind, signals_i[SC_cortex_idx], FFT_cortex_idx, rule=fit_method, save_curves=True)
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
    history["curves"].append(
        [curves_sim_fft, curves_sim_freq])  # Curves[0]=sim_spectra; [1]sim_freqs; [2]emp_spectra; [3]emp_freqs
    if it == 0:  # in the first iteration save empirical spectra and freqs
        history["curves_emp"] = curves_emp

    # d_cost, d_pw = cost_derivative(history)  # Stop mechanism when max pw fit is achieved; after that undesired gamma peak appears. Avoid this.
    # cost_fft[np.where((d_cost > 0) & (d_pw <= 0))] = - cost_fft[np.where((d_cost > 0) & (d_pw <= 0))]
    if fit_method == "peak_balance":
        cost_fft = cost_fft - cost_fft_gamma
    theta[SC_cortex_idx] = theta[SC_cortex_idx] - (1 / len(w)) * learning_rate * cost_fft  # len(w) as "m"

    theta[theta <= 0] = 0
    theta[theta >= 1] = 1

    ind[2] = theta

    print(' cost %0.2f  -  time: %0.2f/%0.2f min' % (
        np.average(np.abs(cost_fft)), (time.time() - tic) / 60, (time.time() - tic0) / 60,))

    # if (it > 1) & (it % dyn_check != 0) & (it % check_point == 0):
    #     show_evolution(history, regionLabels, report_rois, emp_subj+name, output_folder)
    # elif (it > 1) & (it % dyn_check == 0) & (it % check_point == 0):
    #     show_evolution(history, regionLabels, report_rois, emp_subj+name, output_folder, show=["report", "dynFFT", "dynW"])

## Applying retrospective approach to get best w combination
# Maximizing rFC and minimizing FFT cost.
top_n = 10  # Top results to check
transient = 10  # Initial iterations where fc and cost rapidly shrinks

avgFC_history = np.average(np.asarray(history["fc"])[transient:, :],
                           axis=1)  # 10 iterations to remove initial FC decay and high cost
avgCost_history = np.average(np.asarray(history["cost"])[transient:, 0, :], axis=1)

# Look for iterations where FC is maximized - Extract to 10
top_iterations_short = np.argsort(avgFC_history)[-10:]
top_iterations, top_fc, top_cost, top_w = top_iterations_short + transient, avgFC_history[top_iterations_short], \
                                          avgCost_history[top_iterations_short], np.asarray(history["theta"])[
                                              top_iterations_short]

# Then use minimum cost - as control.
best_fc, best_cost, best_w, best_iteration_short, best_iteration = \
    top_fc[np.argmax(top_fc)], top_cost[np.argmax(top_fc)], top_w[np.argmax(top_fc)], top_iterations_short[
        np.argmax(top_fc)], top_iterations[np.argmax(top_fc)]

## Save resutls
## Folder structure - Local
if "LCCN_Local" in os.getcwd():

    output_folder = "E:/LCCN_Local/PycharmProjects/brainModels/CORTICON/GradientResults/"
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)

    specific_folder = output_folder + "/" + emp_subj + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
    if os.path.isdir(specific_folder) == False:
        os.mkdir(specific_folder)

    ### SAVINGS
    ## Save important data  ## To load it back: open(file, 'r') as f; pickle.load()
    file_params = open(specific_folder + "/" + emp_subj + name + "_best.pkl", "wb")
    bests = [best_fc, best_cost, best_w, best_iteration_short, best_iteration]
    pickle.dump(bests, file_params)
    file_params.close()

    ## Save important data  ## To load it back: open(file, 'r') as f; pickle.load()
    file_params = open(output_folder + "/" + emp_subj + name + "_params.pkl", "wb")
    pickle.dump(params, file_params)
    file_params.close()

    file_history = open(output_folder + "/" + emp_subj + name + "_history.pkl", "wb")
    pickle.dump(history, file_history)
    file_history.close()


## Folder structure - CLUSTER
else:
    main_folder = "PSE"
    if os.path.isdir(main_folder) == False:
        os.mkdir(main_folder)

    os.chdir(main_folder)

    specific_folder = "GradientResults" + emp_subj + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
    if os.path.isdir(specific_folder) == False:
        os.mkdir(specific_folder)

    os.chdir(specific_folder)

    ### SAVINGS
    ## Save important data  ## To load it back: open(file, 'r') as f; pickle.load()
    file_params = open(emp_subj + name + "_best.pkl", "wb")
    bests = [best_fc, best_cost, best_w, best_iteration_short, best_iteration]
    pickle.dump(bests, file_params)
    file_params.close()

    ## Save important data  ## To load it back: open(file, 'r') as f; pickle.load()
    file_params = open(emp_subj + name + "_params.pkl", "wb")
    pickle.dump(params, file_params)
    file_params.close()

    file_history = open(emp_subj + name + "_history.pkl", "wb")
    pickle.dump(history, file_history)
    file_history.close()

## TO LOAD BACK:
# with open("heterogeneity/Gradient_results/GD_d11m11y2021-t13h20m_history.pkl", 'rb') as f:
#     history = pickle.load(f)


# # Left rois full report
# report_rois = np.arange(0, len(regionLabels), 4)
# show_evolution(history, regionLabels, report_rois, name, output_folder, show=["dynFFT", "dynW"])
# # Right rois full report
# report_rois = np.arange(1, len(regionLabels), 2)
# show_evolution(history, regionLabels, report_rois, name, output_folder, show=["dynFFT", "dynW"])


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


#
# #####
# ###### TEST 1: copy of JRD jupyter notebook. Good rPLV (short simulations make it less stable due to transient).
#
# working_points = [["jrd_mix", 'NEMOS_075', 0.4561801536874189, 138, 6.5],
#                   ["jrd_mix", 'NEMOS_059', 0.4633034792146057, 78,  4.5],
#                   ["jrd_mix", 'NEMOS_049', 0.2826762031561815, 50,  4.5],
#                   ["jrd_mix", 'NEMOS_035', 0.3655899127321862, 82,  6.5],
#                   ["jrd_mix", 'NEMOS_071', 0.3828224797103641, 126, 16.5],
#                   ["jrd_mix", 'NEMOS_058', 0.3238695846582934, 146, 6.5],
#                   ["jrd_mix", 'NEMOS_050', 0.4568623593946254, 46,  4.5],
#                   ["jrd_mix", 'NEMOS_065', 0.3433422282538486, 80,  6.5],
#                   ["jrd_mix", 'NEMOS_077', 0.3939552674957418, 146, 4.5],
#                   ["jrd_mix", 'NEMOS_064', 0.4236810275102919, 110, 6.5]]
#
# chosen=2
#
# _, emp_subj, r_plv_wp, g, s = working_points[chosen]
#
#
# # This simulation will generate FC for a virtual "Subject".
# # Define identifier (i.e. could be 0,1,11,12,...)
# ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"
#
#
# samplingFreq = 1000  #Hz
# simLength = 30000  # ms - relatively long simulation to be able to check for power distribution
# transient = 10000  # seconds to exclude from timeseries due to initial transient
#
# # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
# integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)
#
# mon = (monitors.Raw(),)
# # subset = [0,1]
# conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2pTh.zip")
# conn.weights = conn.scaled_weights(mode="tract")
#
#
# cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
#              'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
#              'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
#              'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
#              'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
#              'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
#              'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
#              'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
#              'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
#              'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
#              'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
#              'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
#              'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
#              'ParaHippocampal_R', 'Calcarine_L',
#              'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
#              'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
#              'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
#              'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
#              'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
#              'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
#              'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
#              'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
#              'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
#              'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
#              'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
#              'Temporal_Inf_R']
#
#
# FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
# SClabs = list(conn.region_labels)
#
# # For Functiona analysis: remove subcorticals (i.e. Cerebelum, Thalamus, Caudate)
# FC_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
# SC_idx = [SClabs.index(roi) for roi in cortical_rois]
#
# p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])
# sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
#
# n_rois = len(conn.region_labels)
# w = np.asarray([0.8] * n_rois)
#
# # Parameters from DyF
# m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
#                        tau_e1=np.array([10]), tau_i1=np.array([20]),
#                        He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
#                        tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
#
#                        w=w, c=np.array([135.0]),
#                        c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#                        c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#                        v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
#                        p=p_array, sigma=sigma_array)
#
# ## Remember to hold tau*H constant.
# m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
# m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])
#
# # Coupling function
# coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
# conn.speed = np.array([s])
#
# # Run simulation
# tic0 = time.time()
# sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
# sim.configure()
#
# output = sim.run(simulation_length=simLength)
# print("Simulation time: %0.2f sec" % (time.time() - tic0,))
#
# # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
# # Mount PSP output as: w * (vExc1 - vInh1) + (1-w) * (vExc2 - vInh2)
# psp_t = m.w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
#         (1 - m.w) * (output[0][1][transient:, 5, :, 0].T - output[0][1][transient:, 6, :, 0].T)
#
# regionLabels = conn.region_labels
#
# # Check initial transient and cut data
# # DEMEAN: raw_data-np.average(raw_data, axis=1)[:,None]
# # timeseriesPlot(raw_data, raw_time, regionLabels, main_folder, mode="inline", title="timeseries-w="+str(m.w))
# # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="inline", title="FFT")
# # curves = multitapper(raw_data, samplingFreq, regionLabels=regionLabels, epoch_length=4, ntapper=4, smoothing=0.5, plot=True, mode="inline")
#
# print("done!")
#
# all = list()
# # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
# bands = [["3-alpha"], [(8, 12)]]
# for b in range(len(bands[0])):
#
#     (lowcut, highcut) = bands[1][b]
#
#     # Band-pass filtering
#     filterSignals = filter.filter_data(psp_t[SC_idx], samplingFreq, lowcut, highcut, verbose=False)
#
#     # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
#     efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)
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
#     # plotConversions(raw_data[:, :len(efSignals[0][0])]-np.average(raw_data[:, :len(efSignals[0][0])], axis=1)[:,None], efSignals[0], efPhase[0], efEnvelope[0], band=bands[0][b], regionLabels=regionLabels, n_signals=1, raw_time=raw_time)
#
#     # CONNECTIVITY MEASURES
#     ## PLV
#     plv = PLV(efPhase, verbose=False)
#     # aec = AEC(efEnvelope)
#
#     # Load empirical data to make simple comparisons
#     plv_emp = \
#     np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:, FC_idx][FC_idx]
#     # aec_emp = np.loadtxt(wd+"\\CTB_data\\output\\FC_"+emp_subj+"\\"+bands[0][b]+"corramp.txt")[:, fc_rois_cortex][fc_rois_cortex]
#
#     # Comparisons
#     t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
#     t1[0, :] = plv[np.triu_indices(len(plv), 1)]
#     t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
#     plv_r = np.corrcoef(t1)[0, 1]
#
#     all.append(plv_r)
#     print(bands[0][b] + " | " + str(plv_r))
#
# print("%s  ::  Current  rPLV %0.4f  __  WP check %0.4f " % (emp_subj, np.average(all), r_plv_wp))
# print("g%i s%0.1f" % (g, s))
#
#
#
# #
#
#
#
# ######
# ###### TEST 2: copy of desglosadas upper functions. Good rPLV (short simulations make it less stable due to transient).
#
# ## STRUCTURE
# conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh.zip")
# conn.weights = conn.scaled_weights(mode="tract")
#
# ## NMM
# p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])
# sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
#
# # Parameters edited from David and Friston (2003).
# m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
#                        tau_e1=np.array([10]), tau_i1=np.array([20]),
#                        He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
#                        tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
#
#                        w=w, c=np.array([135.0]),
#                        c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
#                        c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
#                        v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
#                        p=p_array, sigma=sigma_array)
#
# # Remember to hold tau*H constant.
# m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
# m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])
#
# ## COUPLING FUNCTION
# coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
# conn.speed = np.array([s])
#
# ## OTHERS
# # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
# integrator = integrators.HeunDeterministic(dt=1000 / 1000)
#
# mon = (monitors.Raw(),)
#
# tic0 = time.time()
#
# ## RUN SIMULATION
# sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
# sim.configure()
#
# output = sim.run(simulation_length=30000)
# transient = 10000
#
# # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
# # Mount PSP output as: w * (vExc1 - vInh1) + (1 - w) * (vExc2 - vInh2)
# psp_t = m.w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
#         (1 - m.w) * (output[0][1][transient:, 5, :, 0].T - output[0][1][transient:, 6, :, 0].T)
#
# # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
# bands = [["3-alpha"], [(8, 12)]]
# for b in range(len(bands[0])):
#
#     (lowcut, highcut) = bands[1][b]
#
#     # Band-pass filtering
#     filterSignals = filter.filter_data(psp_t[SC_cortex_idx], 1000, lowcut, highcut, verbose=False)
#
#     # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
#     efSignals = epochingTool(filterSignals, 4, 1000, "signals", verbose=False)
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
#     # CONNECTIVITY MEASURES
#     ## PLV
#     plv = PLV(efPhase, verbose=False)
#     # aec = AEC(efEnvelope)
#
#     # Load empirical data to make simple comparisons
#     plv_emp = \
#         np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
#         FC_cortex_idx][FC_cortex_idx]
#
#     # Comparisons
#     t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
#     t1[0, :] = plv[np.triu_indices(len(plv), 1)]
#     t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
#     plv_r = np.corrcoef(t1)[0, 1]
#
#     print("%s  ::  Current  rPLV %0.4f  __  WP check %0.4f " % (emp_subj, np.average(all), r_plv_wp))
#     print("g%i s%0.1f" % (g, s))
