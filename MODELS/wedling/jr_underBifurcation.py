import time
import numpy as np
import pandas as pd
from mne import filter
import scipy
import pickle
import plotly.express as px

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots

#from toolbox import timeseriesPlot, FFTplot, PLV, epochingTool, plotConversions, AEC, FFTpeaks, paramSpace, multitapper
import sys
sys.path.append("E:/LCCN_Local/PycharmProjects")
from toolbox.signals import timeseriesPlot, epochingTool
from toolbox.fft import FFTpeaks, multitapper
from toolbox.fc import PLV
from toolbox.mixes import timeseries_spectra


working_points = [["jrd_mix", 'NEMOS_075', 0.4561801536874189, 138, 6.5],
                  ["jrd_mix", 'NEMOS_059', 0.4633034792146057, 78,  4.5],
                  ["jrd_mix", 'NEMOS_049', 0.2826762031561815, 50,  4.5],
                  ["jrd_mix", 'NEMOS_035', 0.3655899127321862, 82,  6.5],
                  ["jrd_mix", 'NEMOS_071', 0.3828224797103641, 126, 16.5],
                  ["jrd_mix", 'NEMOS_058', 0.3238695846582934, 146, 6.5],
                  ["jrd_mix", 'NEMOS_050', 0.4568623593946254, 46,  4.5],
                  ["jrd_mix", 'NEMOS_065', 0.3433422282538486, 80,  6.5],
                  ["jrd_mix", 'NEMOS_077', 0.3939552674957418, 146, 4.5],
                  ["jrd_mix", 'NEMOS_064', 0.4236810275102919, 110, 6.5],
                  ["jrd_mix", 'NEMOS_064', 0.4236810275102919, 110, 6.5]]

struct_th = ""

dynamic_fft_data = np.ndarray((1, 5))
dynamic_signal_data = np.ndarray((1, 5))
minmax = []

results = []
for wp in working_points[:1]:

    for g in np.arange(30, 35, 0.25):

        tic0 = time.time()

        _, emp_subj, r_plv_wp, _, _ = wp

        # This simulation will generate FC for a virtual "Subject".
        # Define identifier (i.e. could be 0,1,11,12,...)
        ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"

        samplingFreq = 1000  # Hz
        simLength = 6000  # ms - relatively long simulation to be able to check for power distribution
        transient = 2000  # seconds to exclude from timeseries due to initial transient

        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

        mon = (monitors.Raw(),)
        # rois = [140, 2]
        # rois = [81, 2]
        conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL2"+struct_th+"_pass.zip")
        conn.weights = conn.scaled_weights(mode="tract")

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

        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
        SClabs = list(conn.region_labels)

        # For Functional analysis: remove subcorticals (i.e. Cerebelum, Thalamus, Caudate)
        FC_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
        SC_idx = [SClabs.index(roi) for roi in cortical_rois]


        ## NEURAL MASS MODEL
        p_array = np.asarray([0 if 'Thal' in roi else 0 for roi in conn.region_labels])
        sigma_array = np.asarray([0.03 if 'Thal' in roi else 0 for roi in conn.region_labels])

        n_rois = len(conn.region_labels)

        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),

                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                          c=np.array([135.0]), p=p_array, sigma=sigma_array,
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),

                          variables_of_interest=["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"])

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

        # Coupling function
        conn.speed = np.array([14])

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()

        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.2f sec" % (time.time() - tic0,))

        # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
        # Mount PSP output as: w * (vExc1 - vInh1) + (1-w) * (vExc2 - vInh2)
        psp_t = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T

        regionLabels = conn.region_labels

        # Check initial transient and cut data
        # DEMEAN: raw_data-np.average(raw_data, axis=1)[:,None]
        # timeseriesPlot(psp_t[0:2], output[0][0][transient:], regionLabels[0:2], mode="inline", title="timeseries-sigmaJR")
        # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="inline", title="FFT")
        # curves = multitapper(raw_data, samplingFreq, regionLabels=regionLabels, epoch_length=4, ntapper=4, smoothing=0.5, plot=True, mode="inline")
        # timeseries_spectra(psp_t[rois], simLength, transient, regionLabels[rois], folder="PSE", mode="inline")

        peaks, modules, band_modules = FFTpeaks(psp_t, simLength, curves=False)

        subj_fft, subj_s, regLabs_fft, regLabs_s, gs_fft, gs_s, sign_tot, times_tot, fft_tot, freq_tot = \
            [], [], [], [], [], [], [], [], [], []

        for i in range(len(psp_t)):

            fft = abs(np.fft.fft(psp_t[i]))  # FFT for each channel signal
            fft = fft[range(int(len(psp_t[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(psp_t[i]) / 2)
            freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

            fft = fft[(freqs > 0.5) & (freqs < 65)]  # remove undesired frequencies from peak analisis
            freqs = freqs[(freqs > 0.5) & (freqs < 65)]

            fft_tot += list(fft)
            freq_tot += list(freqs)
            regLabs_fft += [regionLabels[i]] * len(fft)
            gs_fft += [g] * len(fft)
            subj_fft += [emp_subj] * len(fft)

            minmax.append([emp_subj, regionLabels[i], g, min(psp_t[i]), max(psp_t[i])])

            # sign_tot += list(psp_t[i])
            # times_tot += list(output[0][0][transient:])
            # regLabs_s += [regionLabels[i]] * len(psp_t[i])
            # gs_s += [g] * len(psp_t[i])
            # subj_s += [emp_subj] * len(psp_t[i])

        temp = np.asarray([subj_fft, regLabs_fft, gs_fft, fft_tot, freq_tot]).transpose()
        dynamic_fft_data = np.concatenate((dynamic_fft_data, temp))

        # temp1 = np.asarray([subj_s, regLabs_s, gs_s, sign_tot, times_tot]).transpose()
        # dynamic_signal_data = np.concatenate((dynamic_signal_data, temp1))

        # print("done!")

        all = list()
        # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
        bands = [["3-alpha"], [(8, 12)]]
        for b in range(len(bands[0])):

            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(psp_t[SC_idx], samplingFreq, lowcut, highcut, verbose=False)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.angle(analyticalSignal))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # plotConversions(raw_data[:, :len(efSignals[0][0])]-np.average(raw_data[:, :len(efSignals[0][0])], axis=1)[:,None], efSignals[0], efPhase[0], efEnvelope[0], band=bands[0][b], regionLabels=regionLabels, n_signals=1, raw_time=raw_time)

            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase, verbose=False)
            # aec = AEC(efEnvelope)

            # Load empirical data to make simple comparisons
            plv_emp = \
            np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:, FC_idx][FC_idx]
            # aec_emp = np.loadtxt(wd+"\\CTB_data\\output\\FC_"+emp_subj+"\\"+bands[0][b]+"corramp.txt")[:, fc_rois_cortex][fc_rois_cortex]

            # Comparisons
            t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
            t1[0, :] = plv[np.triu_indices(len(plv), 1)]
            t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
            plv_r = np.corrcoef(t1)[0, 1]

            all.append(plv_r)
            print(bands[0][b] + " | " + str(plv_r))

        # results.append([emp_subj, struct_th, r_plv_wp, g, psp_t, output[0][0][transient:], ffts, freqs, peaks, modules, plv_r])
        results.append([emp_subj, struct_th, r_plv_wp, g, psp_t, output[0][0][transient:], peaks, modules, plv_r])
        print("%s  g%i   ::  Current  rPLV %0.4f  __  WP check %0.4f \n" % (emp_subj, g, np.average(all), r_plv_wp))


### SAVING - PLOTTING
simname = "JR"+struct_th+"-"+time.strftime("m%md%dy%Y")

main_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\wedling\PSE\\"
specific_folder = main_folder + "\\PSEunderBifurcation_" + simname

if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)


print("saving...")
# dynamic ffts
dyndf_fft = pd.DataFrame(dynamic_fft_data[1:, ], columns=["subj", "roi", "g", "fft", "freqs"])
dyndf_fft = dyndf_fft.astype({"subj": str, "roi": str, "g": float, "fft": float, "freqs": float})
# dyndf_fft.to_csv(specific_folder+"/PSE_"+simname+"-dynamicFFTdf.csv", index=False)

## dynamic signals
# dyndf_s = pd.DataFrame(dynamic_signal_data[1:, ], columns=["subj", "roi", "g", "signal", "time"])
# dyndf_s = dyndf_s.astype({"subj": str, "roi": str, "g": float, "signal": float, "time": float})
# dyndf_s.to_csv(specific_folder+"/PSE_"+simname+"-dynamicSignalsdf.csv", index=False)

# minmax - bifurcation
minmax_df = pd.DataFrame(minmax, columns=["subj", "roi", "g", "min", "max"])
minmax_df = minmax_df.astype({"subj": str, "roi": str, "g": float, "min": float, "max": float})

with open(specific_folder + "\\" + simname + "_output.pkl", "wb") as f:
    pickle.dump([results, dyndf_fft, minmax_df], f)


## Plot dynamical
fig = px.line(dyndf_fft, x="freqs", y="fft", animation_frame="g", animation_group="roi", color="roi",
              title="Dynamic FFT @ " + emp_subj + struct_th)
pio.write_html(fig, file=specific_folder+"/dynFFT_@%s.html" % simname, auto_open=True, auto_play=False)  # auto_open="False")

## Plot bifurcation
fig = go.Figure()
cmap = px.colors.qualitative.Plotly
for i, roi in enumerate(regionLabels):

    subset = minmax_df.loc[minmax_df["roi"] == roi]

    fig.add_trace(go.Scatter(x=subset["g"], y=subset["min"], name=roi, legendgroup=roi, marker=dict(color=cmap[i % len(cmap)])))
    fig.add_trace(go.Scatter(x=subset["g"], y=subset["max"], name=roi, legendgroup=roi, marker=dict(color=cmap[i % len(cmap)]), showlegend=False))

fig.update_layout(title="Bifurcation diagrams", xaxis=dict(title="Coupling factor (g)"), yaxis=dict(title="Voltage"))
pio.write_html(fig, file=specific_folder+"/bifurcations_@%s.html" % simname, auto_open=True)  # auto_open="False")
fig.show(renderer="browser")

## Plot signals before, at and after bifurcation
fig = make_subplots


