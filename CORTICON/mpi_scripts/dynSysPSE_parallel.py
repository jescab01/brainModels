
import time
import numpy as np
import scipy.signal
import scipy.stats

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime


def simulate_dynSysPSE(params_):

    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import multitapper
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.mixes import timeseries_spectra

    ## Folder structure - CLUSTER
    else:
        from toolbox import multitapper, PLV, epochingTool
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data2/"

    # Prepare simulation parameters
    simLength = 30 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 10000  # ms

    for i, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, i+1, len(params_)))

        print(set)
        space, model, struct, emp_subj, rois, r, g, s, w, p, sigma = set
        # "p", "jr", "AAL2", "NEMOS_035", [80,93], 0, 0, 15, 0.8, 0.22, 0.022

        # STRUCTURAL CONNECTIVITY      #########################################
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_" + struct + ".zip")
        conn.weights = conn.scaled_weights(mode="tract")

        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
        SClabs = list(conn.region_labels)

        if "Net" not in space:  # If we want to simulate with just a subset
            conn.weights = conn.weights[:, rois][rois]
            conn.tract_lengths = conn.tract_lengths[:, rois][rois]
            conn.region_labels = conn.region_labels[rois]

        elif "cbNet" in space:
            cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R'] + [roi for roi in conn.region_labels if 'Thal' in roi]
            cb_rois = [SClabs.index(roi) for roi in cingulum_rois]

            conn.weights = conn.weights[:, cb_rois][cb_rois]
            conn.tract_lengths = conn.tract_lengths[:, cb_rois][cb_rois]
            conn.region_labels = conn.region_labels[cb_rois]

            cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R', 'Insula_L', 'Insula_R',
                             'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                             'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                             'ParaHippocampal_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                             'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R']  # Removing subcorticals for FC analysis
            FC_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cb_rois
            SC_idx = [SClabs.index(roi) for roi in cingulum_rois]

        elif "bigNet" in space:
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
                         'Temporal_Inf_R'] # For Functiona analysis: remove subcorticals (i.e. Cerebelum, Thalamus, Caudate)
            FC_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
            SC_idx = [SClabs.index(roi) for roi in cortical_rois]


        # NEURAL MASS MODEL  &    COUPLING FUNCTION         ###################################################

        if "mix" in model:
            p_array = np.asarray([0.22 if 'Thal' in roi else p for roi in conn.region_labels])
            sigma_array = np.asarray([sigma if 'Thal' in roi else 0 for roi in conn.region_labels])

        elif "def" in model:
            p_array = np.asarray([0.22 if 'Thal' in roi else 0.22 for roi in conn.region_labels])
            sigma_array = np.asarray([0.022 if 'Thal' in roi else 0.022 for roi in conn.region_labels])

        elif "custom" in model:
            p_array = np.asarray([p if 'Thal' in roi else p for roi in conn.region_labels])
            sigma_array = np.asarray([sigma if 'Thal' in roi else sigma for roi in conn.region_labels])

        if "jrd_" in model:  # JANSEN-RIT-DAVID
            # Parameters edited from David and Friston (2003).
            m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                   tau_e1=np.array([10]), tau_i1=np.array([20]),
                                   He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                   tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                   w=np.array([w]), c=np.array([135.0]),
                                   c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                   c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                   v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                   p=p_array, sigma=sigma_array,

                                   variables_of_interest=["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                                                          "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"])

            coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

            # # Remember to hold tau*H constant: Spiegler (2010) pp.1045;
            m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
            m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        elif "jr_" in model:  # JANSEN-RIT
            m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                              tau_e=np.array([10]), tau_i=np.array([20]),

                              c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                              c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                              c=np.array([135.0]), p=p_array, sigma=sigma_array,
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

        print("Simulating %s (%is)  || structure: %s \nPARAMS: g%i s%i w%0.2f p%0.4f sigma%0.4f" %
              (model, simLength/1000, struct, g, s, w, p, sigma))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        if "jr_" in model:
            psp_t = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
            psp_dt = output[0][1][transient:, 4, :, 0].T - output[0][1][transient:, 5, :, 0].T

        elif "jrd_" in model:
            psp_t = w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
                    (1 - w) * (output[0][1][transient:, 7, :, 0].T - output[0][1][transient:, 8, :, 0].T)
            psp_dt = w * (output[0][1][transient:, 4, :, 0].T - output[0][1][transient:, 5, :, 0].T) + \
                     (1 - w) * (output[0][1][transient:, 10, :, 0].T - output[0][1][transient:, 11, :, 0].T)



        fft = []
        lowcut, highcut = 0, 60
        for signal in psp_t:
            # average signals to obtain mean signal frequency peak
            fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
            fft.append(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT

        freqs = np.arange(len(psp_t[0]) / 2)
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        fft = np.asarray(fft)

        fft = fft[:, (freqs > lowcut) & (freqs < highcut)]  # remove undesired frequencies
        # freqs = freqs[(freqs > lowcut) & (freqs < highcut)]

        # timeseries_spectra(raw_data, simLength, transient, regionLabels, mode="html", folder="figures",
        #                freqRange=[2,40], title=None, auto_open=True)

        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
        # _, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)


        # Minimize the saved signal after FFT
        psp_t_list, psp_dt_list = [], []
        for i in range(len(psp_t)):

            if np.max(psp_t[i])-np.min(psp_t[i]) > 2*3*sigma:
                # id_start = scipy.signal.argrelextrema(np.abs(psp_dt[i]), np.less)[0][-7]
                # id_end = scipy.signal.argrelextrema(np.abs(psp_dt[i]), np.less)[0][-1]
                psp_t_list.append(psp_t[i, -300:])
                psp_dt_list.append(psp_dt[i, -300:])

            else:
                psp_t_list.append(psp_t[i, -1])
                psp_dt_list.append(psp_dt[i, -1])

        if "plv" in space:

            # Extract signals of interest
            psp_t = psp_t[SC_idx, :]
            psp_dt = psp_dt[SC_idx, :]

            bands = [["3-alpha"], [(8, 12)]]
            # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

            for b in range(len(bands[0])):
                (lowcut, highcut) = bands[1][b]

                # Band-pass filtering
                filterSignals = filter.filter_data(psp_t, samplingFreq, lowcut, highcut)

                # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
                efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

                # Obtain Analytical signal
                efPhase = list()
                efEnvelope = list()
                for i in range(len(efSignals)):
                    analyticalSignal = scipy.signal.hilbert(efSignals[i])
                    # Get instantaneous phase and amplitude envelope by channel
                    efPhase.append(np.angle(analyticalSignal))
                    efEnvelope.append(np.abs(analyticalSignal))

                # Check point
                # from toolbox import timeseriesPlot, plotConversions
                # regionLabels = conn.region_labels
                # timeseriesPlot(raw_data, raw_time, regionLabels)
                # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

                # CONNECTIVITY MEASURES
                ## PLV
                plv = PLV(efPhase)

                # Load empirical data to make simple comparisons

                plv_emp = np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:, FC_idx][FC_idx]

                # Comparisons
                t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
                t1[0, :] = plv[np.triu_indices(len(plv), 1)]
                t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
                plv_r = np.corrcoef(t1)[0, 1]

                ## Gather results
                result.append((space, model, struct, emp_subj, rois, r, g, s, w, p, sigma, fft[rois, :], simLength - transient, plv_r))

        else:
            result.append((space, model, struct, emp_subj, rois, r, g, s, w, p, sigma, psp_t_list, psp_dt_list, fft, simLength - transient))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
