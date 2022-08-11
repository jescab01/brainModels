import time
import numpy as np
import scipy.signal
import scipy.stats
import os

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime

def simulate_parallel(params):
    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        # import sys
        # sys.path.append("E:\\LCCN_Local\PycharmProjects\\")
        # from toolbox.signals import timeseriesPlot, epochingTool
        # from toolbox.fc import PLV
        # from toolbox.fft import FFTpeaks
        from toolbox import PLV, PLE, epochingTool, FFTpeaks
        ctb_folder = "E:\\LCCN_Local\\PycharmProjects\\CTB_data2\\"


    ## Folder structure - CLUSTER
    else:
        from toolbox import PLV, PLE, epochingTool, FFTpeaks
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data2/"

    # Prepare simulation parameters
    simLength = 10 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms

    for set in params:
        print(set)

        emp_subj, mode, v1, v2, r, out, test_params = set

        if test_params == "g&s":

            g, s = v1, v2

            # STRUCTURAL CONNECTIVITY      #########################################
            if '_pTh' in mode:
                conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
            else:
                conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
            conn.weights = conn.scaled_weights(mode="tract")

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
            cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                             'Insula_L', 'Insula_R',
                             'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                             'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                             'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                             'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                             'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                             'Thalamus_L', 'Thalamus_R']

            if "gianluca" not in mode:
                # load text with FC rois; check if match SC
                FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
                FC_cortex_idx = [FClabs.index(roi) for roi in
                                 cortical_rois]  # find indexes in FClabs that matches cortical_rois
                SClabs = list(conn.region_labels)
                SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

            # Subset for Cingulum Bundle
            if "cb" in mode:
                FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
                SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
                conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
                conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
                conn.region_labels = conn.region_labels[SC_cb_idx]


            # NEURAL MASS MODEL    #########################################################
            if "jrd" in mode:  # JANSEN-RIT-DAVID
                if "_def" in mode:
                    sigma_array = 0.022
                    p_array = 0.22
                else:  # for jrd_pTh and jrd modes
                    sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
                    p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])
                # Parameters edited from David and Friston (2003).
                m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                         tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                         He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                         tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                         w=np.array([0.8]), c=np.array([135.0]),
                                         c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                         c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                         v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                         p=np.array([p_array]), sigma=np.array([sigma_array]))
                # Remember to hold tau*H constant.
                m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
                m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

            elif mode=="jr_pTh_wNoise":

                sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
                p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])

                m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                                  tau_e=np.array([10]), tau_i=np.array([16]),
                                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),
                                  p=p_array, sigma=sigma_array)

                # Remember to hold tau*H constant.
                m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])


            else:  # JANSEN-RIT
                # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
                m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                                     p_min=np.array([0]),
                                     r=np.array([0.56]), v0=np.array([6]))

            # COUPLING FUNCTION   #########################################
            if "jrd" in mode:
                coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
            else:
                coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                                   r=np.array([0.56]))
            conn.speed = np.array([s])

            # OTHER PARAMETERS   ###
            # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
            # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
            integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

            mon = (monitors.Raw(),)

            tic = time.time()
            print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)

            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            if "jrd" in mode:
                if "psp" in out:
                    raw_data = m.w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
                           (1 - m.w) * (output[0][1][transient:, 5, :, 0].T - output[0][1][transient:, 6, :, 0].T)
                else:
                    raw_data = m.w * output[0][1][transient:, 0, :, 0].T + (1 - m.w) * output[0][1][transient:, 4, :, 0].T
            else:
                if "psp" in out:
                    raw_data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
                else:
                    raw_data = output[0][1][transient:, 0, :, 0].T
            raw_time = output[0][0][transient:]

            # Extract signals of interest
            if "gianluca" in mode:
                raw_data = raw_data

            elif "cb" not in mode:
                raw_data = raw_data[SC_cortex_idx, :]

            # average signals to obtain mean signal frequency peak
            data = np.asarray([np.average(raw_data, axis=0)])
            data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

            # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
            IAF, module, band_module = FFTpeaks(data, simLength - transient)

            # bands = [["3-alpha"], [(8, 12)]]
            bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

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

                # ## PLE - Phase Lag Entropy
                # ## PLE parameters - Phase Lag Entropy
                # tau_ = 25  # ms
                # m_ = 3  # pattern size
                # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)

                # Load empirical data to make simple comparisons
                if "cb" in mode:
                    plv_emp = \
                        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
                        FC_cb_idx][
                            FC_cb_idx]

                elif "gianluca" in mode:
                    plv_emp = \
                        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')
                else:
                    plv_emp = \
                        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
                        FC_cortex_idx][
                            FC_cortex_idx]

                # Comparisons
                t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
                t1[0, :] = plv[np.triu_indices(len(plv), 1)]
                t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
                plv_r = np.corrcoef(t1)[0, 1]

                ## Gather results
                result.append((emp_subj, mode, g, s, r, out, test_params, IAF[0], module[0], band_module[0], bands[0][b], plv_r))

            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))



        elif test_params=="g&p":

            g, p = v1, v2

            # STRUCTURAL CONNECTIVITY      #########################################
            if '_pTh' in mode:
                conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh.zip")
            else:
                conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2.zip")
            conn.weights = conn.scaled_weights(mode="tract")

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
            cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                             'Insula_L', 'Insula_R',
                             'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                             'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                             'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                             'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                             'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                             'Thalamus_L', 'Thalamus_R']

            # load text with FC rois; check if match SC
            FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
            FC_cortex_idx = [FClabs.index(roi) for roi in
                             cortical_rois]  # find indexes in FClabs that matches cortical_rois
            SClabs = list(conn.region_labels)
            SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

            # Subset for Cingulum Bundle
            if "cb" in mode:
                FC_cb_idx = [FClabs.index(roi) for roi in
                             cingulum_rois]  # find indexes in FClabs that matches cortical_rois
                SC_cb_idx = [SClabs.index(roi) for roi in
                             cingulum_rois]  # find indexes in FClabs that matches cortical_rois
                conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
                conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
                conn.region_labels = conn.region_labels[SC_cb_idx]

            # NEURAL MASS MODEL    #########################################################
            if "jrd" in mode:  # JANSEN-RIT-DAVID
                if "_def" in mode:
                    sigma_array = 0.022
                    p_array = p
                else:  # for jrd_pTh and jrd modes
                    sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
                    p_array = np.asarray([0.22 if 'Thal' in roi else p for roi in conn.region_labels])
                # Parameters edited from David and Friston (2003).
                m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                       tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                       He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                       tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                       w=np.array([0.8]), c=np.array([135.0]),
                                       c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                       c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                       v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                       p=np.array([p_array]), sigma=np.array([sigma_array]))

                # Remember to hold tau*H constant.
                m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
                m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

            elif mode == "jr_pTh_wNoise":

                sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
                p_array = np.asarray([0.22 if 'Thal' in roi else p for roi in conn.region_labels])

                m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                                  tau_e=np.array([10]), tau_i=np.array([16]),
                                  c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                                  c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                                  e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),
                                  p=p_array, sigma=sigma_array)

                # Remember to hold tau*H constant.
                m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])


            else:  # JANSEN-RIT
                # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
                m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                                     p_min=np.array([0]),
                                     r=np.array([0.56]), v0=np.array([6]))

            # COUPLING FUNCTION   #########################################
            if "jrd" in mode:
                coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
            else:
                coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                                   r=np.array([0.56]))
            conn.speed = np.array([10])

            # OTHER PARAMETERS   ###
            # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
            # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
            integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

            mon = (monitors.Raw(),)

            tic = time.time()
            print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
            sim.configure()
            output = sim.run(simulation_length=simLength)

            # Extract data: "output[a][b][:,0,:,0].T" where:
            # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
            if "jrd" in mode:
                if "psp" in out:
                    raw_data = m.w * (output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T) + \
                               (1 - m.w) * (output[0][1][transient:, 5, :, 0].T - output[0][1][transient:, 6, :, 0].T)
                else:
                    raw_data = m.w * output[0][1][transient:, 0, :, 0].T + (1 - m.w) * output[0][1][transient:, 4, :,
                                                                                       0].T
            else:
                if "psp" in out:
                    raw_data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
                else:
                    raw_data = output[0][1][transient:, 0, :, 0].T
            raw_time = output[0][0][transient:]

            # Extract signals of interest
            if "cb" not in mode:
                raw_data = raw_data[SC_cortex_idx, :]

            # average signals to obtain mean signal frequency peak
            data = np.asarray([np.average(raw_data, axis=0)])
            data = np.concatenate((data, raw_data),
                                  axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

            # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
            IAF, module, band_module = FFTpeaks(data, simLength - transient)

            # bands = [["3-alpha"], [(8, 12)]]
            bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"],
                     [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

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

                # ## PLE - Phase Lag Entropy
                # ## PLE parameters - Phase Lag Entropy
                # tau_ = 25  # ms
                # m_ = 3  # pattern size
                # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)

                # Load empirical data to make simple comparisons
                if "cb" in mode:
                    plv_emp = \
                        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt",
                                   delimiter=',')[:,
                        FC_cb_idx][
                            FC_cb_idx]
                else:
                    plv_emp = \
                        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt",
                                   delimiter=',')[:,
                        FC_cortex_idx][
                            FC_cortex_idx]

                # Comparisons
                t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
                t1[0, :] = plv[np.triu_indices(len(plv), 1)]
                t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
                plv_r = np.corrcoef(t1)[0, 1]

                ## Gather results
                result.append((emp_subj, mode, g, p, r, out, test_params, IAF[0], module[0], band_module[0], bands[0][b], plv_r))

            print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
