import os

import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
import time
from mpi4py import MPI
import datetime

from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2005



def parallel_D5pse(params_):


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        data_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\.Data\\"

        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import FFTpeaks


    ## Folder structure - CLUSTER
    elif "t192" in os.getcwd():
        wd = "/home/t192/t192950/mpi/"
        data_dir = wd + "SURFdata/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import FFTpeaks


    ## Folder structure - CLUSTER BRIGIT
    else:
        wd = "/mnt/lustre/home/jescab01/"
        data_dir = wd + "SURFdata/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import FFTpeaks



    ## Simulation parameters
    simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
    samplingFreq = 1000  # Hz
    transient = 1000  # ms to exclude from timeseries due to initial transient




    # NEURAL MASS MODEL    #########################################################
    m = JansenRitDavid2005(He=np.array([3.25]), Hi=np.array([29.3]),  # From David (2005)
                           tau_e=np.array([5.6]), tau_i=np.array([7.3]),  # From Lemarechal (2022)
                           gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                           gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                           p=np.array([0]), sigma=np.array([0]),
                           e0=np.array([0.0025]), r=np.array([0.56]))

    m.stvar = np.array([1])  # Define where to input the stimulation


    # OTHER PARAMETERS   ###
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)


    ### ITERATE OVER THE PARAMETER SPACE
    result = []

    for i, set_ in enumerate(params_):

        tic = time.time()

        mode, ff, fb = set_

        n_rois, sc_mode = 2, "bin"

        print("Simulating for mode: %s - FF%i and FB%i  ::  %i/%i " % (mode, ff, fb, i + 1, len(params_)))

        # STRUCTURAL CONNECTIVITY      #########################################
        surfpack = "defsub-r7-surf408_pack/"
        conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
        conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022
        # conn.weights = np.ones(shape=conn.weights.shape)

        # Subset the relevant data to the number of regions in mode
        conn.weights = conn.weights[:, :n_rois][:n_rois]
        conn.tract_lengths = conn.tract_lengths[:, :n_rois][:n_rois]
        conn.centres = conn.centres[:n_rois]
        conn.region_labels = conn.region_labels[:n_rois]
        conn.cortical = conn.cortical[:n_rois]

        # Transform weights
        if ("tract" or "region") in sc_mode:
            conn.weights = conn.scaled_weights(mode=sc_mode)
        else:
            conn.weights = conn.binarized_weights

        # COUPLING    #########################################################
        aV = dict(F=ff, B=fb, L=0)  # Scaling factor for each type of hierarchical connection

        if "r2" in mode:
            r1, g = 1, 1
            aM = np.array(   # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                        # FROM
                [[0,  "B"],
                 ["F", 0]])

            aM = aM[:, :n_rois][:n_rois]

            aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
            aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([0]), aF=aF, aB=aB, aL=aL)

        elif "r3lin" in mode:
            r1, g = 2, 0.5
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d",   "FB", 0, 0, 0, 0, 0],
                 ["FF", "d",  "FB", 0, 0, 0, 0],
                 [0,   "FF", "d", 0, 0, 0, 0],
                 [0,    0,   0, "d", 0, 0, 0],  # TO
                 [0,    0,   0, 0, "d", 0, 0],
                 [0,    0,   0, 0, 0, "d", 0],
                 [0,    0,   0, 0, 0, 0, "d"]])


            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


        elif "r3all" in mode:
            r1, g = 2, 0.5
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d", "FB", "FB", 0, 0, 0, 0],
                 ["FF", "d", "FB", 0, 0, 0, 0],
                 ["FF", "FF", "d", 0, 0, 0, 0],
                 [0, 0, 0, "d", 0, 0, 0],  # TO
                 [0, 0, 0, 0, "d", 0, 0],
                 [0, 0, 0, 0, 0, "d", 0],
                 [0, 0, 0, 0, 0, 0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


        elif "r5" in mode:
            r1, g = 4, 0.25
            aM = np.array(  # Matrix determining the type of hierarchical connection (FF:1, FB:2, L:3, ns:0)
                # FROM
                [["d",  "FB", "FB", "FB",  0,   0,  0],
                 ["FF", "d",  "FB",  0,   "FB", 0,  0],
                 ["FF", "FF", "d",  "FF", "FB", 0,  0],
                 ["FF",  0,   "FB", "d",  "FB", 0,  0],  # TO
                 [0,    "FF", "FF", "FF", "d",  0,  0],
                 [0,     0,    0,    0,    0,  "d", 0],
                 [0,     0,    0,    0,    0,   0, "d"]])

            aF = np.array([[aV["FF"] if val == "FF" else 0 for val in row] for row in aM])
            aB = np.array([[aV["FB"] if val == "FB" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)


        # STIMULUS    #########################################################
        weighting = np.zeros((len(conn.region_labels),))
        weighting[[0]] = 0.1

        stim = patterns.StimuliRegion(
            # temporal=equations.PulseTrain(
            #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
            temporal=equations.DC(parameters=dict(dc_offset=1, t_start=2000.0, t_end=2025.0)),
            weight=weighting, connectivity=conn)  # In the order of unmapped regions


        ## RUN SIMULATION
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                  stimulus=stim)

        sim.configure()

        output = sim.run(simulation_length=simLength)

        data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSP
        raw_time = output[0][0][transient:]

        # Gather spectral data
        peaks_pre, _, band_modules_pre = FFTpeaks(data[:, :1000], transient+1000, transient, samplingFreq)
        peaks_post, _, band_modules_post = FFTpeaks(data[:, 1000:2000], transient+1000, transient, samplingFreq)


        # Gather duration for region A and last region
        signal_peaks = scipy.signal.find_peaks(data[0, 1000:2000])[0]
        signal_peaks_amps = [data[0, int(1000+peak_id)] for peak_id in signal_peaks]
        if signal_peaks_amps:
            ids = np.array(signal_peaks)[signal_peaks_amps > signal_peaks_amps[0]*0.05]
            if ids.any():
                duration_r0 = max(raw_time[1000:2000][ids])-2000  # duration in ms
        else:
            duration_r0 = np.nan

        # Gather duration for last region
        signal_peaks = scipy.signal.find_peaks(data[r1, 1000:2000])[0]
        signal_peaks_amps = [data[0, int(1000+peak_id)] for peak_id in signal_peaks]
        if signal_peaks_amps:
            ids = np.array(signal_peaks)[signal_peaks_amps > signal_peaks_amps[0]*0.05]
            if ids.any():
                duration_r1 = max(raw_time[1000:2000][ids])-2000  # duration in ms
        else:
            duration_r1 = np.nan


        result.append([mode, g, ff, fb,
                        peaks_pre[0], peaks_pre[r1], peaks_post[0], peaks_post[r1],
                        band_modules_pre[0], band_modules_pre[r1], band_modules_post[0], band_modules_post[r1],
                        duration_r0, duration_r1])

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)


