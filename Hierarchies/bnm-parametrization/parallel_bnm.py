
import os

import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
import time
from mpi4py import MPI
import datetime

from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2005, JansenRit1995



def parallel_bnm(params_):


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.Data\\"

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
    simLength = 2000  # ms - relatively long simulation to be able to check for power distribution
    samplingFreq = 1000  # Hz
    transient = 1000  # ms to exclude from timeseries due to initial transient

    ## [np.random.randint(426) for i in range(15)]
    # rand_rois = [0, 24, 333, 113, 128, 236, 3, 276, 357, 178, 398, 298, 127, 136, 167, 260]


    # OTHER PARAMETERS   ###
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)


    ### ITERATE OVER THE PARAMETER SPACE
    result = []

    for i, set_ in enumerate(params_):

        tic = time.time()

        mode, g, ff, fb, l, sigma, He, Hi, taue, taui, speed = set_


        print("Simulating for mode: %s - FF%i, FB%i, L%i - sigma%0.2f  ::  %i/%i "
              % (mode, ff, fb, l, sigma, i + 1, len(params_)))


        # STRUCTURAL CONNECTIVITY      #########################################
        surfpack = "HCPex-r426-surfdisc4k_pack/"
        conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
        conn.weights = conn.scaled_weights(mode="tract")
        conn.speed = np.array([speed])  # Following Lemarechal et al. 2022


        if "hier" in mode:

            # NEURAL MASS MODEL    #########################################################
            m = JansenRitDavid2005(He=np.array([He]), Hi=np.array([Hi]),  # From David (2005)
                                   tau_e=np.array([taue]), tau_i=np.array([taui]),  # From Lemarechal (2022)
                                   gamma1_pyr2exc=np.array([50]), gamma2_exc2pyr=np.array([40]),  # From David (2005)
                                   gamma3_pyr2inh=np.array([12]), gamma4_inh2pyr=np.array([12]),
                                   p=np.array([0]), sigma=np.array([sigma]),
                                   e0=np.array([0.0025]), r=np.array([0.56]))

            m.stvar = np.array([1])  # Define where to input the stimulation


            # COUPLING    #########################################################
            aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', dtype=str)

            aV = dict(F=ff, B=fb, L=l)  # Scaling factor for each type of hierarchical connection

            aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
            aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
            aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

            coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([g]), aF=aF, aB=aB, aL=aL)



        elif "classical" in mode:

            # NEURAL MASS MODEL    #########################################################
            m = JansenRit1995(He=np.array([He]), Hi=np.array([Hi]),
                              tau_e=np.array([taue]), tau_i=np.array([taui]),
                              c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                              c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                              p=np.array([0]), sigma=np.array([sigma]),
                              e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

            m.stvar = np.array([1])  # Define where to input the stimulation

            # COUPLING :: Sigmoidal     #########################################################
            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))

        ## RUN SIMULATION
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)

        sim.configure()

        output = sim.run(simulation_length=simLength)

        data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSP
        raw_time = output[0][0][transient:]


        # Signal data
        S_avg = np.average(data[:, -500:], axis=1)
        S_min = np.min(data[:, -500:], axis=1)
        S_max = np.max(data[:, -500:], axis=1)

        intraS_std = np.std(data[:, -500:], axis=1)

        interS_std = np.std(S_avg)


        # Gather spectral data
        peaks, _, band_modules = FFTpeaks(data, simLength, transient, samplingFreq)


        # for roi_id in rand_rois:
        #
        #     result.append([mode, g, ff, fb, l, sigma, roi_id, conn.region_labels[roi_id],
        #                    peaks[roi_id], band_modules[roi_id],
        #                    S_avg[roi_id], S_min[roi_id], S_max[roi_id],
        #                    intraS_std[roi_id], interS_std])

        result.append([mode, g, ff, fb, l, sigma, He, Hi, taue, taui, speed, "avg", "avg",
                       np.average(peaks), np.average(band_modules),
                       np.average(S_avg), np.average(S_min), np.average(S_max),
                       np.average(intraS_std), interS_std])

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)


