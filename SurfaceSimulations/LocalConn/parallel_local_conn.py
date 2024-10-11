
import os

import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt
import time
from mpi4py import MPI
import datetime

from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2005



def parallel_local_conn(params_):


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\.DataTemp\\"

    ## Folder structure - CLUSTER
    elif "t192" in os.getcwd():
        wd = "/home/t192/t192950/mpi/"
        data_dir = wd + "SURFdata/"

    ## Folder structure - CLUSTER BRIGIT
    else:
        wd = "/mnt/lustre/home/jescab01/"
        data_dir = wd + "SURFdata/"


    ## Simulation parameters
    simLength = 4000  # ms - relatively long simulation to be able to check for power distribution
    samplingFreq = 1000  # Hz
    transient = 0  # ms to exclude from timeseries due to initial transient



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

        surf, lc, g = set_

        surfpack = "HCPex-r426-surf%s_pack/" % surf


        # 1.1 Determine the LC file to use
        # a. Open surface info to obtain mean edge length
        with open(data_dir + surfpack + "surface_info.txt", "r") as file:
            surfinfo = file.read()
        file.close()
        avg_edge_length = float(surfinfo.splitlines()[25].split(' ')[-2])

        # b. LC Kernel parameters
        amp = 1
        sigma = avg_edge_length
        cutoff = avg_edge_length + 0.5 * avg_edge_length

        # c. LC name
        lc_title = "local_connectivity-amp%0.1fsig%0.2fcut%0.2f" % (amp, sigma, cutoff)


        poi = [1474, 462, 752, 494, 1327] if "disc4k" in surfpack else \
            [3556, 1594, 252, 951, 2549] if "disc8k" in surfpack else\
                [3556] if "disc17k" in surfpack else []

        print("Simulating for mode: %s sigma%0.2f - lc%0.2e  ::  %i/%i "
              % (surf, sigma, lc, i + 1, len(params_)))


        # STRUCTURAL CONNECTIVITY      #########################################
        conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
        conn.weights = conn.scaled_weights(mode="tract")
        conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022


        # CORTICAL SURFACE        #########################################
        # look for local connectivity file, in case created previously
        local_conn = data_dir + surfpack + "local_conn/" + lc_title + ".mat"

        cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                                     region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                                     local_connectivity_file=local_conn)

        cx.region_mapping_data.connectivity = conn

        cx.coupling_strength = np.array([lc])


        # COUPLING    #########################################################
        aV = dict(F=0.3, B=0.6, L=0.15)  # Scaling factor for each type of hierarchical connection

        aM = np.loadtxt(data_dir + 'HCPex_hierarchy_proxy.txt', dtype=str)
        aM = aM

        aF = np.array([[aV["F"] if val == "F" else 0 for val in row] for row in aM])
        aB = np.array([[aV["B"] if val == "B" else 0 for val in row] for row in aM])
        aL = np.array([[aV["L"] if val == "L" else 0 for val in row] for row in aM])

        coup = coupling.SigmoidalJansenRitDavid2005(a=np.array([0]), aF=aF, aB=aB, aL=aL)


        # STIMULUS    #########################################################
        # focal_points = [1273, 1799, 1667, 1882, 2078] if "surf4k" in surfpack else [4310, 3111, 4089, 4290] if "surf8k" in surfpack else [7165, 8662, 8278] if "surf17k" in surfpack else []

        stim = patterns.StimuliSurfaceRegion(
            # temporal=equations.PulseTrain(
            #     parameters=dict(onset=1500, offset=1800, T=400, tau=25, amp=0.1)),
            temporal=equations.DC(parameters=dict(dc_offset=0.1, t_start=100.0, t_end=125.0)),
            spatial=equations.Gaussian(parameters=dict(amp=1, sigma=1, midpoint=0, offset=0)),
            surface=cx.surface,
            focal_points_surface=np.array([poi[0]], dtype=int),
            # focal_points_surface=np.array([7165, 8662, 8278], dtype=int),  # 17k V1
            # focal_points_surface=np.array([4310, 3111, 4089, 4290], dtype=int),  # (8k V1) || In the order of vertices
            # focal_points_surface=np.array([1273, 1799, 1667, 1882, 2078], dtype=int),  # 4k V1
            focal_regions=np.array([], dtype=int), )  # In the order of conn.region_labels


        # INITIAL CONDITIONS      ###
        # DavidFriston2005 use to have equilibrium in prebif at 0; therefore using 0-init.
        n_subcx = len(set(range(len(conn.region_labels))).difference(cx.region_mapping_data.array_data))
        init = np.zeros((100, 8, len(cx.region_mapping_data.array_data) + n_subcx, 1))


        ## RUN SIMULATION
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, surface=cx,
                                  integrator=integrator, monitors=mon, stimulus=stim, initial_conditions=init)
        sim.configure()

        output = sim.run(simulation_length=simLength)

        data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSP
        raw_time = output[0][0][transient:]


        # Obtain max and time to max per roi
        for point in poi:

            peak = np.max(np.abs(data[point, :]))

            tp_peak = np.argmax(np.abs(data[point, :]))

            intraS_std = np.std(data[point, :])
            interS_std = np.std(np.average(data, axis=0))

            result.append([surf, lc, sigma, g, point,   peak, tp_peak, intraS_std, interS_std])


        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)


