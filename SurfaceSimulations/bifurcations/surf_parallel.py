
import time
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
import pickle

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995
from mpi4py import MPI
import datetime

surfpack = "subset-r7-surf408_pack\\"
# surfpack = "default-r76-surf16k_pack\\"

main_dir = "E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\SurfSim\\"

def surf_parallel(params_):
    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))


    # Prepare simulation parameters
    simLength = 2 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms


    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        p, tau_e, tau_i = set

        # NEURAL MASS MODEL    #########################################################
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([tau_e]), tau_i=np.array([tau_i]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([p]), sigma=np.array([0]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        m.stvar = np.array([1])  # Define where to input the stimulation

        # STRUCTURAL CONNECTIVITY      #########################################
        conn = connectivity.Connectivity.from_file(main_dir + surfpack + "connectivity.zip")
        # conn.weights = conn.scaled_weights(mode="tract")

        conn.speed = np.array([3.9])  # Following Lemarechal et al. 2022

        coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))

        # CORTICAL SURFACE        #########################################
        local_conn = main_dir + surfpack + "local_connectivity.mat"
        # local = None

        cx = cortex.Cortex.from_file(source_file=main_dir + surfpack + "cortical_surface.zip",
                                     region_mapping_file=main_dir + surfpack + "region_mapping.txt",
                                     local_connectivity_file=local_conn)  # cutoff=40mm (gaussian)

        cx.region_mapping_data.connectivity = conn
        cx.coupling_strength = np.array([0])

        ##### STIMULUS
        ## Pulse train
        # Mixed stimulus _Regional and Surface
        stim_vertices = [113, 265, 275, 267, 393]
        stim = patterns.StimuliSurfaceRegion(
            temporal=equations.DC(
                parameters=dict(dc_offset=1, t_start=1500.0, t_end=1525.0)),
            spatial=equations.Gaussian(
                parameters=dict(amp=1, sigma=0.001, midpoint=0, offset=0)),

            surface=cx.surface,
            focal_points_surface=np.array(stim_vertices),  # In the order of vertices

            focal_regions=np.array([0]), )  # In the order of unmapped regions

        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating %is  ||  PARAMS: p%i " % (simLength / 1000, p))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, surface=cx, integrator=integrator,
                                  monitors=mon, stimulus=stim)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract gexplore_data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(gexplore_data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.

        raw_data = output[0][1][:, 1, :, 0].T - output[0][1][:, 2, :, 0].T

        data_bif = raw_data[:, transient:transient+500]
        data_t1 = raw_data[:, transient+500:transient+500+25*2]
        data_t2 = raw_data[:, transient+500+25*2:transient+500+25*4]
        data_t3 = raw_data[:, transient+500+25*4:]

        labels = [conn.region_labels[id] + "-" + str(pre) for pre, id in enumerate(cx.region_mapping)]

        ## Gather results
        temp_res = [[labels[i], p, tau_e, tau_i,
                     max(data_bif[i]), min(data_bif[i]),
                     max(data_t1[i]), min(data_t1[i]),
                     max(data_t2[i]), min(data_t2[i]),
                     max(data_t3[i]), min(data_t3[i])] for i in stim_vertices]

        result = result + temp_res

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
