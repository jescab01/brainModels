import pandas as pd
from mpi4py import MPI
import numpy as np
from parallel_hier_brain import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python mpi_hier_brain.py
"""

name = "hierarchiesBrain"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


nrois_vals = [426]  # 180, 360, 426
sc_vals = ["tract"]
g_vals = [1]

hier_vals = [(90, 50), (60, 60), (50, 90),  # L = 1/2 * min (FF, FB)
             (75, 20), (40, 40), (20, 75),
             (65, 15), (25, 25), (10, 70),
             (18, 18)]

params = [(n, sc, g, ff, fb, 5, r0) for n in nrois_vals for sc in sc_vals for g in g_vals
          for ff, fb in hier_vals for r0 in range(nrois_vals[0])]

params = np.asarray(params, dtype=object)
n = params.shape[0]

## Distribution of task load in ranks
count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count

local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank

local_results = parallel_hier_brain(local_params)  # run the function for each parameter set and rank

if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process

        # if i < remainder:
        #     rank_size = count + 1
        # else:
        #     rank_size = count
        # tmp = np.empty((rank_size, final_results.shape[1]))  # create empty array to receive results

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            # print(final_results.shape)
            # print(tmp.shape)  # debugging
            # print(i)

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    # print("Results")
    # print(final_results)

    fResults_df = pd.DataFrame(final_results,
                               columns=["n_rois", "sc_mode", "g", "FF", "FB", "L", "r0", "r1",
                                        "peaks_pre_r0", "peaks_pre_r1",
                                        "peaks_post_r0", "peaks_post_r1",
                                        "band_modules_pre_r0", "band_modules_pre_r1",
                                        "band_modules_post_r0", "band_modules_post_r1",
                                        "duration_r0", "duration_r1"])

    ## Save resutls
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        wd = os.getcwd()

        main_folder = wd + "\\" + "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)
        specific_folder = main_folder + "\\PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        fResults_df.to_csv(specific_folder + "/results.csv", index=False)

    ## Folder structure - CLUSTER
    else:
        main_folder = "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)

        os.chdir(main_folder)

        specific_folder = "PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        os.chdir(specific_folder)

        fResults_df.to_csv("results.csv", index=False)
