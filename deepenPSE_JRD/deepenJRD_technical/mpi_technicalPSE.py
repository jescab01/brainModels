import pandas as pd
from mpi4py import MPI
import numpy as np
from corticon_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python dynSysPSE_mpi_speed.py
"""

name = "technicalPSE"


# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Define param combinations
# Common simulation requirements
# subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
# subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")

subjects = ["NEMOS_035"]

models = ["jrd_pTh", "jrd", "jr"]

n_rep = 1

coupling_vals = np.arange(0, 120, 1)
# speed_vals = np.arange(0.5, 25, 1)
# EPSP_vals = np.arange(2.6, 9.75, 0.5)
# IPSP_vals = np.arange(17.6, 110, 2)

# p_vals = np.arange(0, 0.5, 0.005)

sigmaTh_vals = np.arange(0, 0.05, 0.001)


params = [[subj, mode, r, g, 12, 0.22, 0.022] for subj in subjects for mode in modes for r in range(n_rep) for g in coupling_vals] +\
         [[subj, mode, r, 60, 12, 0.22, sigma] for subj in subjects for mode in modes for r in range(n_rep) for sigma in sigmaTh_vals]
        # [[subj, mode, r, 60, s, 0.22, 0.022] for subj in subjects for mode in modes for r in range(n_rep) for s in speed_vals] +\
        # [[subj, mode, r, 60, 12, p, 0.022] for subj in subjects for mode in modes for r in range(n_rep) for p in p_vals] + \

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

local_results = simulate_paramSpace(local_params)  # run the function for each parameter set and rank


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

    # fResults_df = pd.DataFrame(final_results, columns=["subject", "mode", "rep", "g", "speed", "p", "sigma", "rPLV", "max_mV", "min_mV", "fft", "freqs"])

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

        import pickle
        with open("results.pkl", "wb") as f:
            pickle.dump(final_results, f)


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

        import pickle
        with open("results.pkl", "wb") as f:
            pickle.dump(final_results, f)
