
import os
import time

import pandas as pd
from mpi4py import MPI
import numpy as np
from dynSysPSE_parallel import simulate_dynSysPSE
# from CORTICON.dynSysPSE_parallel import simulate_dynSysPSE
import pickle

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python dynSysPSE_mpi_speed.py
"""

spaces = "singles_jrd_sigmas"  # "singles", "coupled", "c_sigma"
name = "dynSys_" + spaces

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


## Define param combinations
# Common simulation requirements
# subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
# subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")
subjects = ["NEMOS_AVG"]

models = ["jrd_custom"]
struct = "AAL2"  # AAL2; AAL2pTh
subject = "NEMOS_AVG"
n_rep = 3

if "pTh" in struct:
    rois = [140, 2]  # Thal_PuM_L, Frontal_Sup_2_L  ::  weight = 0.03 in AAL2pTh
else:
    rois = [80, 93]  # Thalamus_L, Temporal_Inf_R  ::  weight = 0.012 in AAL2


coupling_vals = np.arange(0, 150, 2)
speed_vals = np.arange(0.5, 20, 2)

# w_vals = np.arange(0, 1, 0.05)
p_vals = np.arange(0, 0.5, 0.01)
# sigma_vals = np.arange(0, 0.05, 0.005)
sigma_vals = [0, 0.01, 0.02, 0.04]

# EPSP_vals = np.arange(2.6, 9.75, 0.5)
# IPSP_vals = np.arange(17.6, 110, 2)
# tau_e_vals =
# tau_i_vals =

if spaces == "singles":
    params = [["p_single", model, struct, subject, rois, r, 0, 15, 0.8, p, 0.022] for model in models for r in range(n_rep) for p in p_vals] + \
             [["sigma_single", model, struct, subject, rois, r, 0, 15, 0.8, 0.22, sigma] for model in models for r in range(n_rep) for sigma in sigma_vals] + \
             [["w_single", model, struct, subject, rois, r, 0, 15, w, 0.22, 0.022] for model in models for r in range(n_rep) for w in w_vals] + \
             [["p-sigma_single", model, struct, subject, rois, r, 0, 15, 0.8, p, sigma] for model in models for r in range(n_rep) for p in p_vals for sigma in sigma_vals] #+ \

elif spaces == "coupled":
    params = [["g_coupled", model, struct, subject, rois, r, g, 15, 0.8, 0.22, 0.022] for model in models for r in range(n_rep) for g in coupling_vals] + \
             [["p_coupled", model, struct, subject, rois, r, 80, 15, 0.8, p, 0.022] for model in models for r in range(n_rep) for p in p_vals] + \
             [["sigma_coupled", model, struct, subject, rois, r, 80, 15, 0.8, 0.22, sigma] for model in models for r in range(n_rep) for sigma in sigma_vals] + \
             [["g-p_coupled", model, struct, subject, rois, r, g, 15, 0.8, p, 0.022] for model in models for r in range(n_rep) for g in coupling_vals for p in p_vals]

elif spaces == "c_sigma":
    params = [["g-sigma_coupled", model, struct, subject, rois, r, g, 15, 0.8, 0.22, sigma] for model in models for r in range(n_rep) for g in coupling_vals for sigma in sigma_vals] + \
             [["p-sigma_coupled", model, struct, subject, rois, r, 80, 15, 0.8, p, sigma] for model in models for r in range(n_rep) for p in p_vals for sigma in sigma_vals]

elif spaces == "bigNet_plv":
    params = [["g-p-sigma_bigNet_plv", model, struct, subject, rois, r, g, 15, 0.8, p, sigma]
              for model in models for r in range(n_rep) for g in coupling_vals for p in p_vals for sigma in sigma_vals]

elif spaces == "bigNet_plv_speed":
    params = [["g-p-s_bigNet_plv", model, struct, subject, rois, r, g, s, 0.8, p, 0.022]
              for model in models for r in range(n_rep) for g in coupling_vals for p in p_vals for s in speed_vals]

elif spaces == "bigNet_plv_wp":
    params = [["g-p-s_bigNet_plv", model, struct, subject, rois, r, g, s, 0.8, 0, 0.022]
              for model in models for subject in subjects for r in range(n_rep) for g in coupling_vals for s in speed_vals]

elif spaces == "bigNet_plv_wp2":
    params = [["g-speed_bigNet_plv", model, struct, subject, rois, r, g, s, 0.8, 0, 0.022]
              for model in models for subject in subjects for r in range(n_rep) for g in coupling_vals for s in speed_vals]

elif spaces == "singles_jrd_sigmas":
    params = [["p-sigmadiscrete_single", model, struct, subject, rois, r, 0, 15, 0.8, p, sigma]
              for model in models for r in range(n_rep) for p in p_vals for sigma in sigma_vals]


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

local_results = simulate_dynSysPSE(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    # Segment pickle file
    df_results = pd.DataFrame(final_results)

    if len(df_results.columns) == 14:
        df_results.columns = ["space", "model", "struct", "subject", "rois", "rep", "g", "speed", "w", "p", "sigma", "fft", "simtime", "rPLV"]
    else:
        df_results.columns = ["space", "model", "struct", "subject", "rois", "rep", "g", "speed", "w", "p", "sigma", "psp_t", "psp_dt", "fft", "simtime"]

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

        for space in set(df_results["space"]):
            df_temp = df_results.loc[df_results["space"]==space]

            with open(specific_folder + "\\" + space + "_results.pkl", "wb") as f:
                pickle.dump(df_temp, f)


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

        for space in set(df_results["space"]):
            df_temp = df_results.loc[df_results["space"] == space]

            with open(space + "_results.pkl", "wb") as f:
                pickle.dump(df_temp, f)



