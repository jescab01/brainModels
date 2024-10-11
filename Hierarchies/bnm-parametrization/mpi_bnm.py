import pandas as pd
from mpi4py import MPI
import numpy as np
from parallel_bnm import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 2 python mpi_bnm.py
"""

name = "bnm-params"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# g_vals = np.arange(0, 100, 1)

# ratio_vals = [(0.5, 0.5)]
# ratio_range = np.arange(0.30, 0.71, 0.1)
# ratio_vals = [(round(val, 2), round(1 - val, 2)) for val in ratio_range]

# sigma_vals = [0]  #, 2.2e-1, 2.2e-2, 2.2e-3, 2.2e-4]


# He_vals = np.arange(1.5, 6, 0.25)  # def 3.25
# Hi_vals = np.arange(18, 40, 1)  # def 22
taue_vals = np.arange(2, 14, 0.25)  # def 10
taui_vals = np.arange(4, 25, 0.5)  # def 20
# speed_vals = np.arange(0.5, 7, 0.25)

modes = ["hierarchical", "classical"]

# #              mode, g, ff, fb, l, sigma, He, Hi, taue, taui, speed

params = [(mode, 50, 0.5, 0.5, 0, 0, 3.25, 22, taue, taui, 3.9)
          for mode in modes for taue in taue_vals for taui in taui_vals]



# #              mode, g, ff, fb, l, sigma, He, Hi, taue, taui, speed
# params_He = [("hierarchical", g, ff, fb, 0, 0, He, 22, 10, 20, 3.9)
#                for g in g_vals for (ff, fb) in ratio_vals for He in He_vals]
#
# params_Hi = [("hierarchical", g, ff, fb, 0, 0, 3.25, Hi, 10, 20, 3.9)
#                for g in g_vals for (ff, fb) in ratio_vals for Hi in Hi_vals]
#
# params_taue = [("hierarchical", g, ff, fb, 0, 0, 3.25, 22, taue, 20, 3.9)
#                for g in g_vals for (ff, fb) in ratio_vals for taue in taue_vals]
#
# params_taui = [("hierarchical", g, ff, fb, 0, 0, 3.25, 22, 10, taui, 3.9)
#                for g in g_vals for (ff, fb) in ratio_vals for taui in taui_vals]
#
# params_speed = [("hierarchical", g, ff, fb, 0, 0, 3.25, 22, 10, 20, speed)
#                for g in g_vals for (ff, fb) in ratio_vals for speed in speed_vals]


#              mode, g, ff, fb, l, sigma, He, Hi, taue, taui, speed
# params_He = [("classical", g, 0, 0, 0, 0, He, 22, 10, 20, 3.9)
#                for g in g_vals for He in He_vals]
#
# params_Hi = [("classical", g, 0, 0, 0, 0, 3.25, Hi, 10, 20, 3.9)
#                for g in g_vals for Hi in Hi_vals]

# params_taue = [("classical", g, 0, 0, 0, 0, 3.25, 22, taue, 20, 3.9)
#                for g in g_vals for taue in taue_vals]
#
# params_taui = [("classical", g, 0, 0, 0, 0, 3.25, 22, 10, taui, 3.9)
#                for g in g_vals for taui in taui_vals]
#
# params_speed = [("classical", g, 0, 0, 0, 0, 3.25, 22, 10, 20, speed)
#                for g in g_vals for speed in speed_vals]
#
#
# params = params_He + params_Hi + params_taue + params_taui + params_speed

# modes = ["classical", "classical-reparam", "classical-reparam-cs"]
# params_clas = [(mode, g, 1, 0, 0, sigma) for mode in modes for g in g_vals for sigma in sigma_vals]

# params = params_hier + params_clas


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

local_results = parallel_bnm(local_params)  # run the function for each parameter set and rank

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
                               columns=["mode", "g", "FF", "FB", "L", "sigma", "He", "Hi", "taue", "taui", "speed",
                                        "roi_id", "roi",
                                        "freq_peaks", "band_modules",
                                        "Savg", "Smin", "Smax", "intraS_std", "interS_std"])

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
