import time

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import filter
from toolbox import FFTpeaks,  PLV, PLE, epochingTool, paramSpace

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Common simulation requirements
subjectid = ".1995JansenRit"
emp_subj = "AVG_NEMOS"

# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 1000  # ms

# Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

ctb_folder = "D:\\Users\\Jesus CabreraAlvarez\\PycharmProjects\\brainModels\\CTB_data\\output\\"
conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
conn.weights = conn.scaled_weights(mode="tract")

mon = (monitors.Raw(),)

# Folder structure
wd=os.getcwd()
main_folder = wd+"\\"+"PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

specific_folder = main_folder+"\\PSEparallel"+subjectid+"-"+emp_subj+"-"+time.strftime("m%md%dy%Y")
if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)

# Set up parallelization
num_cores=multiprocessing.cpu_count()-2

coupling_vals = np.arange(0, 120, 1)
speed_vals = np.arange(0.5, 25, 1)
n_rep = 1

inputs = tqdm([(g, s, r) for g in coupling_vals for s in speed_vals for r in range(n_rep)])


def simulate_parallel(inputs_):

    (g, s, r) = inputs_

    #g,s,r=15,15,1

    results_fft_peak = list()
    results_fc = list()
    results_ple = list()

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([s])

    tic = time.time()

    print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]

    # average signals to obtain mean signal frequency peak
    data = np.asarray([np.average(raw_data, axis=0)])
    data = np.concatenate((data, raw_data), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]

    # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
    results_fft_peak.append((g, s, r, FFTpeaks(data, simLength-transient)[0][0][0],
                             FFTpeaks(data, simLength-transient)[1][0][0]))

    newRow_fc = [g, s, r]
    newRow_ple = [g, s, r]
    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase=list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(efPhase[0], np.arange(len(efPhase[0][0])), regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV
        plv = PLV(efPhase)

        ## PLE - Phase Lag Entropy
        ## PLE parameters - Phase Lag Entropy
        tau_ = 25  # ms
        m_ = 3  # pattern size
        ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=85)
        newRow_ple.append(np.average(ple[np.triu_indices(len(ple[0]), 1)]))


        # Load empirical data to make simple comparisons
        plv_emp=np.loadtxt(ctb_folder+"FC_"+emp_subj+"\\"+bands[0][b]+"_plv.txt")

        # Comparisons
        t1 = np.zeros(shape=(2, len(conn.region_labels)**2//2-len(conn.region_labels)//2))
        t1[0, :] = plv[np.triu_indices(len(conn.region_labels), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels), 1)]
        plv_r = np.corrcoef(t1)[0, 1]
        newRow_fc.append(plv_r)

    simname = subjectid+"-"+emp_subj+"-t"+str(simLength)+'_'+str(g)+"_"+str(s)+"_"+str(r)+"-"+time.strftime("t%Hh.%Mm.%Ss")

    df_fft=pd.DataFrame(results_fft_peak, columns=["G", "speed", "round", "mS_peak", "mS_module"])
    df_fft.to_csv(specific_folder+"/PSE_FFTpeaks"+simname+".csv", index=False)
    results_fc.append(newRow_fc)
    df_fc=pd.DataFrame(results_fc, columns=["G", "speed", "round", "Alpha"])
    df_fc.to_csv(specific_folder+"/PSE_FC"+simname+".csv", index=False)
    results_ple.append(newRow_ple)
    df_ple=pd.DataFrame(results_ple, columns=["G", "speed", "round", "Alpha"])
    df_ple.to_csv(specific_folder+"/PSE_PLE"+simname+".csv", index=False)

    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))


if __name__ == "__main__":

    processed_list = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(simulate_parallel)(i) for i in inputs)

