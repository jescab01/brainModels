import time
import numpy as np
import pandas as pd
from mne import filter
import scipy

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

from toolbox import timeseriesPlot, FFTplot, PLV, epochingTool, plotConversions, AEC, FFTpeaks, paramSpace

# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)
subjectid = ".2003JansenRitDavid_N2"
wd = os.getcwd()
main_folder = wd+"\\"+subjectid
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

# if subjectid not in os.listdir(ctb_folder):
#     os.mkdir(main_folder)

emp_subj = "NEMOS_035"

samplingFreq = 1000  #Hz
simLength = 13000  # ms - relatively long simulation to be able to check for power distribution
transient = 1000  # seconds to exclude from timeseries due to initial transient

m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),   # SLOW population
                          tau_e1=np.array([10.0]), tau_i1=np.array([20.0]),
                          He2=np.array([3.25]), Hi2=np.array([22]),   # FAST population
                          tau_e2=np.array([10.0]), tau_i2=np.array([20.0]),
                          w=np.array([0.8]), c=np.array([135.0]),

                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                          v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                          p=np.array([0.22]), sigma=np.array([0.011]))

tau_e1, tau_i1 = 10.8, 22.0
tau_e2, tau_i2 = 4.6, 2.9

## Remember to hold tau*H constant.
m.tau_e1, m.tau_i1 = np.array([tau_e1]), np.array([tau_i1])
m.tau_e2, m.tau_i2 = np.array([tau_e2]), np.array([tau_i2])
m.He1, m.Hi1 = np.array([32.5 / tau_e1]), np.array([440 / tau_i1])
m.He2, m.Hi2 = np.array([32.5 / tau_e2]), np.array([440 / tau_i2])


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
#integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.EulerDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder+emp_subj+"_AAL3_wpTh-wpCer.zip")
conn.weights = conn.scaled_weights(mode="tract")

fc_rois_cortex = list(range(0, 40)) + list(range(46, 74)) + list(range(78, 90))
sc_rois_cortex = list(range(0, 34)) + [146, 147] + list(range(34, 38)) + list(range(44, 72)) + list(range(78, 90))
fc_rois_dmn = [2, 3, 34, 35, 38, 39, 64, 65, 70, 71, 84, 85]
sc_rois_dmn = [2, 3] + [146, 147] + [36, 37, 62, 63, 68, 69, 84, 85]


results = []
results_fft_peaks = []
for k in np.arange(0, 50, 0.5):

    tic0 = time.time()
    print("Simulating for k=%i " % k)

    # Coupling function
    coup = coupling.SigmoidalJansenRitDavid(a=np.array([34]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
    conn.speed = np.array([12])

    mon = (monitors.Raw(),)

    m.sigma = m.sigma/k
    m.p = m.p/k

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
    sim.configure()

    output = sim.run(simulation_length=simLength)
    print("Simulation time: %0.2f sec" % (time.time() - tic0,))
    # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
    # Mount PSP output as: w * (vExc1 - vInh1) + (1-w) * (vExc2 - vInh2)
    raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
               (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels

    data = np.asarray([np.average(raw_data, axis=0)])
    data = np.concatenate((data, raw_data), axis=0)

    # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
    results_fft_peaks.append((k, FFTpeaks(data, simLength-transient)[0][0],
                         FFTpeaks(data, simLength-transient)[1][0]))

    ## HTML
    # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, regionLabels, main_folder, mode="html", title="timeseries-w="+str(m.w))
    # # Fourier Analysis plot
    # FFTplot(raw_data, simLength - transient, regionLabels, main_folder, mode="html", title= "w="+str(m.w), type="linear", epochs=True)

    newRow = [k]
    bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data[sc_rois_cortex, :], samplingFreq, lowcut, highcut)

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
        # plotConversions(raw_data[:, :len(efSignals[0][0])]-np.average(raw_data[:, :len(efSignals[0][0])]), efSignals[0], efPhase[0], efEnvelope[0], band=bands[0][b], regionLabels=regionLabels, n_signals=1, raw_time=raw_time)

        #CONNECTIVITY MEASURES
        ## PLV
        plv = PLV(efPhase)
        # aec = AEC(efEnvelope)

        # Load empirical data to make simple comparisons
        plv_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "\\" + bands[0][b] + "_plv.txt")[:, fc_rois_cortex][fc_rois_cortex]
        # aec_emp = np.loadtxt(wd+"\\CTB_data\\output\\FC_"+emp_subj+"\\"+bands[0][b]+"corramp.txt")[:, fc_rois_cortex][fc_rois_cortex]

        # Comparisons
        t1 = np.zeros(shape=(2, len(conn.region_labels[sc_rois_cortex]) ** 2 // 2 - len(conn.region_labels[sc_rois_cortex]) // 2))
        t1[0, :] = plv[np.triu_indices(len(conn.region_labels[sc_rois_cortex]), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(conn.region_labels[sc_rois_cortex]), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

        # t2 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
        # t2[0, :] = aec[np.triu_indices(len(conn.region_labels), 1)]
        # t2[1, :] = aec_emp[np.triu_indices(len(conn.region_labels), 1)]
        # aec_r = np.corrcoef(t2)[0, 1]

        newRow.append(plv_r)
    results.append(newRow)
    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic0,))

df_fc = pd.DataFrame(results, columns=["k", "Delta", "Theta", "Alpha", "Beta", "Gamma"])
df_fft = pd.DataFrame(results_fft_peaks, columns=["g", "s", "peak", "module"])
