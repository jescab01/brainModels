
import time
import numpy as np
import scipy.signal
import scipy.stats
import pickle
import matplotlib.pyplot as plt

from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995_Ntt5
import datetime


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    from toolbox import multitapper, PLV, epochingTool
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"


## Define working points per subject


# Prepare simulation parameters
simLength = 5 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 1000  # ms

emp_subj, model, g, alpha = "NEMOS_035", "jr", 3, 10000
p_ras, sigma_ras, p_cx, sigma_cx = 0.11, 0.022, 0.09, 2.2e-8

tic = time.time()


# STRUCTURAL CONNECTIVITY      #########################################
n2i_indexes = []  # not to include indexes
# Thalamus structure

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL3_pass.zip")
conn.weights = conn.scaled_weights(mode="tract")

# indexes = [0, 1, 2, 3, 4]
# conn.weights = conn.weights[:, indexes][indexes]
# conn.tract_lengths = conn.tract_lengths[:, indexes][indexes]
# conn.region_labels = conn.region_labels[indexes]


# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                 'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                 'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                 'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Calcarine_L',
                 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                 'Temporal_Inf_R']
ras_rois = ["Raphe_D", "Raphe_M", 'LC_L', 'LC_R', 'VTA_L', 'VTA_R',
            'SN_pc_L', 'SN_pc_R', 'SN_pr_L', 'SN_pr_R', 'Thal_IL_L', 'Thal_IL_R']

# load text with FC rois; check if match SC
# FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
# FC_cortex_idx = [FClabs.index(roi) for roi in
#                  cortical_rois]  # find indexes in FClabs that matches cortical_rois
SClabs = list(conn.region_labels)
# SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]



# NEUROTRANSMISSION variables
ntts = ["5HT", "NE", "D", "ACh", "Glu"]

ntt_sources = [["Raphe_D", "Raphe_M"], ['LC_L', 'LC_R'],
               ['VTA_L', 'VTA_R', 'SN_pc_L', 'SN_pc_R', 'SN_pr_L', 'SN_pr_R'],
               ['Thal_IL_L', 'Thal_IL_R'], ["all"]]

ntt_sourcemasks = [[True if roi in ntt_srcs or "all" in ntt_srcs else False for roi in SClabs] for ntt_srcs in ntt_sources]

with open("E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\\neuromaps-data\\ntt_maps.pkl", "rb") as f:
    receptor_maps_array = pickle.load(f)
    f.close()
receptor_maps_array = np.asarray(receptor_maps_array, dtype=object)

# ntt_receptorstudies = [["radnakrishnan2018", "savli", "believau", "gallezot"],
#                        ["ding2010"],
#                        ["kaller2017", "smith2017", "sandiego2015"],  ## kaller out: 164/166 regions
#                        ["hilmer2016", "naganawa2020", "tuominen", "aghourian2017", "bedard2019"], ## tuomien out: 165/166 regions
#                        ["smart2019", "dubois2015"]] #receptor maps
# Selecting receptor map studies to use
receptor_studies = [["radnakrishnan2018"], ["ding2010"], ["smith2017", "sandiego2015"],
                    ["aghourian2017", "bedard2019"], ["smart2019", "dubois2015"]]  # Receptor maps

receptor_maps_average, receptor_maps_std = [], []
for rs_list in receptor_studies:
    print(rs_list)
    temp_maps = []

    for study in rs_list:
        temp_array = receptor_maps_array[receptor_maps_array[:, 0] == study, 4][0].T  #
        temp_array = (temp_array - np.min(temp_array))/(np.max(temp_array) - np.min(temp_array))  # Normalize
        temp_maps.append(temp_array)

    receptor_maps_average.append(np.average(np.asarray(temp_maps), axis=0))
    receptor_maps_std.append(np.std(np.asarray(temp_maps), axis=0))

receptor_maps_average = np.hstack(receptor_maps_average)

# NEURAL MASS MODEL: Jansen-Rit adapted for neurotransmission    ###########################################
sigma_array = np.asarray([sigma_ras if roi in ras_rois else sigma_cx for roi in conn.region_labels])
p_array = np.asarray([p_ras if roi in ras_rois else p_cx for roi in conn.region_labels])

m = JansenRit1995_Ntt5(
    He=np.array([3.25]), Hi=np.array([22]), tau_e=np.array([10]), tau_i=np.array([20]), c=np.array([1]),
    c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]), c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
    p=np.array([p_array]), sigma=np.array([sigma_array]), e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]),
    # Neurotransmission params: eta - ntt decay rates; tau_m - time constant for ntt
    eta=np.array([0.5]), tau_m=np.array([120]), receptormaps=receptor_maps_average)

coup = coupling.SigmoidalJansenRit_Ntt(
    a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]),
    alpha=np.array([alpha]), ntt_masks=np.array(ntt_sourcemasks))

conn.speed = np.array([3.9])

# OTHER PARAMETERS   ###
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

# mon = (monitors.Raw(), monitors.EEG(projection=pr, sensors=ss, region_mapping=rm_post))
mon = (monitors.Raw(),)

print("Simulating %s (%is)  ||  PARAMS: g%i alpha%i" % (model, simLength / 1000, g, alpha))


# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)

# Extract gexplore_data: "output[a][b][:,0,:,0].T" where:
# a=monitorIndex, b=(gexplore_data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.

raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels


## Plot
# state variables: "vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh", "S_5HT", "M_5HT"
timeseries_spectra(output[0][1][transient:, 0, :, 0].T, simLength, transient, regionLabels, mode="html",
                   folder="figures", freqRange=[1, 35], opacity=0.8, title="sigma10", auto_open=True)


