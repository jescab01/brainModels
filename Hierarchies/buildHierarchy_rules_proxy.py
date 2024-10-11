
import time
import pandas as pd
import numpy as np

data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.Data\\"

# 0. Load info and t1t2map
HCPex_info = pd.read_excel(data_dir + 'HCPex_MNI152_info.xlsx')


# 1. CONSTRUCT hierarchy based on theoretical RULES    #######
# Matrix made of "F" feedforward, "B" feedback, "L" laterals, and 1 for undefined connections
# Rules are based on the hierarchical theory of cortical processing. Felleman (1991), Markov (2013).
hierarchy = np.ones((len(HCPex_info), len(HCPex_info))).astype(object)  # 426 areas * 426 areas = 181475 connections
rules = np.ones((len(HCPex_info), len(HCPex_info))).astype(object)  # 426 areas * 426 areas = 181475 connections

early_sens = ['1 Primary Visual Cortex', '2 Early Visual Cortex', '6 Primary Somatosensory Complex (S1)', '10 Early Auditory Cortex']

# . RULING OUT the hierarchy
# General rule - [Intrahemispheric hierarchy stands for interhemispheric connections] e.g., V1 L < V2 R
tic = time.time()
for i, toROIa in enumerate(HCPex_info.Area):
    for j, fromROIa in enumerate(HCPex_info.Area):

        print(".  Ruling out connection i%i - j%i" % (i, j), end="\r")

        toROIr, toROIg, toROIt = HCPex_info.loc[HCPex_info["Area"] == toROIa].iloc[:, 2:5].values[0]
        fromROIr, fromROIg, fromROIt = HCPex_info.loc[HCPex_info["Area"] == fromROIa].iloc[:, 2:5].values[0]

        # .0 Contralateral connections. V1 L = V1 R
        if (fromROIa != toROIa) and (fromROIa[:-2] == toROIa[:-2]):
            hierarchy[i, j] = "L"
            rules[i, j] = 0

        elif fromROIa == toROIa:  # .0b Self-connections
            hierarchy[i, j] = 0
            rules[i, j] = 0

        # .1 early sensory < (higher sensory | association | motor)
        if (fromROIr in early_sens) and (toROIr not in early_sens) and (toROIt != "Subcortical") and (fromROIa[:-2] != toROIa[:-2]):
            hierarchy[i, j] = "F"
            rules[i, j] = 1
        elif (toROIr in early_sens) and (fromROIr not in early_sens) and (fromROIt != "Subcortical") and (fromROIa[:-2] != toROIa[:-2]):
             hierarchy[i, j] = "B"
             rules[i, j] = 1

        # .2 Visual
        if (toROIg == 'Visual') and (fromROIg == "Visual") and (fromROIa[:-2] != toROIa[:-2]):
            #  .2.1 V1 < Early visual
            if (fromROIr == "1 Primary Visual Cortex") and (toROIr == '2 Early Visual Cortex'):
                hierarchy[i, j] = "F"
                rules[i, j] = 2.1

            elif (toROIr == "1 Primary Visual Cortex") and (fromROIr == '2 Early Visual Cortex'):
                hierarchy[i, j] = "B"
                rules[i, j] = 2.1

            elif (toROIr == '2 Early Visual Cortex') and (fromROIr == '2 Early Visual Cortex') and (fromROIa[:-2] != toROIa[:-2]):
                # .2.2 [Early visual] V2 < V3 < V4
                if "Second_Visual_Area" in fromROIa:
                    hierarchy[i, j] = "F"
                    rules[i, j] = 2.2

                elif "Second_Visual_Area" in toROIa:
                    hierarchy[i,j] = "B"
                    rules[i, j] = 2.2

                elif "Fourth_Visual_Area" in toROIa:
                    hierarchy[i, j] = "F"
                    rules[i, j] = 2.2

                elif "Fourth_Visual_Area" in fromROIa:
                    hierarchy[i, j] = "B"
                    rules[i, j] = 2.2

        # .3 Sensoriomotor
        if (toROIg == 'Sensorimotor') and (fromROIg == "Sensorimotor") and (fromROIa[:-2] != toROIa[:-2]):
            # .3.1 sensory < (motor | POC)
            if (toROIt == 'Sensory') and (fromROIt == "Motor"):
                hierarchy[i, j] = "B"
                rules[i, j] = 3.1

            elif (fromROIt == 'Sensory') and (toROIt == "Motor"):
                hierarchy[i, j] = "F"
                rules[i, j] = 3.1

            elif (fromROIt == 'Sensory') and ('9 Posterior Opercular Cortex' not in fromROIr) and (toROIr in '9 Posterior Opercular Cortex'):
                hierarchy[i, j] = "F"
                rules[i, j] = 3.1

            elif (toROIt == 'Sensory') and ('9 Posterior Opercular Cortex' not in toROIr) and (fromROIr in '9 Posterior Opercular Cortex'):
                hierarchy[i, j] = "B"
                rules[i, j] = 3.1


            # .3.2 (3a|3b) < 1 < 2
            elif (toROIr == '6 Primary Somatosensory Complex (S1)') and (fromROIr == '6 Primary Somatosensory Complex (S1)') and (fromROIa[:-2] != toROIa[:-2]):
                if ('Primary_Sensory_Cortex' in fromROIa or 'Area_3a' in fromROIa) and ('Primary_Sensory_Cortex' not in fromROIa or 'Area_3a' not in fromROIa):
                    hierarchy[i, j] = "B"
                    rules[i, j] = 3.2
                elif ('Primary_Sensory_Cortex' in toROIa or 'Area_3a' in toROIa) and ('Primary_Sensory_Cortex' not in toROIa or 'Area_3a' not in toROIa):
                    hierarchy[i, j] = "F"
                    rules[i, j] = 3.2
                elif ("Area_1" in fromROIa) and ("Area_2" in toROIa):
                    hierarchy[i, j] = "F"
                    rules[i, j] = 3.2
                elif ("Area_1" in toROIa) and ("Area_2" in fromROIa):
                    hierarchy[i, j] = "B"
                    rules[i, j] = 3.2

            # .3.3 area 5 < (motor | POC)
            elif (fromROIr == '7 Area 5') and ((toROIt == 'Motor') or (toROIr == '9 Posterior Opercular Cortex')):
                hierarchy[i, j] = "F"
                rules[i, j] = 3.3

            elif (toROIr == '7 Area 5') and ((fromROIt == 'Motor') or (fromROIr == '9 Posterior Opercular Cortex')):
                hierarchy[i, j] = "B"
                rules[i, j] = 3.3

            # .3.4 (premotor | SM) < M1
            if ((fromROIt == 'Motor') and (fromROIr != '6 Primary Motor Cortex (M1)')) and (toROIr == '6 Primary Motor Cortex (M1)'):
                hierarchy[i, j] = "F"
                rules[i, j] = 3.4

            elif ((toROIt == 'Motor') and (toROIr != '6 Primary Motor Cortex (M1)')) and (fromROIr == '6 Primary Motor Cortex (M1)'):
                hierarchy[i, j] = "B"
                rules[i, j] = 3.4

        # .4 Auditory
        if (toROIr == '10 Early Auditory Cortex') and (fromROIr == "10 Early Auditory Cortex") and (fromROIa[:-2] != toROIa[:-2]):

            # .4.1 A1 < early auditory
            if ('Primary_Auditory_Cortex' in fromROIa) and ('Primary_Auditory_Cortex' not in toROIa):
                hierarchy[i, j] = "F"
                rules[i, j] = 4.1

            elif ('Primary_Auditory_Cortex' in toROIa) and ('Primary_Auditory_Cortex' not in fromROIa):
                hierarchy[i, j] = "B"
                rules[i, j] = 4.1


print(" - elapsed time %0.2fs" % (time.time()-tic))






# Assign hierarchical connections to the rest based on the t1t2 proxy
# t1/t2 ratio is negatively correlated with anatomical hierarchy
# Then, conn from lower to higher in hierarchy = "F" (FeedForward) and vice-versa for "B" (FeedBack).
# when equal, we consider the roi is the same and connection is "L"

hier_proxy = hierarchy.copy()
for i, toROI_t1t2 in enumerate(HCPex_info["t1w/t2w"]):
    for j, fromROI_t1t2 in enumerate(HCPex_info["t1w/t2w"]):
        if hier_proxy[i, j] == 1:
            hier_proxy[i, j] = "F" if fromROI_t1t2 > toROI_t1t2 else "B"


np.savetxt(data_dir + 'HCPex_hierarchy_proxy_v2.txt', hier_proxy, fmt='%s')