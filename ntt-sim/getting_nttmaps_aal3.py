
import os
import pickle
import neuromaps
import numpy as np
from neuromaps.parcellate import Parcellater
import plotly.graph_objects as go
from nilearn import image

"""
Introductory code
"""

# Tag - description of dataset content
neuromaps.datasets.available_tags()

# Available maps - (paper, desc, space, res)
neuromaps.datasets.available_annotations(tags=["receptors"], space="MNI152")
len(neuromaps.datasets.available_annotations(tags=["receptors"], space="MNI152"))


"""
PROCESSING RECEPTOR MAPS
the main magic is done by nilearn toolbox (NiftiLabelsMasker)
https://nilearn.github.io/stable/auto_examples/06_manipulating_images/plot_nifti_labels_simple.html
"""

folder = os.getcwd() + "/neuromaps-data/"

# 0) Prepare parcellation
aal3 = Parcellater(parcellation=folder + "AAL3/ROI_MNI_V7_1mm.nii", space="mni152")

# 1) Gather information
maps_parcelled = []

for i, map in enumerate(neuromaps.datasets.available_annotations(tags=["receptors"], space="MNI152")):

    [source, desc, space, res] = map

    try:
        # 1.1) Download a map -
        map = neuromaps.datasets.fetch_annotation(data_dir=folder, source=source, desc=desc, space=space, res=res)

        # 1.2) Apply parcellation scheme (aal3)
        maps_parcelled.append([source, desc, space, res, aal3.fit_transform(map, "mni152")])

    except:
        pass



# 2) Check resulting parcellations; The result is not perfect but works well.
# (Computing some masked averages and checking it out by eye on MRIcron)

# aal3 = image.get_data(folder + "AAL3/AAL3v1_1mm.nii.gz")
# map = neuromaps.datasets.fetch_annotation(data_dir=folder, source=source, desc=desc, space=space, res=res)
# map, aal3 = neuromaps.resample_images(map, folder + "AAL3/AAL3v1_1mm.nii.gz",
#                              "mni152", "mni152", resampling='transform_to_trg',
#                              method="linear")
#
# roi_mask = image.get_data(aal3)==32  # The number represents a ROI
# roi_average = np.average(image.get_data(map)[roi_mask])


# 3) Save maps
with open(folder + "/ntt_maps.pkl", "wb") as f:
    pickle.dump(maps_parcelled, f)
    f.close()