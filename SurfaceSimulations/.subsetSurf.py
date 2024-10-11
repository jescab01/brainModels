
"""
In this script, I show how to subset previous surface+regional data.

"""

import os

import numpy as np
import scipy
from tvb.simulator.lab import *
from zipfile import ZipFile



main_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\3Energy\\"
data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\\.DataTemp\\"

surfpack = "HCPex-r426-surfdisc17k_pack\\"
lc_title = "local_connectivity-amp1.0sig3.67cut5.51.mat"



## 1. Load the surface to subset and configure
local = data_dir + surfpack + "local_conn\\" + lc_title
cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                                         region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                                         local_connectivity_file=local)

conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")
cx.region_mapping_data.connectivity = conn

cx.configure()






### 2. Subset the mesh

# 2.1 Select cortical surface regions :: a) [1, 2, 3] =  V1,2,3 (left)
sel = [180, 181]

# 2.2 Identify vertex of selected regions
ids_vsel = [list(np.argwhere(cx.region_mapping_data.array_data == roi)[:, 0]) for roi in sel]
ids_vsel = [item for sublist in ids_vsel for item in sublist]


# 2.3 Subset identified vertices for each data
rm_sub = cx.region_mapping_data.array_data[ids_vsel]
vertex_sub = cx.surface.vertices[ids_vsel]
normals_sub = cx.surface.vertex_normals[ids_vsel]

# 2.4 Subset triangles whose vertices are in ids_vsel
tri_sub = np.array([t for t in cx.surface.triangles if (t[0] in ids_vsel) and (t[1] in ids_vsel) and (t[2] in ids_vsel)])

# 2.5 Substitute old triangles ids with the new ones
tri_sub_trans = [[ids_vsel.index(t[0]), ids_vsel.index(t[1]), ids_vsel.index(t[2])] for t in tri_sub]

# 2.6 Subset the local connectivity
localCoup = cx.local_connectivity.matrix[ids_vsel][:, ids_vsel]

# 2.7 Substitute old region mapping with new ids
for i, id in enumerate(sel):
    rm_sub[rm_sub == id] = i


# 3. Subset connectivity data
subcx = []
rois = sorted(sel + subcx)

weights = conn.weights[rois][:, rois]
tracts = conn.tract_lengths[rois][:, rois]
cortex = conn.cortical[rois]

centres_txt = ["%s %0.5f %0.5f %0.5f None\n" %
               (conn.region_labels[roi], conn.centres[roi, 0], conn.centres[roi, 1], conn.centres[roi, 2])
               for roi in rois]





## 4. Save result
outpack = "subHCPex-r%i-surf%i_pack\\" % (len(rois), len(ids_vsel)) if len(ids_vsel) < 1000 else \
    "subHCPex-r%i-surf%ik_pack\\" % (len(rois), len(ids_vsel)//1000)

# 4.1 Create output folder
if not os.path.isdir(data_dir + outpack):
    os.mkdir(data_dir + outpack)

# 4.2 Save surface data
np.savetxt(data_dir + outpack + "region_mapping.txt", rm_sub, delimiter=" ")


if not os.path.isdir(data_dir + outpack + "local_conn\\"):
    os.mkdir(data_dir + outpack + "local_conn\\")
scipy.io.savemat(data_dir + outpack + 'local_conn\\' + lc_title, {"LocalCoupling": localCoup})

np.savetxt(data_dir + outpack + "vertices.txt", vertex_sub, delimiter=" ")
np.savetxt(data_dir + outpack + "vertex_normals.txt", normals_sub, delimiter=" ")
np.savetxt(data_dir + outpack + "triangles.txt", tri_sub_trans, delimiter=" ")

# Create a ZipFile Object
with ZipFile(data_dir + outpack + 'cortical_surface.zip', 'w') as zip_object:
   # Adding files that need to be zipped
   zip_object.write(data_dir + outpack + "vertices.txt")
   zip_object.write(data_dir + outpack + "vertex_normals.txt")
   zip_object.write(data_dir + outpack + "triangles.txt")


# 4.3 Save structural connectivity data
np.savetxt(data_dir + outpack + "weights.txt", weights, delimiter=" ")
np.savetxt(data_dir + outpack + "tract_lengths.txt", tracts, delimiter=" ")
np.savetxt(data_dir + outpack + "cortical.txt", cortex, delimiter=" ", fmt="%i")

with open(data_dir + outpack + "centres.txt", 'w') as file:
    for row in centres_txt:
        file.write(row)
file.close()


# Create a ZipFile Object
with ZipFile(data_dir + outpack + 'connectivity.zip', 'w') as zip_object:
   # Adding files that need to be zipped
   zip_object.write(data_dir + outpack + "weights.txt")
   zip_object.write(data_dir + outpack + "tract_lengths.txt")
   zip_object.write(data_dir + outpack + "cortical.txt")
   zip_object.write(data_dir + outpack + "centres.txt")



# Create a readme file and note the features of the subseted modelling data
description = ("Subset of the %s to %i regions + %i subcortical: %s + %s \n"
               "Cortex - %i vertices and %i faces."
               % (surfpack, len(sel), len(subcx), str(sel), str(subcx), len(vertex_sub), len(tri_sub)))

with open(data_dir + outpack + 'readme.txt', 'w') as f:
    f.write(description)



## Delete aux files created
remove_files = ["centres.txt", "cortical.txt", "tract_lengths.txt", "triangles.txt", "vertex_normals.txt", "vertices.txt", "weights.txt"]

for file in remove_files:
    if os.path.isfile(data_dir + outpack + file):
        os.remove(data_dir + outpack + file)



