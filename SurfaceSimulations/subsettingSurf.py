
"""
In this script, I show how to subset previous surface+regional data.

"""

import numpy as np
import scipy
from tvb.simulator.lab import *
import matplotlib.pylab as plt

import plotly.express as px
import plotly.graph_objects as go


# Locate the reference data

main_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\\3Energy\\"
data_dir = "E:\LCCN_Local\PycharmProjects\BrainRhythms\\.DataTemp\\"

surfpack = "HCPex-r426-surfdisc17k_pack\\"
lc_title = "local_connectivity-amp1.0sig3.67cut5.51.mat"


## Simulation parameters
simLength = 500  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  #Hz
transient = 0  # ms to exclude from timeseries due to initial transient

oscillator = models.JansenRit()

conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")

coup = coupling.SigmoidalJansenRit(a=np.array([4]))

#Initialise an Integrator
heunint = integrators.HeunDeterministic(dt=1000/samplingFreq)

# Monitors
mons = (monitors.Raw(), )


#Initialise a surface


# local = None

cx = cortex.Cortex.from_file(source_file=main_dir + surfpack + "cortical_surface.zip",
                                         region_mapping_file=main_dir + surfpack + "region_mapping.txt",
                                         local_connectivity_file=local)



sim = simulator.Simulator(model=oscillator, connectivity=conn, coupling=coup, surface=cx,
                          integrator=heunint, monitors=mons)
sim.configure()

if not local:
    localCoup = cx.local_connectivity.matrix
    scipy.io.savemat(main_dir + surfpack + 'local_connectivity.mat', {"LocalCoupling": localCoup})
    plt.spy(localCoup)





#### TRying to subset the whole mesh  #####

rm = cx.region_mapping_data.array_data
vertices = cx.surface.vertices
normals = cx.surface.vertex_normals
triangles = cx.surface.triangles



## region mapping number == vertices number != triangles
len(rm)
len(vertices)
len(triangles)

# todas las regiones estan representadas en el mesh?
len(conn.region_labels)
len(set(rm))


# Select cortical surface regions :: a) [3, 4, 35, 36] =  ACCr, PCCr, V1r, V2r
#                                    b) [0, 1, 3, 4] = rA1, rA2, rACC, rPCC, rCCR, rCCS
sel = [0, 1, 3, 4, 5, 6]
sel_sub = [2]

# subsetea los vertices de esas regiones
idx_sel = [list(np.argwhere(rm == roi)[:, 0]) for roi in sel]
idx_sel = [item for sublist in idx_sel for item in sublist]

# recupera los vertices y el region mapping
rm_sub = rm[idx_sel]

vertex_sub = vertices[idx_sel]
normals_sub = vertices[idx_sel]

# Quédate con los triangulos cuyos vertices están contenidos en idx_sel
tri_sub = np.array([t for t in triangles if (t[0] in idx_sel) and (t[1] in idx_sel) and (t[2] in idx_sel)])

# sustituye los ids antiguos por los nuevos
tri_sub_trans = [[idx_sel.index(t[0]), idx_sel.index(t[1]), idx_sel.index(t[2])] for t in tri_sub]



##  Guarda los datos en la carpeta de destino
out_pack = "subset-r7-surf408_pack\\"

np.savetxt(main_dir + out_pack + "region_mapping.txt", rm_sub, delimiter=" ")

localCoup = cx.local_connectivity.matrix[idx_sel][:, idx_sel]
scipy.io.savemat(main_dir + out_pack + 'local_connectivity.mat', {"LocalCoupling": localCoup})
# plt.spy(localCoup)

np.savetxt(main_dir + out_pack + "vertices.txt", vertex_sub, delimiter=" ")
np.savetxt(main_dir + out_pack + "vertex_normals.txt", normals_sub, delimiter=" ")
np.savetxt(main_dir + out_pack + "triangles.txt", tri_sub_trans, delimiter=" ")
####  TODO (Manually) Zip cortical surface files (vertices, normals, triangles) in cortical_surface.zip



# Subset connectivity data
ids = sorted(sel + sel_sub)

weights = conn.weights[ids][:, ids]
tl = conn.tract_lengths[ids][:, ids]
cortx = np.array([1, 1, 0, 1, 1, 1, 1])

# write them
np.savetxt(main_dir + out_pack + "weights.txt", weights, delimiter=" ")
np.savetxt(main_dir + out_pack + "tract_lengths.txt", tl, delimiter=" ")
np.savetxt(main_dir + out_pack + "cortical.txt", cortx, delimiter=" ", fmt="%i")
####  TODO (Manually) Zip cortical conn files in connectivity.zip
## TODO create and save centres manually


# Create a readme file and note the features of the subseted modelling data
description = ("Subset of the default cortical mesh included in TVB data to 7 regions with one subcortical (amygdala_r). \n\n"
               "Connectivity - 7 rois; \n"
               "Cortex - 408 vertices from 6 regions (rA1, rA2, rACC, rPCC, rCCR, rCCS); "
               "the regions that are contained in the mesh, will be simulated through the vertices. \n\n"
               "local_connectivity and region_mapping are also subsets from the default surface data in TVB.")

with open(main_dir + out_pack + 'readme.txt', 'w') as f:
    f.write(description)

















#Initialise a surface

local = main_dir + out_pack + "local_connectivity.mat"
# local = None

cx = cortex.Cortex.from_file(source_file=main_dir + out_pack + "cortical_surface.zip",
                                         region_mapping_file=main_dir + out_pack + "region_mapping.txt",
                                         local_connectivity_file=local)

if not local:
    cx.local_connectivity = local_connectivity.LocalConnectivity(cutoff=30.0,  # def=40mm
                                             surface=cx.region_mapping_data.surface,
                                             equation=equations.Gaussian())

    # localCoup = cx.local_connectivity.matrix
    # scipy.io.savemat(main_dir + surfpack + 'local_coup.mat', {"LocalCoupling": localCoup})
    # plt.spy(localCoup)


cx.region_mapping_data.connectivity = conn
cx.coupling_strength = np.array([2**-10])


# Monitors
mons = (monitors.Raw(), monitors.TemporalAverage(), monitors.SpatialAverage(), )


sim = simulator.Simulator(model=oscillator, connectivity=conn, coupling=coup, surface=cx,
                          integrator=heunint, monitors=mons)
sim.configure()

output = sim.run(simulation_length=simLength)




