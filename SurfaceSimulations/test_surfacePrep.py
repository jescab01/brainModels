
import gdist
import numpy as np
from tvb.simulator.lab import *
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio


surfpack = "subHCPex-r3R-surf504_pack\\"
# surfpack = "default-r76-surf16k_pack\\"

data_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\.DataTemp\\"
main_dir = "E:\LCCN_Local\PycharmProjects\AlphaOrigins\SurfSim\\"

conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")

cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                             region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                             local_connectivity_file=None)  # cutoff=40mm (gaussian)

vertices = cx.surface.vertices
triangles = cx.surface.triangles

fig = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=cx.region_mapping_data.array_data,
                          colorbar=dict(title='mV', x=0.9, thickness=15),
                          colorscale="Turbo",
                          # Intensity of each vertex, which will be interpolated and color-coded
                          intensity=cx.region_mapping_data.array_data,
                          # i, j and k give the vertices of triangles
                          # here we represent the 4 triangles of the tetrahedron surface
                          i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], showscale=True))
fig.update_layout(template="plotly_white",)
pio.write_html(fig, file="figures/3Dbrain.html", auto_open=True)









## GDIST
vertices = cx.vertices.astype(np.float64)
triangles = cx.triangles.astype(np.int32)

src = np.array([1], dtype=np.int32)
trg = np.array([2], dtype=np.int32)

gdist.compute_gdist(vertices, triangles, src, trg)

# gdist.local_gdist_matrix(vertices, triangles, 1)


##############
import numpy
temp = numpy.loadtxt("E:\LCCN_Local\PycharmProjects\IntegrativeRhythms\SurfSim\\flat_triangular_mesh.txt", skiprows=1)
vertices = temp[0:121].astype(numpy.float64)
triangles = temp[121:321].astype(numpy.int32)
src = numpy.array([1], dtype=numpy.int32)
trg = numpy.array([2], dtype=numpy.int32)

triangles = triangles[:-2]

import gdist
gdist.compute_gdist(vertices, triangles, source_indices=src, target_indices=trg)
