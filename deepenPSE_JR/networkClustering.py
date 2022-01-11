######
#####

# Second approach, with communities
# https://pypi.org/project/communities/
# https://datascience.stackexchange.com/questions/4974/partitioning-weighted-undirected-graph
####
####
from tvb.simulator.lab import connectivity
import numpy as np
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from communities.algorithms import *
from communities.visualization import draw_communities
from communities.utilities import modularity_matrix, modularity
import os


conn = connectivity.Connectivity.from_file(os.getcwd()+ "\\CTB_data\\output\\AVG_NEMOS_AAL2red.zip")
plv_emp = np.loadtxt("CTB_data\\output\\FC_AVG_NEMOS\\3-alpha_plv.txt")


communs = spectral_clustering (conn.weights, 10)
title='weights - spectral'

indexes = [item for c in communs for item in c]
plv_sorted = plv_emp[:, indexes][indexes]


fig = go.Figure(data=go.Heatmap(z=plv_sorted, colorscale='Viridis'))
fig.update_layout(title='Phase Locking Value - clustered by %s' % title)
pio.write_html(fig, file= "figures/PLV_%s.html" % title, auto_open=True)






######
#####

# First approach, with scikit network. Failed to organize my network.
####
####




from IPython.display import SVG
import os
#
# import numpy as np
# import scipy.sparse
# from sknetwork.clustering import Louvain, modularity
# from sknetwork.visualization import svg_graph
#
# # Create network and plot simply
# plv_emp = np.loadtxt("CTB_data\\output\\FC_AVG_NEMOS\\3-alpha_plv.txt")
#
# # ######## work on LEFT HEMISPHERE
# # left_h=np.arange(0, 92, 2)
# # plv_left = plv_emp[:, left_h][left_h]
# # plv_left_csr = scipy.sparse.csr_matrix(plv_left)
# #
# # # Cluster nodes and plot by cluster
# # louvain = Louvain()
# # labels = louvain.fit_transform(plv_left_csr)
# #
# # # image = svg_graph(plv_emp_adj, labels=labels, display_node_weight=True,node_size_max=12)
# # # SVG(image)
# #
# # # Count members per cluster
# # labels_unique, counts = np.unique(labels, return_counts=True)
# # # Measure clusters modularity
# # modularity(plv_left, labels)
# #
# # # Create indexes array
# # count = 0
# # indexes = np.ndarray((len(plv_emp)), dtype=int)
# # for i in labels_unique:
# #     for j in range(len(labels)):
# #         if labels[j] == i:
# #             indexes[j] = int(count)
# #             count += 1
#
#
# # ######## work on RIGHT HEMISPHERE
# # right_h=np.arange(1, 92, 2)
# # plv_right = plv_emp[:, right_h][right_h]
# # plv_right_csr = scipy.sparse.csr_matrix(plv_right)
# #
# # # Cluster nodes and plot by cluster
# # labels = louvain.fit_transform(plv_right)
# #
# # # image = svg_graph(plv_emp_adj, labels=labels, display_node_weight=True,node_size_max=12)
# # # SVG(image)
# #
# # # Count members per cluster
# # labels_unique, counts = np.unique(labels, return_counts=True)
# # # Measure clusters modularity
# # modularity(plv_left, labels)
# #
# # # Create indexes array
# # for i in labels_unique:
# #     for j in range(len(labels)):
# #         if labels[j] == i:
# #             indexes[46+j] = int(count)
# #             count += 1
#
#
# # # Reorder matrix
# # plv_sorted=plv_emp[:, indexes][indexes]
# #
# # import plotly.graph_objects as go  # for data visualisation
# # import plotly.io as pio
# #
# # fig = go.Figure(data=go.Heatmap(z=plv_sorted, colorscale='Viridis'))
# # fig.update_layout(title='Phase Locking Value')
# # pio.write_html(fig, file= "figures/dFC_w1s0.05.html", auto_open=True)
#
#
# # ######## all toghether
# plv_csr = scipy.sparse.csr_matrix(plv_emp)
#
# # Cluster nodes and plot by cluster
# louvain = Louvain()
# labels = louvain.fit_transform(plv_csr)
#
# # # image = svg_graph(plv_emp_adj, labels=labels, display_node_weight=True,node_size_max=12)
# # # SVG(image)
#
# # # Count members per cluster
# labels_unique, counts = np.unique(labels, return_counts=True)
# # # Measure clusters modularity
# modularity(plv_csr, labels)
#
# # Create indexes array
# count = 0
# indexes = np.ndarray((len(plv_emp)), dtype=int)
# for i in labels_unique:
#     for j in range(len(labels)):
#         if labels[j] == i:
#             indexes[j] = int(count)
#             count += 1
#
# plv_sorted=plv_emp[:, indexes][indexes]
#
# import plotly.graph_objects as go  # for data visualisation
# import plotly.io as pio
#
# fig = go.Figure(data=go.Heatmap(z=plv_sorted, colorscale='Viridis'))
# fig.update_layout(title='Phase Locking Value')
# pio.write_html(fig, file= "figures/dFC_w1s0.05.html", auto_open=True)
#
#
#
