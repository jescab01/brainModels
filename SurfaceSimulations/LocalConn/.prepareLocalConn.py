
import os
import numpy as np
import scipy.io
from scipy.stats import multivariate_normal
from tvb.simulator.lab import cortex, local_connectivity, equations, connectivity
from sklearn.preprocessing import normalize

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio




ao_dir = os.getcwd().split("PycharmProjects")[0] + 'PycharmProjects\\AlphaOrigins\\'

data_dir = ao_dir + ".DataTemp\\"


surfs = ["surfdisc4k"]#, "surfdisc8k", "surfdisc17k"]
norms = [False]


for surf in surfs:
    for norm in norms:

        surfpack = "HCPex-r426-" + surf + "_pack\\"


        # Open surface info to obtain mean edge length
        with open(data_dir + surfpack + "surface_info.txt", "r") as file:
            surfinfo = file.read()
        file.close()

        avg_edge_length = float(surfinfo.splitlines()[25].split(' ')[-2])


        # LC Kernel parameters
        amp = 1
        sigma = avg_edge_length
        cutoff = 1.5 * avg_edge_length


        norm_text = "-norm" if norm else ""
        lc_title = "local_connectivity-amp%0.1fsig%0.2fcut%0.2f%s" % (amp, sigma, cutoff, norm_text)


        # STRUCTURAL CONNECTIVITY      #########################################
        conn = connectivity.Connectivity.from_file(data_dir + surfpack + "connectivity.zip")


        # CORTICAL SURFACE        #########################################
        cx = cortex.Cortex.from_file(source_file=data_dir + surfpack + "cortical_surface.zip",
                                     region_mapping_file=data_dir + surfpack + "region_mapping.txt",
                                     local_connectivity_file=None)

        cx.region_mapping_data.connectivity = conn


        cx.local_connectivity = (
            local_connectivity.LocalConnectivity(
                cutoff=cutoff, surface=cx.region_mapping_data.surface,
                equation=equations.Gaussian(parameters=dict(amp=amp, sigma=sigma, midpoint=0, offset=0))))


        cx.configure()

        # Normalize if required
        if norm:

            cx.local_connectivity.matrix = normalize(cx.local_connectivity.matrix, norm="max")


        # SAVE local conn
        if not os.path.isdir(data_dir + surfpack + "local_conn"):
            os.mkdir(data_dir + surfpack + "local_conn")

        scipy.io.savemat(data_dir + surfpack + 'local_conn\\' + lc_title + ".mat", {"LocalCoupling": cx.local_connectivity.matrix})

        # plt.spy(localCoup)






        ## PLOTTING
        # Plot mesh with local conn for some selected node
        vertex = 2606 if "4k" in surf else \
            5111 if "8k" in surf else\
            10661 if "17k" in surf else\
            5444 if "9k" in surf else\
            10661 if "18k" in surf else 5

        title = surf + "_" + lc_title + "_v%i" % vertex

        row_data = cx.local_connectivity.matrix[vertex, :].toarray()[0]
        col_data = cx.local_connectivity.matrix[:, vertex].toarray()[:, 0]

        vertices, triangles = cx.surface.vertices, cx.surface.triangles

        labels = ["%i - %s <br> From v0 = %0.3e;<br> To v0 = %0.3e" % (pre, conn.region_labels[id], col_data[pre], row_data[pre])
                  for pre, id in enumerate(cx.region_mapping)]

        # Colorbar
        bar_title = 'LocalConn<br>(n.s.)'
        data = np.concatenate((row_data, col_data))
        cmax, cmin = np.max(data[~np.isnan(data) & ~np.isinf(data)]), np.min(data[~np.isnan(data) & ~np.isinf(data)])
        zeroc = np.abs(cmin / (cmax - cmin))
        fiftyc = zeroc + ((1 - zeroc) / 2)

        colorscale = [[zeroc, 'whitesmoke'], [fiftyc, 'gold'], [1, 'crimson']]

        # PLOT
        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
                            column_titles=["To vertex", "From vertex", ""])

        for j, data in enumerate([row_data, col_data]):

            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], hovertext=labels, hoverinfo="text",
                      colorbar=dict(title=bar_title, x=0.675, thickness=10, len=0.8), colorscale=colorscale, cmax=cmax, cmin=cmin,
                      # Intensity of each vertex, which will be interpolated and color-coded
                      intensity=data,
                      # i, j and k give the vertices of triangles
                      # here we represent the 4 triangles of the tetrahedron surface
                      i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], showscale=True), row=1, col=1+j)

        # Add the Gaussian
        x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        pos = np.dstack((x, y))
        covariance = [[sigma**2, 0], [0, sigma**2]]
        # Generate Gaussian values
        z = amp * multivariate_normal.pdf(pos, [0, 0], covariance)
        # Create 3D scatter plot
        fig.add_trace(go.Surface(z=z, x=x, y=y, colorbar=dict(title="Gaussian", x=1, thickness=10, len=0.5)), row=1, col=3)

        cam_far = 2.75
        fig.update_layout(template="plotly_white", title=title, height=600,
                          scene3=dict(camera=dict(eye=dict(x=cam_far, y=cam_far, z=cam_far), center=dict(x=0.5, y=0, z=0.25))))

        pio.write_html(fig, file=data_dir + surfpack + "local_conn/" + title + ".html", auto_open=True)


