import os
import glob

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import time


def WPplot(df, folder, title, auto_open, working_point=None, threshold=1):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("rPLV (emp-sim)", "optimum", "PLE"),
                        specs=[[{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Conduction speed (m/s)", y_title="Coupling factor")

    fig.add_trace(go.Heatmap(z=df.Alpha, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(x=0.3, thickness=7),
                             reversescale=True, zmin=-0.5,  zmax=0.5), row=1, col=1)

    th=threshold*len(df)
    temp=df.sort_values(by=["optimum"], ascending=False)
    temp=temp.head(int(th))

    fig.add_trace(go.Heatmap(z=temp.optimum, x=temp.speed, y=temp.G, colorscale='Aggrnyl',
                             colorbar=dict(x=0.66, thickness=7)), row=1, col=2)

    fig.add_trace(go.Heatmap(z=df.ple_alpha, x=df.speed, y=df.G, colorscale='Plasma',
                             colorbar=dict(thickness=7)), row=1, col=3)

    if working_point:
        (owp_g, owp_s) = working_point

        [fig.add_trace(
            go.Scatter(
                x=[owp_s], y=[owp_g], mode='markers',
                marker=dict(color='black', line_width=2, size=[10], opacity=0.5, symbol='circle-open',
                            showscale=False)),
            row=1, col=j) for j in range(1, 4)]

    fig.update_layout(showlegend=False,
        title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
    pio.write_html(fig, file=folder + "/WorkingPoints-g&s_%s.html" % title, auto_open=auto_open)

for mode in ["jrd", "cb", "jrdcb", "jr"]:
    ### CLUSTER
    # folder='D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\Jescab01\PSE\\' # CLUSTER
    # folder_ple = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\deepenPSE\PSE\cluster.PSEparallel-FFT.PLV.PLE - all subjects\\'
    # folder_fc = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\deepenPSE\PSE\cluster.PSE-PLV_10reps - all subjects\\'
    cluster_folder = "D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\\wpavg\PSE\\"

    wd = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\working_points\_wPLEavg\\'
    main_folder = wd + "\\" + "PSE"
    if os.path.isdir(main_folder) == False:
        os.mkdir(main_folder)

    specific_folder = main_folder + "\\PSE_singleWORKINGPOINT-avgNEMOS." + mode + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
    os.mkdir(specific_folder)

    ### GLOBAL WORKING POINT
    subjects = ["NEMOS_AVG"]

    df_ple = pd.DataFrame()
    df_fft = pd.DataFrame()
    df_fc = pd.DataFrame()

    for subj in subjects:
        ple_csv = glob.glob(cluster_folder + 'PSEp' + subj + mode + '-*/PSE_PLE*.csv')
        fft_csv = glob.glob(cluster_folder + 'PSEp' + subj + mode + '-*/PSE_FFT*.csv')
        fc_csv = glob.glob(cluster_folder + 'PSEp' + subj + mode + '-*/PSE_FC*.csv')

        for f in ple_csv:
            df_ple = df_ple.append(pd.read_csv(f))
        for f in fft_csv:
            df_fft = df_fft.append(pd.read_csv(f))
        for f in fc_csv:
            df_fc = df_fc.append(pd.read_csv(f))

        #### Individual PSE
        # Merge PLE and PLV dataframes
        df_OWP1 = df_fc.sort_values(by=["G", "speed"])
        df_OWP1["ple_alpha"] = list(df_ple.sort_values(by=["G", "speed"])["Alpha"])

        ## Calculate working point - plv all bands and ple; threshold above 4 m/s; WEIGHTED
        dfOWP_opt = df_OWP1
        dfOWP_opt["optimum"] = dfOWP_opt[["Alpha", "ple_alpha"]].mean(axis=1)
        dfOWP_opt.to_csv(specific_folder + "/wpData_" + subj + "_PLValpha-PLEalpha.csv")

        WPplot(df_OWP1, specific_folder, subj + " - PLValpha.PLEalpha _ mode " + mode, False)

        #### Working Point
        # Merge PLE and PLV dataframes
        df_OWP = df_fc.sort_values(by=["G", "speed"])
        df_OWP["ple_alpha"] = list(df_ple.sort_values(by=["G", "speed"])["Alpha"])

        dfOWP_opt = df_OWP.groupby(["G", "speed"])[["G", "speed", "Alpha", "ple_alpha"]].mean()
        dfOWP_opt["optimum"] = dfOWP_opt[["Alpha", "ple_alpha"]].mean(axis=1)

        (g, s) = dfOWP_opt.loc[dfOWP_opt["speed"] > 0.5]["optimum"].idxmax()

        WPplot(dfOWP_opt, specific_folder, "Group Average - PLValpha.PLEalpha  _  mode " + mode, True, working_point=(g, s))



        ## Calculate working point - plv all bands and ple; threshold above 4 m/s; WEIGHTED
        # threshold = 4
        # dfOWP_1 = dfOWP
        # dfOWP_1["optimum"] = \
        #     dfOWP_1["Delta"] * 0.05 + dfOWP_1["Theta"] * 0.15 + dfOWP_1["Alpha"] * 0.3 + \
        #     dfOWP_1["Beta"] * 0.15 + dfOWP_1["Gamma"] * 0.05 + dfOWP_1["ple_alpha"] * 0.3
        #
        # (g, s) = dfOWP_1.loc[dfOWP["speed"] > threshold]["optimum"].idxmax()
        #
        # wp_temp = [[subj]+dfOWP_1.loc[(dfOWP_1["G"] == g) & (dfOWP_1["speed"] == s)].values.tolist()[0]]
        # wp_df = wp_df.append(pd.DataFrame(wp_temp))
        #
        # WPplot(dfOWP_1, specific_folder, subj+" - PLVallbands.PLE", False, working_point=(g, s))


        ## calculate working point - just alpha 0.5ple|0.5plv

        # dfOWP_1 = dfOWP
        # dfOWP_1["optimum"] = dfOWP_1["Alpha"] * 0.5 + dfOWP_1["ple_alpha"] * 0.5
        #
        # (g, s) = dfOWP_1["optimum"].idxmax()
        #
        #
        # wp_temp = [[subj]+dfOWP_1.loc[(dfOWP_1["G"] == g) & (dfOWP_1["speed"] == s)].values.tolist()[0]]
        # wp_df=wp_df.append(pd.DataFrame(wp_temp))
        #
        # WPplot(dfOWP_1, specific_folder, subj+" - alpha ple.plv", False, working_point=(g, s))


        ## Calculate working point -

    # wp_df.columns=["subject", "G", "speed", "PLVd", "PLVt", "PLVa", "PLVb", "PLVg", "PLEa", "optimum"]
    # wp_df.to_csv(specific_folder+"/working_points.csv", index=False)

    ## WORKING POINT - SUBJECT specific
    # for subj in subjects:
    #
    #     df_ple = pd.DataFrame()
    #     df_fft = pd.DataFrame()
    #     df_fc = pd.DataFrame()
    #
    #     ple_csv = glob.glob(cluster_folder + 'PSEp' + subj + mode + '-*/PSE_PLE*.csv')
    #     fft_csv = glob.glob(cluster_folder + 'PSEp' + subj +'-*/PSE_FFT*.csv')
    #     fc_csv = glob.glob(cluster_folder + 'PSEp' + subj + '-*/PSE_FC*.csv')

        # for f in ple_csv:
        #     df_ple = df_ple.append(pd.read_csv(f))
        # for f in fft_csv:
        #     df_fft = df_fft.append(pd.read_csv(f))
        # for f in fc_csv:
        #     df_fc = df_fc.append(pd.read_csv(f))








