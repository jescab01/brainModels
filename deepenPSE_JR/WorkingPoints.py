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



wd = os.getcwd()
main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_singleWORKINGPOINT-allNEMOS-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)


### CLUSTER
# folder='D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\Jescab01\PSE\\' # CLUSTER

wp_df=pd.DataFrame()

folder_ple = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\deepenPSE\PSE\cluster.PSEparallel-FFT.PLV.PLE - all subjects\\'
folder_fc = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\deepenPSE\PSE\cluster.PSE-PLV_10reps - all subjects\\'

subjects = ["NEMOS_0" + str(i) for i in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]]
# subjects.append("AVG_NEMOS")

intersubject_OWP=pd.DataFrame()

for subj in subjects:
    ple_csv = glob.glob(folder_ple + 'PSEparallel.1995JansenRit-' + subj + '-m06d18y2021/PLE*.csv')[0]
    fft_csv = glob.glob(folder_fc + 'PSE.1995JansenRit-' + subj + '-m05d17y2021/*FFT*.csv')[0]
    fc_csv = glob.glob(folder_fc + 'PSE.1995JansenRit-' + subj + '-m05d17y2021/*FC*.csv')[0]

    df_ple = pd.read_csv(ple_csv)
    df_fft = pd.read_csv(fft_csv)
    df_fc = pd.read_csv(fc_csv)

    dfPLE = df_ple.drop(columns="Unnamed: 0").sort_values(by=['G', 'speed'])
    dfPLE = dfPLE.groupby(["G","speed"])[["G", "speed", "Alpha"]].mean()

    dfFFT_m = df_fft.groupby(["G", "speed"])[["G", "speed", "mS_peak"]].mean()

    dfPLV_m = df_fc.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
    dfPLV_m.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]


    #### Working Point
    # Merge PLE and PLV dataframes
    dfOWP = dfPLV_m
    dfOWP["ple_alpha"] = list(dfPLE["Alpha"])

    intersubject_OWP = intersubject_OWP.append(dfOWP)


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


dfOWP_1 = intersubject_OWP[["Delta", "Theta", "Alpha", "Beta", "Gamma", "ple_alpha"]].reset_index().groupby(["G","speed"])[["G", "speed","Delta", "Theta", "Alpha", "Beta", "Gamma", "ple_alpha"]].mean()
dfOWP_1["optimum"] = dfOWP_1[["Delta", "Theta", "Alpha", "Beta", "Gamma", "ple_alpha"]].mean(axis=1)

(g, s) = dfOWP_1["optimum"].idxmax()

WPplot(dfOWP_1, specific_folder, "Subjects Average - PLVallbands.PLE", False, working_point=(g, s))

for subj in subjects:

    ple_csv = glob.glob(folder_ple + 'PSEparallel.1995JansenRit-' + subj + '-m06d18y2021/PLE*.csv')[0]
    fft_csv = glob.glob(folder_fc + 'PSE.1995JansenRit-' + subj + '-m05d17y2021/*FFT*.csv')[0]
    fc_csv = glob.glob(folder_fc + 'PSE.1995JansenRit-' + subj + '-m05d17y2021/*FC*.csv')[0]

    df_ple = pd.read_csv(ple_csv)
    df_fft = pd.read_csv(fft_csv)
    df_fc = pd.read_csv(fc_csv)

    dfPLE = df_ple.drop(columns="Unnamed: 0").sort_values(by=['G', 'speed'])
    dfPLE = dfPLE.groupby(["G","speed"])[["G", "speed", "Alpha"]].mean()

    dfFFT_m = df_fft.groupby(["G", "speed"])[["G", "speed", "mS_peak"]].mean()

    dfPLV_m = df_fc.groupby(["G", "speed"])[["G", "speed", "plvD_r", "plvT_r", "plvA_r", "plvB_r", "plvG_r"]].mean()
    dfPLV_m.columns = ["G", "speed", "Delta", "Theta", "Alpha", "Beta", "Gamma"]

    #### Working Point
    # Merge PLE and PLV dataframes
    dfOWP = dfPLV_m
    dfOWP["ple_alpha"] = list(dfPLE["Alpha"])

    ## Calculate working point - plv all bands and ple; threshold above 4 m/s; WEIGHTED

    dfOWP_1 = dfOWP
    dfOWP_1["optimum"] = \
        dfOWP_1["Delta"] * 0.05 + dfOWP_1["Theta"] * 0.15 + dfOWP_1["Alpha"] * 0.3 + \
        dfOWP_1["Beta"] * 0.15 + dfOWP_1["Gamma"] * 0.05 + dfOWP_1["ple_alpha"] * 0.3

    WPplot(dfOWP_1, specific_folder, subj+" - PLVallbands.PLE", False, working_point=(g, s))






