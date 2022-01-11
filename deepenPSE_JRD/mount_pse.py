import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob
import time


# main_dir = "D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\\ths\PSE\\"
main_dir="D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\\jrd\PSE\\"

for test, w in [("JRD", 0.8), ("basicJR", 1.0)]:

    emp_subj = "NEMOS_035"  # "basicJR"
    output_dir = main_dir + "PSE_"+test+"." + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + "\\"

    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    folders = glob.glob1(main_dir, 'PSE'+test+'*')

    df_fc = pd.DataFrame()
    df_amp = pd.DataFrame()

    for folder in folders:
        specific_folder = main_dir + folder + "\\"
        file = glob.glob1(specific_folder, 'PSE_amp*.csv')
        for f in file:
            df_amp = df_amp.append(pd.read_csv(specific_folder + f), ignore_index=True)
        file = glob.glob1(specific_folder, 'PSE_FC*.csv')
        for f in file:
            df_fc = df_fc.append(pd.read_csv(specific_folder + f), ignore_index=True)

    df_fc.to_csv(output_dir+'JRD_FC_fullDF.csv')
    df_amp.to_csv(output_dir+'JRD_amp_fullDF.csv')


    ## Average rounds
    df_m = df_fc.groupby(["g", "s", "band"])[["PLVr", "dFC_ksd", "meanKO", "sdKO", "sdKO_emp"]].mean().reset_index()
    df_sd = df_fc.groupby(["g", "s", "band"])[["PLVr", "dFC_ksd", "meanKO", "sdKO", "sdKO_emp"]].std().reset_index()

    fig = make_subplots(rows=3, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                        specs=[[{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}], [{}, {}, {}, {}, {}]],
                        shared_yaxes=True, shared_xaxes=True, row_titles=["FCr", "dFC (ksd)", "sdKO"],
                        x_title="Conduction speed (m/s)", y_title="Coupling factor")

    bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma1"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]
    for i, band in enumerate(range(len(bands[0]))):

        df_subset = df_m.loc[df_m["band"] == bands[0][band]]

        fig.add_trace(go.Heatmap(z=df_subset.PLVr, x=df_subset.s, y=df_subset.g, colorscale='RdBu', reversescale=True,
                                 colorbar=dict(title="avg rFC", y=0.9, len=0.3, thickness=10), zmin=-0.5, zmax=0.5, showlegend=False), row=1, col=i+1)

        fig.add_trace(go.Heatmap(z=df_subset.dFC_ksd, x=df_subset.s, y=df_subset.g, colorscale='Viridis', reversescale=True,  # lower stat -> higher emp-sim correspondance
                                  zmin=0, zmax=1, colorbar=dict(title="ksd", y=0.5, len=0.3, thickness=10)), row=2, col=i+1)

        fig.add_trace(go.Heatmap(z=df_subset.sdKO, x=df_subset.s, y=df_subset.g,
                                 zmin=0, zmax=0.3, colorbar=dict(title="sdKO", thickness=10, y=0.1, len=0.3)), row=3, col=i+1)

    fig.update_layout(title_text='%s simulations || [w = %0.2f] %s' % (test, w, emp_subj))
    pio.write_html(fig, file=output_dir + "/paramSpace-%s.html" % test, auto_open=True)





