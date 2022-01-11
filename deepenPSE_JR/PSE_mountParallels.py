import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# for i in [35,49,50,58,59,64,65,71,75,77]:

emp_subj="NEMOS_049"  # "NEMOS_0"+str(i)

# dir = os.getcwd()+"/PSE/PSEparallel.1995JansenRit-AVG_NEMOS-m06d16y2021/"
# dir = "D:\\Users\\Jesus CabreraAlvarez\\Desktop\\CLUSTER\\parallel\\PSE\\PSEparallel.1995JansenRit-"+emp_subj+"-m06d18y2021\\"
dir = "D:\\Users\Jesus CabreraAlvarez\Desktop\CLUSTER\woTh\PSE\\PSEp_woTh.1995JansenRit-"+emp_subj+"-m07d01y2021\\"

files = os.listdir(dir)

df_fc = pd.DataFrame()
df_ple = pd.DataFrame()
df_fft = pd.DataFrame()

for file in files:
    if 'PSE_PLE' in file:
        df_ple=df_ple.append(pd.read_csv(dir+file), ignore_index=True)

    if 'PSE_FC' in file:
        df_fc=df_fc.append(pd.read_csv(dir + file), ignore_index=True)

    if 'PSE_FFTpeaks' in file:
        df_fft=df_fft.append(pd.read_csv(dir + file), ignore_index=True)

df_ple.to_csv(dir+'PLE_fullDF.csv')
df_fft.to_csv(dir+'FFT_fullDF.csv')
df_fc.to_csv(dir+'FC_fullDF.csv')


fig = make_subplots(rows=1, cols=3, subplot_titles=("FFT peak", "rPLV (sim-emp)", "PLE - Phase Lag Entropy"),
                    specs=[[{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                    x_title="Conduction speed (m/s)", y_title="Coupling factor")

fig.add_trace(go.Heatmap(z=df_fft.mS_peak, x=df_fft.speed, y=df_fft.G, colorscale='Viridis',
                         reversescale=False, showscale=True, colorbar=dict(x=0.30, thickness=7)), row=1, col=1)
fig.add_trace(go.Heatmap(z=df_fc.Alpha, x=df_fc.speed, y=df_fc.G, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                         showscale=True, colorbar=dict(x=0.66, thickness=7)), row=1, col=2)
fig.add_trace(go.Heatmap(z=df_ple.Alpha, x=df_ple.speed, y=df_ple.G, colorscale='Plasma', colorbar=dict(thickness=7),
                         reversescale=False, showscale=True), row=1, col=3)


fig.update_layout(
    title_text='Mix of measures in '+emp_subj)
pio.write_html(fig, file=dir + "mix_paramSpace-g&s.html", auto_open=True)
