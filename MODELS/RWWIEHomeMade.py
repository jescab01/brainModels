import numpy as np
#
# S=np.array([0.5, 0.5])
# G=np.array([0])
# I_o=np.array([0.381])
# J_N=np.array([0.15])
# J_i=np.array([1])
# W_e=np.array([1])
# W_i=np.array([0.7])
# a_e=np.array([0.310])
# a_i=np.array([0.615])
# b_e=np.array([0.125])
# b_i=np.array([0.177])
# d_e=np.array([160])
# d_i=np.array([87])
# gamma_e=np.array([0.641])
# gamma_i=np.array([1])
# lamda=np.array([0])
# tau_e=np.array([100])
# tau_i=np.array([10])
# w_p=np.array([1.4])



import numpy as np


def ReducedWongWangIE(S=np.array([0.5, 0.5]),
                      G=np.array([0]), I_o=np.array([0.5]),
                      J_N=np.array([0.15]), J_i=np.array([1]),
                      W_e=np.array([1]), W_i=np.array([0.7]),
                      a_e=np.array([0.310]), a_i=np.array([0.615]),
                      b_e=np.array([0.125]), b_i=np.array([0.177]),
                      d_e=np.array([160]), d_i=np.array([87]),
                      gamma_e=np.array([0.641]), gamma_i=np.array([1]),
                      lamda=np.array([0]),
                      tau_e=np.array([100]), tau_i=np.array([10]),
                      w_p=np.array([1.4]), sigma=np.array([0.01])):

    I_e = W_e * I_o + w_p * J_N * S[0] - J_i * S[1]
    H_e = a_e * I_e - b_e / (1 - np.exp(-d_e * (a_e * I_e - b_e)))
    dS_e = - (S[0] / tau_e) + (1 - S[0]) * gamma_e * H_e + np.random.normal() * sigma

    I_i = W_i * I_o + w_p * J_N * S[0] - S[1]
    H_i = a_i * I_i - b_i / (1 - np.exp(-d_i * (a_i * I_i - b_i)))
    dS_i = - (S[1] / tau_i) + gamma_i * H_i + np.random.normal() * sigma

    computed = np.array([dS_e, dS_i, H_e, H_i, I_e, I_i])

    return computed


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
import time


def timeseriesPlot(signals, timepoints):

    fig=make_subplots(rows=4, cols=2, subplot_titles=("S_e","S_i","dS_e","dS_i","H_e","H_i","I_e","I_i"))

    fig.update_layout(title="ReducedWongWangIE")

    fig.add_trace(go.Scatter(y=signals[0], x=timepoints), row=1, col=1)
    fig.add_trace(go.Scatter(y=signals[2], x=timepoints), row=2, col=1)
    fig.add_trace(go.Scatter(y=signals[4], x=timepoints), row=3, col=1)
    fig.add_trace(go.Scatter(y=signals[6], x=timepoints), row=4, col=1)

    fig.add_trace(go.Scatter(y=signals[1], x=timepoints), row=1, col=2)
    fig.add_trace(go.Scatter(y=signals[3], x=timepoints), row=2, col=2)
    fig.add_trace(go.Scatter(y=signals[5], x=timepoints), row=3, col=2)
    fig.add_trace(go.Scatter(y=signals[7], x=timepoints), row=4, col=2)

    pio.write_html(fig, file="RWWIE.html", auto_open=True)


## Simulation
simLength = 3000 # ms
timepoints=np.arange(0, simLength)

S = np.array([[0.5], [0.5]])  # initial states for Se and Si
output=np.zeros((8,1))

for t in timepoints:
    print("Simulation time - %0.2f" % t, end="\r")
    computed = ReducedWongWangIE(S)
    S+=computed[:2]

    output=np.append(output, np.append(S, computed, axis=0), axis=1)

output=output[:,1:]
timeseriesPlot(output, timepoints)

from toolbox import FFTplot
FFTplot(output, simLength, ["S_e","S_i","dS_e","dS_i","H_e","H_i","I_e","I_i"], mode="html")