import os

import numpy as np
import scipy.signal

from mne import filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio

from toolbox import PLV, AEC, epochingTool


###### Sliding Window Approach
def dynamic_fc(data, samplingFreq, window, step, measure="PLV", plot="OFF", folder='figures', lowcut=8, highcut=12, filtered=True, auto_open=False):
    '''
    Calculates dynamical Functional Connectivity using the classical method of sliding windows.

    :param data: Signals in shape [ROIS x time]
    :param samplingFreq: sampling frequency (Hz)
    :param window: Seconds of sliding window
    :param step: Movement step for sliding window
    :param measure: FC measure (PLV; AEC)
    :param plot: Plot dFC matrix?
    :param folder: To save figures output
    :param auto_open: on browser.
    :return:
    '''
    window_ = window * 1000
    step_ = step * 1000

    if len(data[0]) > window_:
        if filtered:
            filterSignals = data
        else:
            # Band-pass filtering
            filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut)

        print("Calculating dFC matrix...")
        matrices_fc = list()
        for w in np.arange(0, (len(data[0])) - window_, step_, 'int'):

            print('%s %i / %i' % (measure, w / step_, ((len(data[0])) - window_) / step_))

            signals = filterSignals[:, w:w + window_]

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            if window_ >= 4000:
                efSignals = epochingTool(signals, 4, samplingFreq, "signals")
            else:
                efSignals = epochingTool(signals, window_//1000, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # # timeseriesPlot(emp_signals, raw_time, regionLabels)
            # plotConversions(emp_signals[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0], band="alpha")

            if measure == "PLV":
                matrices_fc.append(PLV(efPhase))

            elif measure == "AEC":
                matrices_fc.append(AEC(efEnvelope))

        dFC_matrix = np.zeros((len(matrices_fc), len(matrices_fc)))
        for t1 in range(len(matrices_fc)):
            for t2 in range(len(matrices_fc)):
                dFC_matrix[t1, t2] = np.corrcoef(matrices_fc[t1][np.triu_indices(len(matrices_fc[0]), 1)],
                                                 matrices_fc[t2][np.triu_indices(len(matrices_fc[0]), 1)])[1, 0]

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=dFC_matrix, x=np.arange(0, len(data[0]), step), y=np.arange(0, len(data[0]), step_),
                                            colorscale='Viridis'))
            fig.update_layout(title='Functional Connectivity Dynamics')
            fig.update_xaxes(title="Time 1 (seconds)")
            fig.update_yaxes(title="Time 2 (seconds)")
            pio.write_html(fig, file=folder + "/PLV_" ".html", auto_open=auto_open)

        return dFC_matrix

    else:
        print('Error: Signal length should be longer than window length (%i sec)' % window)


def kuramoto_order(data, samplingFreq, lowcut=8, highcut=12, filtered=False):

    print("Calculating Kuramoto order paramter...")
    if filtered:
        filterSignals=data

    else:
        # Band-pass filtering
        filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut)

    analyticalSignal = scipy.signal.hilbert(filterSignals)
    # Get instantaneous phase by channel
    efPhase = np.angle(analyticalSignal)

    # Kuramoto order parameter in time
    kuramoto_array = abs(np.sum(np.exp(1j * efPhase), axis=0))/len(efPhase)
    # Average Kuramoto order parameter for the set of signals
    kuramoto_avg = np.average(kuramoto_array)
    kuramoto_sd = np.std(kuramoto_array)

    return kuramoto_sd, kuramoto_avg