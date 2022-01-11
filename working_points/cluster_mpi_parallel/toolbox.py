# This is going to be a script with useful functions I will be using frequently.
import math
import time
import glob
import os
from collections import Counter

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.signal import firwin, filtfilt, hilbert
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


# Signals
def timeseriesPlot(signals, timepoints, regionLabels, folder="figures", title=None, mode="html"):
    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='time (ms)'), yaxis=dict(title='Voltage')))
    for ch in range(len(signals)):
        fig.add_scatter(x=timepoints, y=signals[ch], name=regionLabels[ch])

    if title is None:
        title = "TimeSeries"

    if mode == "html":
        pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=True)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/TimeSeries_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


def epochingTool(signals, epoch_length, samplingFreq, msg="", verbose=True):
    """
    Epoch length in seconds; sampling frequency in Hz
    """
    tic = time.time()
    nEpochs = math.trunc(len(signals[0]) / (epoch_length * samplingFreq))
    # Cut input signals to obtain equal sized epochs
    signalsCut = signals[:, :nEpochs * epoch_length * samplingFreq]
    epochedSignals = np.ndarray((nEpochs, len(signals), epoch_length * samplingFreq))

    if verbose:
        print("Epoching %s" % msg, end="")

    for channel in range(len(signalsCut)):
        split = np.array_split(signalsCut[channel], nEpochs)
        for i in range(len(split)):
            epochedSignals[i][channel] = split[i]

    if verbose:
        print(" - %0.3f seconds.\n" % (time.time() - tic,))

    return epochedSignals


def bandpassFIRfilter(signals, lowcut, highcut, windowtype, samplingRate, times=None, plot="OFF"):
    """
     Truncated sinc function in whatever order, needs to be windowed to enhance frequency response at side lobe and rolloff.
     Some famous windows are: bartlett, hann, hamming and blackmann.
     Two processes depending on input: epoched signals or entire signals
     http://www.labbookpages.co.uk/audio/firWindowing.html
     https://en.wikipedia.org/wiki/Window_function
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window
    """
    tic = time.time()
    try:
        signals[0][0][0]
        order = int(len(signals[0][0]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass",
                           fs=samplingRate)
        efsignals = np.ndarray((len(signals), len(signals[0]), len(signals[0][0])))
        print("Band pass filtering epoched signals: %i-%iHz " % (lowcut, highcut), end="")
        for channel in range(len(signals)):
            print(".", end="")
            for epoch in range(len(signals[0])):
                efsignals = np.ndarray((len(signals), len(signals[channel]), len(signals[channel][epoch])))
                efsignals[channel][epoch] = filtfilt(b=firCoeffs, a=[1.0], x=signals[channel][epoch],
                                                     padlen=int(2.5 * order))
                # a=[1.0] as it is FIR filter (not IIR).
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return efsignals

    except IndexError:
        order = int(len(signals[0, :]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass",
                           fs=samplingRate)
        filterSignals = filtfilt(b=firCoeffs, a=[1.0], x=signals,
                                 padlen=int(2.5 * order))  # a=[1.0] as it is FIR filter (not IIR).
        if plot == "ON":
            plt.plot(range(len(firCoeffs)), firCoeffs)  # Plot filter shape
            plt.title("FIR filter shape w/ %s windowing" % windowtype)
            for i in range(1, 10):
                plt.figure(i + 1)
                plt.xlabel("time (ms)")
                plt.plot(times, signals[i], label="Raw signal")
                plt.plot(times, filterSignals[i], label="Filtered Signal")
            plt.show()
            plt.savefig("figures/filterSample%s" % str(i))
        return filterSignals


def plotConversions(raw_signals, filterSignals, phase, amplitude_envelope, band, regionLabels=None, n_signals=1,
                    raw_time=None):
    for channel in range(n_signals):

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_scatter(x=raw_time, y=raw_signals[channel], name="Raw signal")
        fig.add_scatter(x=raw_time, y=filterSignals[channel], name="Filtered signal")
        fig.add_scatter(x=raw_time, y=phase[channel], name="Instantaneous phase", secondary_y=True)
        fig.add_scatter(x=raw_time, y=amplitude_envelope[channel], name="Amplitude envelope")

        fig.update_layout(title="%s filtered - %s signal conversions" % (band, regionLabels[channel]))
        fig.update_xaxes(title_text="Time (ms)")

        fig.update_yaxes(title_text="Amplitude", range=[-max(raw_signals[channel]), max(raw_signals[channel])],
                         secondary_y=False)
        fig.update_yaxes(title_text="Phase", tickvals=[-3.14, 0, 3.14], range=[-15, 15], secondary_y=True,
                         gridcolor='mintcream')

        pio.write_html(fig, file="figures/%s_%s_conversions.html" % (band, regionLabels[channel]), auto_open=True)

# FFT
def multitapper(signals, samplingFreq, regionLabels=None, epoch_length=4, ntapper=4, smoothing=0.5, peaks=False, plot=False, folder="figures/", title="", mode="html"):
    # Demean data
    demean_data = signals - np.average(signals, axis=1)[:, None]

    # Epoch data
    nsamples = epoch_length * 1000
    epoched_data = epochingTool(demean_data, epoch_length, 1000, verbose=False)

    # Generate the slepian tappers
    dpss = scipy.signal.windows.dpss(nsamples, nsamples * (smoothing / samplingFreq), ntapper)

    # select frequency band
    freqs = np.arange(nsamples - 1) / (nsamples / samplingFreq)
    limits = [2, 40]
    freqsindex = np.where((freqs >= limits[0]) & (freqs <= limits[1]))

    # Applies the tapper
    multitappered_data = np.asarray([filter * epoched_data for filter in dpss])

    # Fourier transform of the filtered data
    fft_multitappered_data = np.fft.fft(multitappered_data).real / np.sqrt(nsamples)

    # Power calculation
    power_data = abs(fft_multitappered_data) ** 2

    # Average FFTs by trials and tappers
    avg_power_data = np.average(power_data, (0, 1))

    # Keep selected frequencies
    ffreqs = freqs[freqsindex]
    fpower = np.squeeze(avg_power_data[:, freqsindex])

    if plot:
        fig = go.Figure(
            layout=dict(title="Brainstorm processed data - ROIs spectra (Py)", xaxis=dict(title='Frequency (Hz)'),
                        yaxis=dict(title='Power')))
        for roi in range(len(signals)):
            fig.add_trace(go.Scatter(x=ffreqs, y=fpower[roi], mode="lines", name=regionLabels[roi]))

        if mode == "html":
            pio.write_html(fig, file=folder + "BS_" + title + "_Spectra.html", auto_open=True)
        elif mode == "inline":
            plotly.offline.iplot(fig)

    if peaks:
        IAF = ffreqs[fpower.argmax(axis=1)]
        modules = [fpower[spectrum, fpower.argmax(axis=1)[spectrum]] for spectrum in range(len(fpower))]
        band_modules = [scipy.integrate.simpson(fpower[spectrum, (IAF[spectrum] - 2 < ffreqs) & (ffreqs < IAF[spectrum] + 2)]) for spectrum in range(len(fpower))]  # Alpha band integral
        return ffreqs, fpower, IAF, np.asarray(modules), np.asarray(band_modules)

    else:
        return ffreqs, fpower


def FFTarray(signals, simLength, transient, regionLabels, param1=None, param2=None, param3=None, lowcut=5, highcut=15):

    regLabs = list()
    fft_tot = list()
    freq_tot = list()
    param1array = list()
    param2array = list()
    param3array = list()

    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[i]) / 2)
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        fft = fft[(freqs > lowcut) & (freqs < highcut)]  # remove undesired frequencies
        freqs = freqs[(freqs > lowcut) & (freqs < highcut)]

        regLabs += [regionLabels[i]] * len(fft)
        fft_tot += list(fft)
        freq_tot += list(freqs)
        param1array += [param1] * len(fft)
        param2array += [param2] * len(fft)
        param3array += [param3[i, 0]] * len(fft)

    return np.asarray([param1array, param2array, regLabs, fft_tot, freq_tot, param3array], dtype=object).transpose()


def PSD(signals, samplingFreq, window=4, overlap=0.5):

    fft_result=[]
    freqs_result=[]
    window_size = int(window * samplingFreq)
    step_size = int(window_size * (1 - overlap))

    for roi, roi_signal in enumerate(signals):
        eSignal = [roi_signal[i: i + window_size] for i in range(0, len(roi_signal) - window_size, step_size)]
        fft_vector = []
        for epoch in eSignal:
            fft_temp = np.real(np.fft.fft(epoch))  # FFT for each channel signal
            fft_temp = fft_temp[range(int(len(epoch) / 2))]  # Select just positive side of the symmetric FFT
            fft_vector.append(fft_temp)

        fft_result.append(np.average(fft_vector, axis=0))
    freqs = np.arange(window_size / 2)
    freqs_result.append(freqs / window)

    return fft_result, freqs_result


def PSDplot(signals, samplingFreq, regionLabels, folder="figures", title=None, mode="html", max_hz=80, min_hz=1,
            type="log", window=4, overlap=0.5):

    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='Frequency', type=type), yaxis=dict(title='Log power (dB)', type=type)))

    window_size = int(window * samplingFreq)
    step_size = int(window_size * (1 - overlap))

    for roi, roi_signal in enumerate(signals):
        eSignal = [roi_signal[i: i + window_size] for i in range(0, len(roi_signal) - window_size, step_size)]
        fft_vector = []
        for epoch in eSignal:
            fft_temp = abs(np.fft.fft(epoch))  # FFT for each channel signal
            fft_temp = fft_temp[range(int(len(epoch) / 2))]  # Select just positive side of the symmetric FFT
            fft_vector.append(fft_temp)

        fft = np.average(fft_vector, axis=0)
        freqs = np.arange(window_size / 2)
        freqs = freqs / window

        cut_high = np.where(freqs >= max_hz)[0][0]  # Hz. Number of frequency points until cut at xHz point
        cut_low = np.where(freqs >= min_hz)[0][0]
        fig.add_scatter(x=freqs[int(cut_low):int(cut_high)], y=fft[int(cut_low):int(cut_high)], name=regionLabels[roi])

    if mode == "html":
        pio.write_html(fig, file=folder + "/FFT_" + title + ".html", auto_open=True)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/FFT_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


def FFTplot(signals, simLength, regionLabels, folder="figures", title=None, mode="html", max_hz=80, min_hz=1,
            type="log"):

    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='Frequency', type=type), yaxis=dict(title='Module', type=type)))

    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[0]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / (simLength / 1000)  # simLength (ms) / 1000 -> segs

        cut_high = (simLength / 2) / 500 * max_hz  # Hz. Number of frequency points until cut at xHz point
        cut_low = (simLength / 2) / 500 * min_hz
        fig.add_scatter(x=freqs[int(cut_low):int(cut_high)], y=fft[int(cut_low):int(cut_high)], name=regionLabels[i])

    if mode == "html":
        pio.write_html(fig, file=folder + "/FFT_" + title + ".html", auto_open=True)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/FFT_" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "inline":
        plotly.offline.iplot(fig)


def FFTpeaks(signals, simLength):
    if signals.ndim != 2:
        print("Array should be an array with shape: channels x timepoints.")

    else:
        peaks = list()
        modules = list()
        band_modules = list()
        for i in range(len(signals)):
            fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
            fft = fft[range(int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(signals[i]) / 2)
            freqs = freqs / (simLength / 1000)  # simLength (ms) / 1000 -> segs

            fft = fft[freqs > 0.5]  # remove undesired frequencies from peak analisis
            freqs = freqs[freqs > 0.5]

            IAF = freqs[np.where(fft == max(fft))][0]
            peaks.append(IAF)
            modules.append(fft[np.where(fft == max(fft))][0])
            band_modules.append(scipy.integrate.simpson(fft[(IAF-2 < freqs) & (freqs < IAF+2)]))  # Alpha band integral

    return np.asarray(peaks), np.asarray(modules), np.asarray(band_modules)


# FC
def CORR(signals, regionLabels, plot="OFF"):
    """
    To compute correlation between signals you need to standarize signal values and then to sum up intersignal products
    divided by signal length.
    """

    normalSignals = np.ndarray((len(signals), len(signals[0])))
    for channel in range(len(signals)):
        mean = np.mean(signals[channel, :])
        std = np.std(signals[channel, :])
        normalSignals[channel] = (signals[channel, :] - mean) / std

    CORR = np.ndarray((len(normalSignals), len(normalSignals)))
    for channel1 in range(len(normalSignals)):
        for channel2 in range(len(normalSignals)):
            CORR[channel1][channel2] = sum(normalSignals[channel1] * normalSignals[channel2]) / len(normalSignals[0])

    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=CORR, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Correlation')
        pio.write_html(fig, file="figures/CORR.html", auto_open=True)

    return CORR


def PLV(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False, verbose=True):
    tic = time.time()
    try:
        efPhase[0][0][0]  # Test whether signals have been epoched
        PLV = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        if verbose:
            print("Calculating PLV", end="")
        for channel1 in range(len(efPhase[0])):
            if verbose:
                print(".", end="")
            for channel2 in range(len(efPhase[0])):
                plv_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]
                    value = abs(np.average(np.exp(1j * phaseDifference)))
                    plv_values.append(value)
                PLV[channel1, channel2] = np.average(plv_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Locking Value')
            pio.write_html(fig, file=folder + "/PLV_" + subject + ".html", auto_open=auto_open)
        if verbose:
            print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLV

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def PLE(efPhase, time_lag, pattern_size, samplingFreq, subsampling=1):
    """
    It calculates Phase Lag Entropy on a bunch of filtered and epoched signals with shape [epoch,rois,time]
    It is based on the diversity of temporal patterns between two signals phases.

    A pattern S(t) is defined as:
        S(t)={s(t), s(t+tau), s(t+2tau),..., s(t + m*tau -1)} where
                (s(t)=1 if delta(Phi)>0) & (s(t)=0 if delta(Phi)<0)

    PLE = - sum( p_j * log(p_j) ) / log(2^m)

        where
        - p_j is the probability of the jth pattern, estimated counting the number of times each pattern
        occurs in a given epoch and
        - m is pattern size.

    :param efPhase: Phase component of Hilbert transform (filtered in specific band and epoched)
    :param time_lag: "tau" temporal distance between elements in pattern
    :param pattern_size: "m" number of elements to consider in each pattern
    :param samplingFreq: signal sampling frequency
    :param subsampling: If your signal has high temporal resolution, maybe gathering all possible patters is not
     efficient, thus you can omit some timepoints between gathered patterns

    :return: PLE - matrix shape (rois, rois) with PLE values for each couple; patts - patterns i.e. all the patterns
    registered in the signal and the number of times they appeared.
    """

    tic = time.time()
    try:
        efPhase[0][0][0]  # Test whether signals have been epoched
        PLE = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        time_lag = int(np.trunc(time_lag * samplingFreq / 1000))  # translate time lag in timepoints
        patts=list()
        print("Calculating PLE ", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                ple_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]

                    # Binarize to create the pattern and comprehend into list
                    patterns = [str(np.where(phaseDifference[t: t + time_lag * pattern_size: time_lag] > 0, 1, 0)) for t in
                                np.arange(0, len(phaseDifference) - time_lag * pattern_size, step=subsampling)]
                    patt_counts = Counter(patterns)
                    patts.append(patt_counts)
                    summation = 0
                    for key, value in patt_counts.items():
                        p = value / len(patterns)
                        summation += p * np.log10(p)

                    ple_values.append((-1 / np.log10(2 ** pattern_size)) * summation)
                PLE[channel1, channel2] = np.average(ple_values)
        print("%0.3f seconds.\n" % (time.time() - tic,))

        return PLE, patts

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def dPLV(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    ws = (len(efPhase[0][0]) - 500) / 500  # number of temporal windows per epoch
    dPLV = np.ndarray((np.int(len(efPhase) * ws), len(efPhase[0]), len(efPhase[0])))

    for e in range(len(efPhase)):
        for ii, t in enumerate(range(500, len(efPhase[e][0]), 500)):
            print("For time %0.2f ms:" % t)
            dPLV[np.int(e * ws + ii)] = PLV(np.array([efPhase[e][:, t - 500:t + 500]]))

    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Phase Locking Value')
        pio.write_html(fig, file=folder + "/PLV_" + subject + ".html", auto_open=auto_open)
    print("%0.3f seconds.\n" % (time.time() - tic,))

    return dPLV


def PLI(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efPhase[0][0][0]
        PLI = np.ndarray(((len(efPhase[0])), len(efPhase[0])))

        print("Calculating PLI", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                pli_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]
                    value = np.abs(np.average(np.sign(np.sin(phaseDifference))))
                    pli_values.append(value)
                PLI[channel1, channel2] = np.average(pli_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLI, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Lag Index')
            pio.write_html(fig, file=folder + "/PLI_" + subject + ".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLI

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def AEC(efEnvelope, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efEnvelope[0][0][0]
        AEC = np.ndarray(((len(efEnvelope[0])), len(efEnvelope[0])))  # averaged AECs per channel x channel

        print("Calculating AEC", end="")
        for channel1 in range(len(efEnvelope[0])):
            print(".", end="")
            for channel2 in range(len(efEnvelope[0])):
                values_aec = list()  # AEC per epoch and channel x channel
                for epoch in range(len(efEnvelope)):  # CORR between channels by epoch
                    r = np.corrcoef(efEnvelope[epoch][channel1], efEnvelope[epoch][channel2])
                    values_aec.append(r[0, 1])
                AEC[channel1, channel2] = np.average(values_aec)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=AEC, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Amplitude Envelope Correlation')
            pio.write_html(fig, file=folder + "/AEC_" + subject + ".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return AEC

    except IndexError:
        print("IndexError. Signals must be epoched before calculating AEC. Use epochingTool().")


# ParamSpace
def paramSpace(df, z=None, title=None, names=None, folder="figures", auto_open="True", show_owp=False):
    if any(measure in title for measure in ["AEC", "PLI", "PLV"]):
        fig = make_subplots(rows=1, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            specs=[[{}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                            x_title="Conduction speed (m/s)", y_title="Coupling factor")

        fig.add_trace(go.Heatmap(z=df.Delta, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                                 reversescale=True, zmin=-z, zmax=z), row=1, col=1)

        fig.add_trace(go.Heatmap(z=df.Theta, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                                  showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=df.Alpha, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                                  showscale=False), row=1, col=3)
        fig.add_trace(go.Heatmap(z=df.Beta, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                                  showscale=False), row=1, col=4)
        fig.add_trace(go.Heatmap(z=df.Gamma, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                                  showscale=False), row=1, col=5)

        if show_owp:
            owp_g = df['avg'].idxmax()[0]
            owp_s = df['avg'].idxmax()[1]

            [fig.add_trace(
                go.Scatter(
                    x=[owp_s], y=[owp_g], mode='markers',
                    marker=dict(color='black', line_width=2, size=[10], opacity=0.5, symbol='circle-open', showscale=False)),
                row=1, col=j) for j in range(1, 6)]

        fig.update_layout(
            title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
        pio.write_html(fig, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)

    elif title == "significance":
        fig = make_subplots(rows=1, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            specs=[[{}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                            x_title="Conduction speed (m/s)", y_title="Coupling factor")

        fig.add_trace(
            go.Heatmap(z=df.Delta_sig, x=df.speed, y=df.G, colorscale='RdBu', colorbar=dict(title="Pearson's r p-value"),
                       reversescale=True, zmin=0, zmax=z), row=1, col=1)
        fig.add_trace(
            go.Heatmap(z=df.Theta_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                       showscale=False), row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=df.Alpha_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                       showscale=False), row=1, col=3)
        fig.add_trace(
            go.Heatmap(z=df.Beta_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                       showscale=False), row=1, col=4)
        fig.add_trace(
            go.Heatmap(z=df.Gamma_sig, x=df.speed, y=df.G, colorscale='RdBu', reversescale=True, zmin=0, zmax=z,
                       showscale=False), row=1, col=5)

        fig.update_layout(
            title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
        pio.write_html(fig, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)

    elif title == "FC_comparisons":
        bands = ["2-theta", "3-alfa", "4-beta", "5-gamma"]  # omiting first band in purpose
        fig = make_subplots(rows=2, cols=5, subplot_titles=(
        "Delta", "Theta", "Alpha", "Beta", "Gamma", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            row_titles=("AEC", "PLV"), shared_yaxes=True, shared_xaxes=True)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].r,
                                 x=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].frm,
                                 y=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].to,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                 zmin=min(df.r), zmax=1), row=1, col=1)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].r,
                                 x=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].frm,
                                 y=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].to,
                                 colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=1)
        for i, b in enumerate(bands):
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == b)].r,
                                     x=df[(df["FC_measure"] == "aec") & (df["band"] == b)].frm,
                                     y=df[(df["FC_measure"] == "aec") & (df["band"] == b)].to,
                                     colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=1, col=i + 2)
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == b)].r,
                                     x=df[(df["FC_measure"] == "plv") & (df["band"] == b)].frm,
                                     y=df[(df["FC_measure"] == "plv") & (df["band"] == b)].to,
                                     colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=i + 2)
        fig.update_layout(title_text="FC correlation between subjects")
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif title == "sf_comparisons":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PLV", "AEC"),
                            shared_yaxes=True, shared_xaxes=True,
                            x_title="Frequency bands")

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"] == "aec"].r,
                                 x=df[df["FC_measure"] == "aec"].band,
                                 y=df[df["FC_measure"] == "aec"].subj,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                 zmin=min(df.r), zmax=max(df.r)), row=1, col=1)

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"] == "plv"].r,
                                 x=df[df["FC_measure"] == "plv"].band,
                                 y=df[df["FC_measure"] == "plv"].subj,
                                 colorscale='Viridis', showscale=False,
                                 zmin=min(df.r), zmax=max(df.r)), row=1, col=2)
        fig.update_layout(
            title_text='Structural - Functional correlations')
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif title == "interW":
        fig = ff.create_annotated_heatmap(df, names, names, colorscale="Viridis", showscale=True,
                                          colorbar=dict(title="Pearson's r"))
        fig.update_layout(title_text="Correlations in sctructural connectivity between real subjects")
        pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

    elif "fft-bm" in title:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))
        fig.add_trace(
            go.Scatter(text=np.round(df.mS_bm, 2), x=df.speed, y=df.G, mode="text", textfont=dict(color="crimson")))
        fig.update_layout(
            title_text="Heatmap for simulation's mean signal FFT peak; in red Hartigans' bimodality test's p-value. (0 -> p<2.2e-16)")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder + "/paramSpace-%s.html" % title, auto_open=auto_open)

    elif "regional" in title:
        fig = go.Figure()
        # for regional plots

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))

        fig.update_layout(
            title_text="FFT peak of simulated signals by Coupling factor and Conduction speed")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder + "/paramSpace-FFTpeak_%s.html" % title, auto_open=auto_open)


# def emp_sim():
#     fig = make_subplots(rows=2, cols=5, subplot_titles=(
#     "Delta", "Theta", "Alpha", "Beta", "Gamma", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
#                         row_titles=("AEC", "PLV"), shared_yaxes=True, shared_xaxes=True)
#     fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].r,
#                              x=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].frm,
#                              y=df[(df["FC_measure"] == "aec") & (df["band"] == "1-delta")].to,
#                              colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
#                              zmin=min(df.r), zmax=1), row=1, col=1)
#     fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].r,
#                              x=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].frm,
#                              y=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].to,
#                              colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=1)
#     for i, b in enumerate(bands):
#         fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "aec") & (df["band"] == b)].r,
#                                  x=df[(df["FC_measure"] == "aec") & (df["band"] == b)].frm,
#                                  y=df[(df["FC_measure"] == "aec") & (df["band"] == b)].to,
#                                  colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=1, col=i + 2)
#         fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == b)].r,
#                                  x=df[(df["FC_measure"] == "plv") & (df["band"] == b)].frm,
#                                  y=df[(df["FC_measure"] == "plv") & (df["band"] == b)].to,
#                                  colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=i + 2)
#     fig.update_layout(title_text="FC correlation between subjects")
#     pio.write_html(fig, file=folder + "/%s.html" % title, auto_open=auto_open)

# Stimulation

# Stimulacion
def stimulation_fft(df, folder, title=None, auto_open="True"):
    fig = make_subplots(rows=2, cols=1, subplot_titles=(["FFT peak", "Average activity amplitude"]),
                        shared_yaxes=False, shared_xaxes=False)

    fig.add_trace(go.Heatmap(z=df.tFFT_peak, x=df.stimFreq, y=df.stimAmplitude,
                             colorscale='Viridis',
                             colorbar=dict(title="Hz", thickness=20, y=0.82, ypad=120),
                             zmin=np.min(df.tFFT_peak), zmax=np.max(df.tFFT_peak)), row=1, col=1)

    fig.add_trace(go.Heatmap(z=df.tAvg_activity, x=df.stimFreq, y=df.stimAmplitude,
                             colorscale='Inferno',
                             colorbar=dict(title="mV", thickness=20, y=0.2, ypad=120),
                             zmin=np.min(df.tAvg_activity), zmax=np.max(df.tAvg_activity)), row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="stimulation frequency", row=1, col=1)
    fig.update_xaxes(title_text="stimulation frequency", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="stimulus amplitude", row=1, col=1)
    fig.update_yaxes(title_text="stimulus amplitude", row=2, col=1)

    fig.update_layout(title_text=title)
    pio.write_html(fig, file=folder + "/" + title + ".html", auto_open=auto_open)


def stimulation_fc(df, structure, regionLabels, folder, region="Left-Cuneus", title="w-conn", t_c="target",
                   auto_open="True"):
    if "FFT" not in region:

        sp_titles = ["Stimulation Weight = " + str(ws) for ws in set(df.stimAmplitude)]

        fig = make_subplots(rows=len(set(df.stimAmplitude)) + 1, cols=1,
                            subplot_titles=(sp_titles + ["Structural Connectivity"]),
                            shared_yaxes=True, shared_xaxes=True, y_title="Stimulation Frequency")

        for i, ws in enumerate(set(df.stimAmplitude)):
            subset = df[df["stimAmplitude"] == ws]

            if i == 0:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               colorbar=dict(title="PLV"), zmin=0, zmax=1), row=i + 1, col=1)

            else:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               zmin=0, zmax=1, showscale=False), row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=regionLabels, y=structure), row=i + 2, col=1)

        fig.update_layout(
            title_text='FC of simulated signals by stimulation frequency and weight || ' + t_c + ' region: ' + region,
            template="simple_white")
        pio.write_html(fig, file=folder + "/stimSpace-f&a%s_" % t_c + title + ".html", auto_open=auto_open)

    else:

        sp_titles = ["Stimulation Weight = " + str(ws) for ws in set(df.stimAmplitude)]

        fig = make_subplots(rows=len(set(df.stimAmplitude)) + 1, cols=1,
                            subplot_titles=(sp_titles + ["Structural Connectivity"]),
                            shared_yaxes=True, shared_xaxes=True, y_title="Stimulation Frequency")

        for i, ws in enumerate(set(df.stimAmplitude)):
            subset = df[df["stimAmplitude"] == ws]

            if i == 0:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               colorbar=dict(title="PLV")), row=i + 1, col=1)

            else:
                fig.add_trace(
                    go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                               zmin=0, zmax=1, showscale=False), row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=regionLabels, y=structure), row=i + 2, col=1)

        fig.update_layout(
            title_text='FC of simulated signals by stimulation frequency and weight || ' + t_c + ' region: ' + region,
            template="simple_white")
        pio.write_html(fig, file=folder + "/stimSpace-f&a%s_FFTpeaks_" % t_c + title + ".html", auto_open=auto_open)


# stimWfit
def collectData_stimWfit(specific_folder, emp_subj=None, write_csv=True):

    if emp_subj:
        files = glob.glob1(specific_folder, emp_subj+'*')
    else:
        files = os.listdir(specific_folder)

    df_fft = pd.DataFrame()
    df_ar = pd.DataFrame()

    for file in files:

        if 'alphaRise' in file:
            df_ar = df_ar.append(pd.read_csv(specific_folder + '\\' + file), ignore_index=True)

        elif 'FFT' in file:
            df_fft = df_fft.append(pd.read_csv(specific_folder + '\\' + file), ignore_index=True)



    if write_csv:
        df_fft.to_csv(specific_folder + '\\' + 'FFT_fullDF.csv')
        df_ar.to_csv(specific_folder + '\\' + 'alphaRise_fullDF.csv')

    return df_fft, df_ar


def boxPlot_stimWfit(df_ar, emp_subj, specific_folder,  n_simulations):

    # calculate percentages
    df_ar_avg = df_ar.groupby('w').mean()

    df_ar["percent"] = [
        ((df_ar.peak_module[i] - df_ar_avg.peak_module[0]) / df_ar_avg.peak_module[0]) * 100 for i in
        range(len(df_ar))]

    df_ar_avg = df_ar.groupby('w').mean()
    df_ar_avg["sd"] = df_ar.groupby('w')[['w', 'peak_module']].std()


    fig = px.box(df_ar, x="w", y="peak_module",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "peak_module": "Alpha peak module"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_modules_" + str(n_simulations) + "sim.html",
                   auto_open=False)

    fig = px.box(df_ar, x="w", y="percent",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "percent": "Percentage of alpha peak rise"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_percent_" + str(n_simulations) + "sim.html",
                   auto_open=True)


def lines3dFFT_stimWfit(df_fft, specific_folder, show_rois=False):

    rois_ = list(set(df_fft.regLab))
    rois_.sort()
    rois=rois_[0:len(rois_):2]+rois_[1:len(rois_):2]
    weights = np.sort(np.array(list(set(df_fft.w))))
    reps = list(set(df_fft.rep))
    initPeak=np.average(df_fft.initPeak)

    fig_global = make_subplots(rows=2, cols=5, vertical_spacing=0.1, horizontal_spacing=0.001,
                               subplot_titles=(
                               "Alpha rise <br>@Parietal complex + precuneus", rois[0],
                               rois[1], rois[2], rois[3], rois[4], rois[5]),
                               specs=[[{"rowspan": 2, "colspan": 2, 'type': 'surface'}, None, {'type': 'surface'},
                                       {'type': 'surface'},{'type': 'surface'}],
                                      [None, None, {'type': 'surface'}, {'type': 'surface'},{'type': 'surface'}]])

    pos = [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]

    # Plots per ROI
    for i, roi in enumerate(rois[0:5]):
        dft_roi=df_fft.loc[df_fft["regLab"] == roi]

        weights_subset = np.sort(np.random.choice(weights[1:], 4, replace=False))
        fig = make_subplots(rows=2, cols=4, vertical_spacing=0.1, horizontal_spacing=0.001,
                            subplot_titles=(roi + "<br> 40 sim averaged", "w = "+str(weights_subset[0]),
                                            "w = "+str(weights_subset[1]),"w = "+str(weights_subset[2]),"w = "+str(weights_subset[3])),
                            specs=[[{"rowspan": 2, "colspan": 2, 'type': 'surface'}, None, {'type': 'surface'}, {'type': 'surface'}],
                                   [None, None, {'type': 'surface'}, {'type': 'surface'}]])

        (row, col)=pos[i]
        dft_roi_repavg = dft_roi.groupby(['w', 'freq'])[['w', 'freq', 'rep', 'fft_module', 'initPeak']].mean()
        for w_ in np.sort(np.array(list(set(dft_roi_repavg.w)))):
            dft_roi_repavg_w = dft_roi_repavg.loc[dft_roi_repavg["w"] == w_]

            # GLOBAL
            fig_global.add_trace(go.Scatter3d(x=dft_roi_repavg_w.w, y=dft_roi_repavg_w.freq, z=dft_roi_repavg_w.fft_module,
                                              legendgroup="w = " + str(w_), showlegend=False, mode="lines", line=dict(width=2.5, color="gray")), row=row, col=col)
            fig_global.add_trace(go.Scatter3d(x=np.array([max(dft_roi_repavg_w.w)]),
                                       y=np.array([initPeak]),
                                       z=np.array([max(dft_roi_repavg_w.fft_module)]), legendgroup="w = " + str(w_), showlegend=False,
                                       marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=row, col=col)
            # ROIS
            fig.add_trace(go.Scatter3d(x=dft_roi_repavg_w.w, y=dft_roi_repavg_w.freq, z=dft_roi_repavg_w.fft_module, legendgroup="w = " + str(w_), name="w = " + str(round(w_, 4)), mode="lines", line=dict(width=4)), row=1, col=1)
            fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_repavg_w.w)]),
                                       y=np.array([initPeak]),
                                       z=np.array([max(dft_roi_repavg_w.fft_module)]), legendgroup="w = " + str(w_), showlegend=False,
                                       marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=1)

        del dft_roi_repavg_w, dft_roi_repavg

        for j, w_ in enumerate(weights_subset[:2]):
            dft_roi_w = dft_roi.loc[dft_roi["w"] == w_]
            for r_ in reps:
                dft_roi_w_r = dft_roi_w.loc[dft_roi["rep"] == r_]
                if j==0:
                    fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), name="rep = " + str(r_), mode="lines", line=dict(width=2.5, color="gray")), row=1, col=3+j)
                else:
                    fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), showlegend=False, mode="lines", line=dict(width=2.5, color="gray")), row=1, col=3+j)
                fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_w_r.rep)]),
                                           y=np.array([initPeak]),
                                           z=np.array([max(dft_roi_w_r.fft_module)]), legendgroup="rep = " + str(r_), showlegend=False, marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=3+j)

        for j, w_ in enumerate(weights_subset[2:]):
            dft_roi_w = dft_roi.loc[dft_roi["w"] == w_]
            for r_ in reps:
                dft_roi_w_r = dft_roi_w.loc[dft_roi["rep"] == r_]
                fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), showlegend=False, name="rep = " + str(r_), mode="lines", line=dict(width=2, color="gray")), row=2, col=3+j)
                fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_w_r.rep)]),
                                       y=np.array([initPeak]),
                                       z=np.array([max(dft_roi_w_r.fft_module)]), legendgroup="rep = " + str(r_), showlegend=False, marker=dict(symbol='cross', size=5, color="black", opacity=0.5)), row=2, col=3+j)

        del dft_roi_w, dft_roi_w_r, dft_roi

        fig.update_layout(legend=dict(y=1.2, x=-0.1),
            scene1=dict(xaxis_title='Stim Weight', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
            scene2=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
            scene3=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
            scene4=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
            scene5=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'))

        pio.write_html(fig, file=specific_folder + "/lines3dFFT-%s.html" % roi, auto_open=show_rois)


    # Global plot
    dft_roiavg = df_fft.groupby(['w', 'freq'])[['w', 'freq', 'rep', 'fft_module', 'initPeak']].mean()
    for iii, w_ in enumerate(np.sort(np.array(list(set(dft_roiavg.w))))):

        dft_roiavg_w = dft_roiavg.loc[dft_roiavg["w"]==w_]

        fig_global.add_trace(
            go.Scatter3d(x=dft_roiavg_w.w, y=dft_roiavg_w.freq, z=dft_roiavg_w.fft_module, legendgroup="w = " + str(w_), name="w = " + str(round(w_, 4)), mode="lines", line=dict(width=4)), row=1, col=1)

        fig_global.add_trace(go.Scatter3d(x=np.array([max(dft_roiavg_w.w)]),
                                          y=np.array([initPeak]),
                                          z=np.array([max(dft_roiavg_w.fft_module)]),
                                          legendgroup="w = " + str(w_), showlegend=False,
                                          marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=1)
    del dft_roiavg_w, dft_roiavg
    fig_global.update_layout(legend=dict(y=1.2, x=-0.1),
        scene1=dict(xaxis_title='Stim Weight', yaxis_title='Frequency (Hz)', zaxis_title='Module'))

    pio.write_html(fig_global, file=specific_folder + "/global_lines3dFFT-ParietalComplex.html", auto_open=True)
