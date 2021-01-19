# This is going to be a script with useful functions I will be using frequently.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, hilbert
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import math
import time


def timeseriesPlot(signals, timepoints, regionLabels, folder="figures", title=None, mode="html"):
    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='time (ms)'), yaxis=dict(title='Voltage')))
    for ch in range(len(signals)):
        fig.add_scatter(x=timepoints, y=signals[ch], name=regionLabels[ch])

    if mode=="html":
        pio.write_html(fig, file=folder+"/TimeSeries.html", auto_open=True)
    elif mode=="png":
        pio.write_image(fig, file=folder+"/TimeSeries_"+str(time.time())+".png", engine="kaleido")
    elif mode=="inline":
        plotly.offline.iplot(fig)

def epochingTool(signals, epoch_length, samplingFreq, msg=""):
    """
    Epoch length in seconds; sampling frequency in Hz
    """
    tic = time.time()
    nEpochs=math.trunc(len(signals[0])/(epoch_length*samplingFreq))
    # Cut input signals to obtain equal sized epochs
    signalsCut=signals[:, :nEpochs*epoch_length*samplingFreq]
    epochedSignals=np.ndarray((nEpochs,len(signals),epoch_length*samplingFreq))
    print("Epoching %s" % msg, end="")
    for channel in range(len(signalsCut)):
        split = np.array_split(signalsCut[channel], nEpochs)
        for i in range(len(split)):
            epochedSignals[i][channel] = split[i]

    print(" - %0.3f seconds.\n" % (time.time() - tic,))
    return epochedSignals


def FFTplot(signals, simLength, regionLabels, folder="figures", title=None, mode="html"):
    fig = go.Figure(layout=dict(title=title, xaxis=dict(title='Frequency'), yaxis=dict(title='Module')))
    for i in range(len(signals)):
        fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
        fft = fft[range(int(len(signals[0]) / 2))]  # Select just positive side of the symmetric FFT
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / (simLength/1000)  # simLength (ms) / 1000 -> segs

        cut = (simLength/2) / 500 * 80#Hz. Number of frequency points until cut at xHz point

        fig.add_scatter(x=freqs[:int(cut)], y=fft[:int(cut)], name=regionLabels[i], )

    if mode=="html":
        pio.write_html(fig, file=folder+"/FFT.html", auto_open=True)
    elif mode=="png":
        pio.write_image(fig, file=folder+"/FFT_"+str(time.time())+".png", engine="kaleido")
    elif mode=="inline":
        plotly.offline.iplot(fig)

def FFTpeaks(signals, simLength):

    if signals.ndim!=2:
        print("Array should be an array with shape: channels x timepoints.")

    else:
        peaks = list()
        modules = list()
        for i in range(len(signals)):
            fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
            fft = fft[range(np.int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(signals[i]) / 2)
            freqs = freqs / (simLength/1000)  # simLength (ms) / 1000 -> segs

            fft=fft[freqs>0.5] # remove undesired frequencies from peak analisis
            freqs=freqs[freqs>0.5]

            modules.append(fft[np.where(fft == max(fft))])
            peaks.append(freqs[np.where(fft == max(fft))])

    return np.asarray(peaks), np.asarray(modules)


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
        efsignals=np.ndarray((len(signals),len(signals[0]), len(signals[0][0])))
        print("Band pass filtering epoched signals: %i-%iHz " % (lowcut, highcut), end="")
        for channel in range(len(signals)):
            print(".", end="")
            for epoch in range(len(signals[0])):
                efsignals = np.ndarray((len(signals), len(signals[channel]), len(signals[channel][epoch])))
                efsignals[channel][epoch] = filtfilt(b=firCoeffs, a=[1.0], x=signals[channel][epoch],padlen=int(2.5 * order))
                # a=[1.0] as it is FIR filter (not IIR).
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return efsignals

    except IndexError:
        order = int(len(signals[0, :]) / 3)
        firCoeffs = firwin(numtaps=order + 1, cutoff=[lowcut, highcut], window=windowtype, pass_zero="bandpass", fs=samplingRate)
        filterSignals = filtfilt(b=firCoeffs, a=[1.0], x=signals, padlen=int(2.5 * order)) # a=[1.0] as it is FIR filter (not IIR).
        if plot == "ON":
            plt.plot(range(len(firCoeffs)), firCoeffs) # Plot filter shape
            plt.title("FIR filter shape w/ %s windowing" % windowtype)
            for i in range(1, 10):
                plt.figure(i+1)
                plt.xlabel("time (ms)")
                plt.plot(times, signals[i], label="Raw signal")
                plt.plot(times, filterSignals[i], label="Filtered Signal")
            plt.show()
            plt.savefig("figures/filterSample%s" % str(i))
        return filterSignals

def plotConversions(raw_signals, filterSignals, phase, amplitude_envelope, regionLabels=None, n_signals=1, raw_time=None):

    for channel in range(n_signals):
        for i in range(len(phase[channel])):
            if phase[channel, i] > np.pi:
                phase[channel, i:] = phase[channel, i:] - 2 * np.pi

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_scatter(x=raw_time, y=raw_signals[channel], name="Raw signal")
        fig.add_scatter(x=raw_time, y=filterSignals[channel], name="Filtered signal")
        fig.add_scatter(x=raw_time, y=phase[channel], name="Instantaneous phase", secondary_y=True)
        fig.add_scatter(x=raw_time, y=amplitude_envelope[channel], name="Amplitude envelope")

        fig.update_layout(title="%s channel conversions" % regionLabels[channel])
        fig.update_xaxes(title_text="Time (ms)")

        fig.update_yaxes(title_text="Amplitude", range=[-max(raw_signals[channel]),max(raw_signals[channel])], secondary_y=False)
        fig.update_yaxes(title_text="Phase", tickvals=[-3.14, 0, 3.14], range=[-15, 15], secondary_y=True, gridcolor='mintcream')

        pio.write_html(fig, file="figures/%s_channel_conversions.html" % regionLabels[channel], auto_open=True)


def CORR(signals, regionLabels, plot="OFF"):
    """
    To compute correlation between signals you need to standarize signal values and then to sum up intersignal products
    divided by signal length.
    """

    normalSignals = np.ndarray((len(signals), len(signals[0])))
    for channel in range(len(signals)):
        mean = np.mean(signals[channel,:])
        std = np.std(signals[channel,:])
        normalSignals[channel] = (signals[channel,:] - mean) / std

    CORR = np.ndarray((len(normalSignals), len(normalSignals)))
    for channel1 in range(len(normalSignals)):
        for channel2 in range(len(normalSignals)):
            CORR[channel1][channel2] = sum(normalSignals[channel1] * normalSignals[channel2]) / len(normalSignals[0])


    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=CORR, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Correlation')
        pio.write_html(fig, file="figures/CORR.html", auto_open=True)

    return CORR

def PLV(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efPhase[0][0][0] # Test whether signals have been epoched
        PLV = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        print("Calculating PLV", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                plv_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]
                    value = abs(np.average(np.exp(1j*phaseDifference)))
                    plv_values.append(value)
                PLV[channel1, channel2] = np.average(plv_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Locking Value')
            pio.write_html(fig, file=folder+"/PLV_"+subject+".html", auto_open=auto_open)
        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLV

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


def dPLV(efPhase, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):

    tic = time.time()
    ws=(len(efPhase[0][0])-500)/500 # number of temporal windows per epoch
    dPLV = np.ndarray((np.int(len(efPhase)*ws), len(efPhase[0]), len(efPhase[0])))

    for e in range(len(efPhase)):
        for ii, t in enumerate(range(500,len(efPhase[e][0]),500)):
            print("For time %0.2f ms:" % t)
            dPLV[np.int(e*ws+ii)]=PLV(np.array([efPhase[e][:, t-500:t+500]]))

    if plot == "ON":
        fig = go.Figure(data=go.Heatmap(z=PLV, x=regionLabels, y=regionLabels, colorscale='Viridis'))
        fig.update_layout(title='Phase Locking Value')
        pio.write_html(fig, file=folder+"/PLV_"+subject+".html", auto_open=auto_open)
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
                    value=np.abs(np.average(np.sign(np.sin(phaseDifference))))
                    pli_values.append(value)
                PLI[channel1, channel2] = np.average(pli_values)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=PLI, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Phase Lag Index')
            pio.write_html(fig, file=folder+"/PLI_"+subject+".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return PLI

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")

def AEC(efEnvelope, regionLabels=None, folder=None, subject="", plot="OFF", auto_open=False):
    tic = time.time()
    try:
        efEnvelope[0][0][0]
        AEC = np.ndarray(((len(efEnvelope[0])), len(efEnvelope[0]))) # averaged AECs per channel x channel

        print("Calculating AEC", end="")
        for channel1 in range(len(efEnvelope[0])):
            print(".", end="")
            for channel2 in range(len(efEnvelope[0])):
                values_aec = list()  # AEC per epoch and channel x channel
                for epoch in range(len(efEnvelope)): # CORR between channels by epoch
                    r = np.corrcoef(efEnvelope[epoch][channel1], efEnvelope[epoch][channel2])
                    values_aec.append(r[0,1])
                AEC[channel1, channel2] = np.average(values_aec)

        if plot == "ON":
            fig = go.Figure(data=go.Heatmap(z=AEC, x=regionLabels, y=regionLabels, colorscale='Viridis'))
            fig.update_layout(title='Amplitude Envelope Correlation')
            pio.write_html(fig, file=folder+"/AEC_"+subject+".html", auto_open=auto_open)

        print("%0.3f seconds.\n" % (time.time() - tic,))
        return AEC

    except IndexError:
        print("IndexError. Signals must be epoched before calculating AEC. Use epochingTool().")


def paramSpace(df, z=None, title=None, names=None, folder="figures", auto_open="True"):
    if any(measure in title for measure in ["AEC","PLI","PLV"]):
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

        fig.update_layout(
            title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % title)
        pio.write_html(fig, file=folder+"/paramSpace-g&s_%s.html" % title, auto_open=auto_open)

    elif title=="FC_comparisons":
        bands=["2-theta","3-alfa","4-beta","5-gamma"] # omiting first band in purpose
        fig = make_subplots(rows=2, cols=5, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
                            row_titles=("AEC", "PLV"), shared_yaxes=True, shared_xaxes=True)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"]=="aec") & (df["band"]=="1-delta")].r,
                                 x=df[(df["FC_measure"]=="aec") & (df["band"]=="1-delta")].frm,
                                 y=df[(df["FC_measure"]=="aec") & (df["band"]=="1-delta")].to,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                  zmin=min(df.r), zmax=1), row=1, col=1)
        fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].r,
                                 x=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].frm,
                                 y=df[(df["FC_measure"] == "plv") & (df["band"] == "1-delta")].to,
                                 colorscale='Viridis', showscale=False, zmin=min(df.r), zmax=1), row=2, col=1)
        for i, b in enumerate(bands):
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"]=="aec") & (df["band"]==b)].r,
                                 x=df[(df["FC_measure"]=="aec") & (df["band"]==b)].frm,
                                 y=df[(df["FC_measure"]=="aec") & (df["band"]==b)].to,
                                 colorscale='Viridis',showscale=False,  zmin=min(df.r), zmax=1), row=1, col=i+2)
            fig.add_trace(go.Heatmap(z=df[(df["FC_measure"] == "plv") & (df["band"] == b)].r,
                                 x=df[(df["FC_measure"] == "plv") & (df["band"] == b)].frm,
                                 y=df[(df["FC_measure"] == "plv") & (df["band"] == b)].to,
                                 colorscale='Viridis',showscale=False,  zmin=min(df.r), zmax=1), row=2, col=i+2)
        fig.update_layout(title_text="FC correlation between subjects")
        pio.write_html(fig, file=folder+"/%s.html" % title, auto_open=auto_open)

    elif title=="sf_comparisons":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PLV", "AEC"),
                            shared_yaxes=True, shared_xaxes=True,
                            x_title="Frequency bands")

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"]=="aec"].r,
                                 x=df[df["FC_measure"]=="aec"].band,
                                 y=df[df["FC_measure"]=="aec"].subj,
                                 colorscale='Viridis', colorbar=dict(title="Pearson's r", thickness=10),
                                  zmin=min(df.r), zmax=max(df.r)), row=1, col=1)

        fig.add_trace(go.Heatmap(z=df[df["FC_measure"] == "plv"].r,
                                 x=df[df["FC_measure"] == "plv"].band,
                                 y=df[df["FC_measure"] == "plv"].subj,
                                 colorscale='Viridis', showscale=False,
                                 zmin=min(df.r), zmax=max(df.r)), row=1, col=2)
        fig.update_layout(
            title_text='Structural - Functional correlations')
        pio.write_html(fig, file=folder+"/%s.html" % title, auto_open=auto_open)

    elif title=="interW":
        fig=ff.create_annotated_heatmap(df, names, names, colorscale="Viridis", showscale=True, colorbar=dict(title="Pearson's r"))
        fig.update_layout(title_text="Correlations in sctructural connectivity between real subjects")
        pio.write_html(fig, file=folder+"/%s.html" % title, auto_open=auto_open)

    elif "fft-bm" in title:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))
        fig.add_trace(go.Scatter(text=np.round(df.mS_bm, 2), x=df.speed, y=df.G, mode="text", textfont=dict(color="crimson")))
        fig.update_layout(
            title_text="Heatmap for simulation's mean signal FFT peak; in red Hartigans' bimodality test's p-value. (0 -> p<2.2e-16)")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder+"/paramSpace-%s.html" % title, auto_open=auto_open)

    elif "regional" in title:
        fig = go.Figure()
        # for regional plots

    else:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=df.mS_peak, x=df.speed, y=df.G, colorscale='Viridis', colorbar=dict(title="FFT peak (Hz)")))

        fig.update_layout(
            title_text="FFT peak of simulated signals by Coupling factor and Conduction speed")
        fig.update_xaxes(title_text="Conduction speed (m/s)")
        fig.update_yaxes(title_text="Coupling factor")
        pio.write_html(fig, file=folder+"/paramSpace-FFTpeak_%s.html" % title, auto_open=auto_open)


def emp_sim():
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