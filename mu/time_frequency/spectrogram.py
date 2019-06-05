# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
import numpy as np
import matplotlib.pyplot as plt
from .stft import (stft, istft)
from ..viz.spectra_plot import (_plot_spectrogram)
class Spectrogram(object):
    """ This class represent a time series waveform into spectrogram.
    Note: At the moment, the class only used a Fourier based method.

    nch: int
        The number of channels of the processing signal.

    nsamp: int
        The sample size of the processing signal.

    sample_rate: int
        Sampling of the input signal.

    binsize: int (default: 1024)
        The number of samples used for each analysis window for STFT.

    hopsize: int
        The number of sample it takes to hop between rows.

    overlap_factor: float (default: 0.5)
        The ratio of overlapping between chuncks.
    """
    def __init__(self, nch=1, nsamp=2**11, sample_rate=None, binsize=2**14, hopsize=None, overlap_factor=.5):

        self._overlap_factor = overlap_factor
        self._binsize = binsize
        self._hopsize = hopsize
        self._sample_rate = sample_rate
        self._nsamp = nsamp

        self._istft = None
        self._stft = None

    def analysis(self, x, axis=-1):
        """
        Processing to get the spectra.

        Parameters:
        -----------
        x: ndarray, (nch x nsamp)
            The input signal.

        axis: int (default: -1)
            The processing axis.
        """
        self._stft = stft(x, binsize = self.binsize,
                                overlap_factor = self.overlap_factor,
                                hopsize = self.hopsize,
                                window = 'hanning',
                                planner_effort='FFTW_ESTIMATE', axis=axis)

        return self._stft

    def synthesis(self, X=None):
        if X is None:
            if self._stft is None:
                raise ValueError("'analysis' method has yet to run.")
            else:
                X = self._stft

        self._istft = istft(X, nsamp=None,
                               binsize=self.binsize,
                               overlap_factor=self.overlap_factor,
                               hopsize=self.hopsize)
        return self._istft

    def reconstruction_error(self, x):
        if self._istft is None:
            return self._istft - x
        else:
            raise ValueError("'synthsis' method has yet to run.")

    def plot_spectra(self, ch=None, axs=None, tlim=None, flim=None, figsize=None, norm='db',
                           title=None, label=False, xlabel=False, ylabel=False,
                           fontsize={'ticks': 15, 'axis': 15, 'title': 20}):

        spec_ = self._stft[ch,:,:][np.newaxis,:,:] if ch is not None else self._stft
        nch, tbins, fbins = spec_.shape

        # Build Figures
        figsize = (4 * nch, 5) if figsize is None else figsize
        if axs is None:
            fig, axs = plt.subplots(1, nch, figsize=figsize)
        else:
            fig = axs[0].figure

        self._axs = np.array(axs).ravel()
        self._fig = fig

        for (i,), ax in np.ndenumerate(self._axs):
            _plot_spectrogram(spec_[i,:,:],
                axs=ax, title=title, cmap='jet',
                srate=self.sample_rate, nsamp=self.nsamp,
                label=label, xlabel=xlabel, ylabel=ylabel, tlim=tlim, flim=flim, norm=norm,
                fontsize=fontsize,
            )

    @property
    def overlap_factor(self):
        return self._overlap_factor

    @property
    def binsize(self):
        return self._binsize

    @property
    def hopsize(self):
        return self._hopsize

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def sample_rate(self):
        return self._sample_rate
