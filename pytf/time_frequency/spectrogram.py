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

    nch:

    nsamp:

    """
    def __init__(self, nch=1, nsamp=2**11, binsize=2**14, sample_rate=None):

        self._overlap_factor = .5
        self._binsize = binsize
        self._hopsize = None
        self._sample_rate = sample_rate
        self._nsamp = nsamp

    def analysis(self, x, axis=-1):
        self._spectra = stft(x, binsize = self.binsize,
                                overlap_factor = self.overlap_factor,
                                hopsize = self.hopsize,
                                window = 'hanning',
                                planner_effort='FFTW_ESTIMATE', axis=axis)

        return self._spectra

    def synthesis(self, X):
        return

    def plot_spectra(self, ch=None, axs=None, tlim=None, flim=None, figsize=None,
                           title=None, label=False, xlabel=False, ylabel=False,
                           fontsize={'ticks': 15, 'axis': 15, 'title': 20}):

        spec_ = self._spectra[ch,:,:][np.newaxis,:,:] if ch is not None else self._spectra
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
                label=label, xlabel=xlabel, ylabel=ylabel, tlim=tlim, flim=flim, norm='db',
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
