import numpy as np
import matplotlib.pyplot as plt
from .utils.normalization import (logscale_normalization)
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
def _plot_spectrogram(spectra, axs=None, figsize=None, title=None, cmap='jet',
                      srate=None, nsamp=None,
                      label=False, xlabel=False, ylabel=False, tlim=None, flim=None, norm='db',
                      fontsize={'ticks': 15, 'axis': 15, 'title': 20}, **kwargs):
    """ Plot spectrogram of a given spectra.

    Parameters:
    -----------
    spectra: ndarray (n_win x nsamp)
        n_win is the length of the time indices, and nsamp is the length of the frequency indices in this case.
    """
    xlabel = True if label else xlabel
    ylabel = True if label else ylabel

    spec_, freq = logscale_normalization(spectra, factor=1, srate=srate)
    if norm is 'db':
        spec_ = 20. * np.log10(np.abs(spec_)/10e-6) # amplitude to decibel

    tbins_, fbins_ = spec_.shape

    # The x and y indices
    tvecs = np.linspace(0, nsamp, tbins_) / srate

    tlim = (tvecs[0], tvecs[-1]) if tlim is None else tlim
    flim = (freq[0], freq[-1]) if flim is None else flim

    t_ = np.where(np.logical_and(tvecs>=tlim[0], tvecs<=tlim[-1]))[0]
    f_ = np.where(np.logical_and(freq>=flim[0], freq<=flim[-1]))[0]

    tidx = slice(t_[0], t_[-1])
    fidx = slice(f_[0], f_[-1])

    tvecs = tvecs[tidx]
    freq = freq[fidx]
    spec = spec_[tidx, fidx]

    t_bins, f_bins = spec.shape

    # Build Figures
    figsize = (4, 5) if figsize is None else figsize
    if axs is None:
        _fig, _ax = plt.subplots(1, 1, figsize=figsize)
    else:
        _ax = axs
        _fig = _ax.figure

    ima = _ax.imshow(np.transpose(spec), origin="lower", aspect="auto", cmap=cmap, **kwargs)

    if title is not None:
        _ax.set_title(title, fontsize=fontsize['title'])

    if xlabel:
        # x-axis
        xlocs = np.int16(np.linspace(0, t_bins-1, 5))
        _ax.set_xticks(xlocs)
        _ax.set_xticklabels(['{}'.format(int(tvecs[i])) for i in xlocs], fontsize=fontsize['ticks'])
        _ax.set_xlabel('Time [{}]'.format('s'), fontsize=fontsize['axis'])
        for tick in _ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize['ticks'])
    else:
        _ax.set_xticks([])

    if ylabel:
        # y-axis
        ylocs = np.int16(np.linspace(0, f_bins-1, 10))
        _ax.set_yticks(ylocs)
        _ax.set_yticklabels(['{}'.format(int(freq[i])) for i in ylocs], fontsize=fontsize['ticks'])
        _ax.set_ylabel('Frequency [{}]'.format('Hz'), fontsize=fontsize['axis'])
        for tick in _ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize['ticks'])
    else:
        _ax.set_yticks([])

    _fig.colorbar(ima)

    return _fig
