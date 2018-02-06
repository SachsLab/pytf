from __future__ import division
""" A module for filter bank.
"""
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
import logging

import numpy as np
from scipy.signal import (get_window, group_delay)
from pyfftw.interfaces.numpy_fft import (rfft, irfft, ifft, fftfreq)

import matplotlib.pyplot as plt

from .filter import (create_filter)
from ..reconstruction.overlap import (overlap_add)
from ..time_frequency.stft import (_check_winsize, stft)
from ..utilities.parallel import (Parallel, ParallelDummy)
from ..viz.filter_plot import (_plot_filter)

def _is_uniform_distributed_cf(cf):
    """ Check if the provided center frequencies are uniformly distributed.
    """
    return np.any(np.diff(np.diff(cf))!=0)

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.info('Start reading database')

class FilterBank(object):
    """ Create a filter bank object for signal processing.

    Parameters:
    -----------
    binsize: int (default: 1024)
        The number of samples used for each analysis window for STFT.

    decimate_by: int (default: 1)
        The decimating factor.

    nprocs: int (default:1)
        The number of processes for filtering.

    domain: str (default: 'freq')
        Specify if the return to be in frequency domain ('freq'), or time domain ('time').

    bandwidth: float (default: None)
        The bandwidth of the filter. In this case, it's twice the cutoff frequency of the lowpass filter.

    center_freqs: ndarray (default: None)
        The center frequencies of each frequency bands of interest.

    freq_bands: ndarray (default: None)
        The frequency bands of interest.

    order: int (default: None)
        The order of the filter.

    sample_rate: int (default: None)
        The sample rate of the signal and the filter.

    filt: bool (default: False)
        If False, no filter will be used.
        If True, the pre-defined filter will be used.

    hilbert: bool (default: False)
        If False, the output signal is real.
        If True, the output signal is analytical (real and imaginary).
    """
    def __init__(self, nch=1, nsamp=2**14, binsize=2**10, decimate_by=1, \
                 bandwidth=None, center_freqs=None, freq_bands=None, order=None, sample_rate=None, \
                 hilbert=False, domain='time', nprocs=1, mprocs=False,
                 logger=None):

        self.logger = logging.getLogger("%s" % self.__class__)
        self.logger.info("Creating the FilterBank class.")
        # Pre-defined Parameters
        self._factor = .6
        _overlap_factor = 0.5

        # Filter Output Parameters
        self.hilbert = hilbert
        self.domain = domain

        # Signal Parameters
        self._nch = nch
        self._decimate_by = decimate_by
        self._nsamp = nsamp

        # Overlap-Window Parameters
        self._binsize = binsize
        self._nwin = int((self.nsamp / self._binsize) / _overlap_factor + 1)

        # Frequency Parameters
        self._sample_rate = sample_rate
        self._center_freqs, self._bandwidth, self._freq_bands = self.get_all_frequencies(center_freqs, bandwidth, freq_bands)

        self._nfreqs = self.freq_bands.shape[0]
        self._interval_per_hz = self._binsize / self.sample_rate # interval per Hz

        # The decimated sample size
        self._binsize_ = self._binsize // self.decimate_by

        # Create indices for efficiently filtering the signal
        self._get_indices_for_frequency_shifts()

        # Create a prototype filter
        self._order = order
        self._filts = self._create_prototype_filter(shift=True, output='freq')[1]
        self.logger.info("Created the prototype filter.")

        self._delay = self.delayed_samples()
        self._delay_ = self.delay // self.decimate_by

        # Initializing for multiprocessing
        self._nprocs = nprocs
        self._mprocs = True if self.nprocs > 1 else mprocs

        if self._mprocs:
            self.logger.info("Enabled multiprocessing.")

        ndtype = np.complex64 if self.hilbert else np.float32
        self._pfunc = Parallel(
                        self._fft_procs, nprocs=self.nprocs, axis=2,
                        ins_shape = [(self.nch, self._nwin, self._binsize//2 + 1),
                                    (self.nfreqs, int((self.bandwidth * self._factor) * 2 * self.interval_per_hz)),
                                    (self.nfreqs, int((self.bandwidth * self._factor) * 2 * self.interval_per_hz)),
                                    (self.nfreqs, int((self.bandwidth * self._factor) * 2 * self.interval_per_hz))],
                        out_shape=(self.nch, self._nwin, self.nfreqs, self._binsize // self.decimate_by),
                        ins_dtype = [np.complex64, np.int32, np.int32, np.int32],
                        out_dtype = ndtype,
                        dtype = ndtype,
                        filts = self._filts,
                        nfreqs = self.nfreqs
                    ) if self.mprocs else ParallelDummy(self._fft_procs, dtype=ndtype, filts=self._filts, nfreqs=self.nfreqs)

        self.logger.info("Initialized FilterBank.")

    # def __str__(self):
    #     return self
    #
    # def __repr__(self):
    #     return self

    def kill(self, opt=None): # kill the multiprocess
        """ Killing all the multiprocessing processes.
        """
        self._pfunc.kill(opt=opt)

    def analysis(self, x, window='hamming'):
        """ Generate the analysis bank.

        Parameters:
        -----------
        x: ndarray, (nch x nsamp)
            The input signal.

        window: str (default: 'hamming')
            The window used to create overlapping slices of the time domain signal.
        """
        ndtype = np.complex64 if self.hilbert else np.float32

        nch, nsamp = x.shape
        nsamp //= self.decimate_by

        X = stft(x, binsize=self._binsize, window=window, axis=-1, \
                    planner_effort='FFTW_ESTIMATE') / self.decimate_by

        x_ = self._pfunc.result(X, self._idx1, self._idx2, self._fidx)
        x_ = np.concatenate([x_[:,:,:,self.delay_:], x_[:,:,:,:self.delay_]], axis=-1)\
                if self._filts is not None else x_

        # Reconstructing the signal using overlap-add
        _x = overlap_add(x_, self._binsize_, overlap_factor=.5, dtype=ndtype)
        return _x[:,:,self._binsize_//2:nsamp+self._binsize_//2]

    def synthesis(self, x, **kwargs):
        """ TODO: Reconstruct the signal from the analysis bank.
        """
        return

    def _fft_procs(self, X, idx1, idx2, fidx, filts=None, nfreqs=None, \
                        slices_idx=[slice(None)]*4, dtype=np.float32):
        """ FFT filtering using STFT on the signal.

        Paramters:
        ----------
        X: ndarray (nch x nwin x nsamp)
            The STFT of the signal to be analyzed.

        idx1: ndarray
            The fancy index on X. This demodulates the signal. See self._get_indices_for_frequency_shifts().

        idx2: ndarray
            The fancy index for reconstructing X_. This modulates
            the demodulated signal. See self._get_indices_for_frequency_shifts().

        fidx: ndarray
            The fancy index for slicing the specific frequency components from the frequency
            response of the filter coefficients. See self._get_indices_for_frequency_shifts().

        slices_idx: list
            This argument is only needed when implementing in the Parallel class.
            This specifies how to split the given indices into the number of processes as evenly as possible.

        dtype: ndarray type (default: np.float32)
            The ndarray type of the signal output.
        """
        nch, nwin, nsamp = X.shape
        X_ = np.zeros((nch, nwin, nfreqs, self._binsize_//2), dtype=np.complex64)
        X_[:,:,idx2,idx1] = X[:,:,idx1] * filts[fidx]

        if dtype == np.float32:
            _ifft = irfft
        else:
            X_[:,:,:,1:] *= 2
            _ifft = ifft

        if self.domain == 'freq':
            return X_

        elif self.domain == 'time':
            return _ifft(X_[slices_idx], n=self._binsize_, axis=-1, planner_effort='FFTW_ESTIMATE')

    def delayed_samples(self):
        """ The group delay from the prototype filter.
        """
        filt = self._create_prototype_filter(output='time')[1]
        return int(np.mean(group_delay([filt,1])[1]))

    def plot_filter(self, xlim=None, ylim=None,
                    label=False, xlabel=False, ylabel=False,
                    plot_group_delay=None):
        """ Visualize the prototype filter.
        """
        xlabel = True if label else xlabel
        ylabel = True if label else ylabel

        _w, _filts = self._create_prototype_filter(shift=True, output='freq')
        _fig, _ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        _ax[0].plot(_w, np.abs(_filts))
        _ax[1].plot(_w, np.angle(_filts))
        if xlim is not None:
            _ax[0].set_xlim(xlim)
            _ax[1].set_xlim(xlim)

        if xlabel:
            _ax[1].set_xlabel('Frequencies [Hz]')
        if ylabel:
            _ax[0].set_ylabel('Magnitude')
            _ax[1].set_ylabel('Phases [rad]')

        if plot_group_delay is not None:
            _filt = self._create_prototype_filter(output='time')[1]
            _w, _gd = group_delay([_filt,1])
            _w *= (self.sample_rate / (2*np.pi))

            _fig1, _ax1 = plt.subplots(1,1,figsize=(8,3))
            _ax1.plot(_w, _gd)

            if xlim is not None:
                _ax1.set_xlim([0, xlim[-1]])

            _ax1.set_ylim([0, _gd.max()*1.25])

            if xlabel:
                _ax1.set_xlabel('Frequencies [Hz]')
            if ylabel:
                _ax1.set_ylabel('Number of Samples')

            return [_fig, _fig1]
        else:
            return _fig

    def _create_prototype_filter(self, **kwargs):
        """ Create the prototype filter, which is the only filter require for
        windowing in the frequency domain of the signal. This filter is a lowpass filter.
        """
        return create_filter(self.order, self.bandwidth/2., self.sample_rate/2.,\
                             self._binsize, **kwargs)

    def _get_indices_for_frequency_shifts(self):
        """ Get the indices for properly shifting the fft of signal to DC, and the indices
        for shifting the fft of signal back to the correct frequency indices for ifft.
        """
        fois_ix_ = np.asarray(self.freq_bands * self.interval_per_hz, dtype=np.int32)
        cf_ix_ = np.asarray(self.center_freqs * self.interval_per_hz, dtype=np.int32)

        # Get indices for filter coeffiecients
        self._fidx = np.zeros((self.nfreqs, int((self.bandwidth * self._factor) * 2 * self.interval_per_hz)), dtype=np.int32)
        cf0 = self._binsize // 2
        for ix, f_ix in enumerate(fois_ix_):
            l_bound = cf0 - int(self.interval_per_hz * self.bandwidth * self._factor)

            diff = self._fidx[ix,:].shape[-1] - np.arange(l_bound, l_bound + (self.bandwidth * self._factor) * 2 * self.interval_per_hz).size
            self._fidx[ix,:] = np.arange(l_bound, l_bound + (self.bandwidth * self._factor) * 2 * self.interval_per_hz + diff)

        self._fidx = np.asarray(self._fidx, dtype=np.int32)

        # Code 1: Does the same thing as below
        x = np.arange(0, int((self.bandwidth * self._factor) * 2 * self.interval_per_hz))
        y = np.arange(0, self.nfreqs)
        index1, index2 = np.meshgrid(x, y)

        index1 += (np.atleast_2d(cf_ix_) - int(self.interval_per_hz * self.bandwidth * self._factor))
        self._idx1 = np.asarray(index1, dtype=np.int32)
        self._idx2 = np.asarray(index2, dtype=np.int32)

    @staticmethod
    def get_center_frequencies(fois):
        """ Convert an array of frequency bands into center frequencies and a bandwidth.
        TODO: Support for varying bandwidths.

        Parameters:
        -----------
        fois: ndarray, (nfreq x 2)
            An array of frequency bands of interents. Each row consists of upper and lower
            bound of frequencies. The unit for frequencies must be consistant with the sampling rate.

        Return:
        -------
        cf: ndarray, (nfreqs x 1)
            An array of center frequencies corresponding to the fois.

        bw: float
            The bandwidth. The width between the upper and lower cutoff frequencies.
        """
        foi = np.asarray(fois)
        if fois.shape[0] == 2 and fois.shape[1] != 2:
            fois = fois.T

        cf = np.atleast_2d(fois.mean(axis=-1)).T
        bw = np.diff(fois, axis=-1)
        bw = bw[0] if np.diff(bw) else bw.mean()
        return cf, float(bw)

    @staticmethod
    def get_frequency_bands(cf, bw):
        """ Convert an array of center frequencies and a bandwidth into an array of frequency bands.
        TODO: Support for varying bandwidths.

        Parameters:
        -----------
        cf: ndarray, (nfreqs x 1)
            An array of center frequencies corresponding to the fois.

        bw: float
            The bandwidth. The width between the upper and lower cutoff frequencies.

        Returns:
        --------
        fois: ndarray, (nfreq x 2)
            An array of frequency bands of interents. Each row consists of upper and lower
            bound of frequencies. The unit for frequencies must be consistant with the sampling rate.
        """
        cf = np.asarray(cf)
        bw = np.asarray(bw)
        if cf.ndim == 1:
            cf = np.atleast_2d(cf).T
        else:
            if cf.shape[1] == cf.size:
                cf = cf.T

        bw = bw * np.ones((cf.size, 2))
        bw[:,0] *= -.5
        bw[:,1] *= .5

        return cf + bw

    @staticmethod
    def get_all_frequencies(cf=None, bw=None, fois=None):
        """ Ensures all frequencies, fois, cf, and bw.

        Parameters:
        -----------
        cf: ndarray, (nfreqs x 1)
            An array of center frequencies corresponding to the fois.

        bw: float
            The bandwidth. The width between the upper and lower cutoff frequencies.

        fois: ndarray, (nfreq x 2)
            An array of frequency bands of interents. Each row consists of upper and lower
            bound of frequencies. The unit for frequencies must be consistant with the sampling rate.

        Return:
        --------
        cf, bw, fois
        """
        if (cf, bw, fois) is (None, None, None):
            raise ValueError("Must enter one of the following arguments: 'cf', 'bw', 'fois.")

        if fois is None:
            if cf.ndim == 1:
                cf = np.atleast_2d(cf).T
            else:
                if cf.shape[1] == cf.size:
                    cf = cf.T
            fois = FilterBank.get_frequency_bands(cf, bw)

        if cf is None or bw is None:
            cf, bw = FilterBank.get_center_frequencies(fois)

        return cf, bw, fois

    @property
    def freq_bands(self):
        return self._freq_bands

    @property
    def center_freqs(self):
        return self._center_freqs

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def nfreqs(self):
        return self._nfreqs

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def order(self):
        return self._order

    @property
    def interval_per_hz(self):
        return self._interval_per_hz

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def mprocs(self):
        return self._mprocs

    @property
    def delay(self):
        return self._delay

    @property
    def delay_(self):
        return self._delay_

    @property
    def nch(self):
        return self._nch

    @property
    def decimate_by(self):
        return self._decimate_by

    @property
    def nsamp(self):
        return self._nsamp
