from __future__ import division
"""A module for filter bank.

"""
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
from copy import deepcopy

import numpy as np
from scipy.signal import firwin
import pyfftw.interfaces.numpy_fft as fft
from . import stft

class FilterBank(object):

    def __init__(self, nch, nsamp, binsize=1024, overlap_factor=.5, hopsize=None,
                 bw=None, cf=None, foi=None, order=None, sfreq=None):

        # Organize data
        self.nch, self.nsamp = nch, nsamp

        # Create indices for overlapping window
        self.binsize = binsize
        self.hopsize = hopsize
        self.overlap_factor = overlap_factor
        self.nwin = None
        args = [self.binsize, self.overlap_factor, self.hopsize, self.nwin]
        self.win_idx = stft.create_time_idx(self.nsamp, args, create_indices=True)
        self.binsize, self.overlap_factor, self.hopsize, self.nwin = args

        # Organize frequencies of interests
        self._foi = foi
        self._center_f = cf
        self._bandwidth = bw
        self._get_all_frequencies()

        self._nfreqs = self.foi.shape[0]

        self._sfreq = sfreq
        self._int_phz = self.binsize / self.sfreq # interval per Hz

        # Create indices for efficiently filtering the signal
        self._get_indices_for_frequency_shifts()

        # Create a prototype filter
        self._order = order
        self.filts = self._create_prototype_filter(self.order, self.bandwidth/2., self.sfreq, self.binsize)
        self.filts = np.atleast_2d(self.filts)

        self.time = (np.arange(self.win_idx[-1,-1]+1)[self.win_idx] - self.binsize//2) / self.sfreq
        self.time_ix = (np.arange(self.win_idx[-1,-1]+1)[self.win_idx] - self.binsize//2)

    def analysis(self, x, nsamp=None, filt=False, window='hanning', domain='freq', decimate_by=1, **kwargs):
        """
        Generate the analysis bank.

        Parameters:
        -----------
        x: ndarray, (nch x nsamp)
            The input signal.

        filt: ndarray (default: False)
            If False, no filter will be used.
            If True, the pre-defined filter will be used.

        window: str (default: 'hanning')
            The window used to create overlapping slices of the time domain signal.

        domain: str (default: 'freq')
            Specify if the return to be in frequency domain ('freq'), or time domain ('time').

        kwargs:
            The key-word arguments for pyfftw.
        """

        nsamp = x.shape[-1] if nsamp is None else nsamp
        nsamp = nsamp // decimate_by
        fftsize = self.binsize // decimate_by

        X = stft.stft(x, win_idx=self.win_idx, window=window, **kwargs) / decimate_by
        X_ = np.zeros((self.nch, self.nwin, self.nfreqs, fftsize//2), dtype=np.complex64)
        if filt:
            X_[:,:,self.idx2,self.idx1] = X[:,:,self.idx1] * self.filts[:,self.fidx][:,np.newaxis,:]
        else:
            X_[:,:,self.idx2,self.idx1] = X[:,:,self.idx1]

        if domain == 'freq':
            return X_
        elif domain == 'time':
            return stft.istft(X_, nsamp=nsamp, binsize=fftsize, \
                              overlap_factor=self.overlap_factor, \
                              axis=-1, planner_effort='FFTW_ESTIMATE')

    def synthesis(self, X, **kwargs):
        """
        Construct using iSTFT.
        Full Reconstruction of the signal.
        Frequency band reconstruction of the signal.
        """
        return stft.istft(X, nsamp=self.analysis_size, binsize=self.binsize, overlap_factor=self.overlap_factor, **kwargs)

    @property
    def foi(self):
        return self._foi

    @property
    def center_f(self):
        return self._center_f

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def nfreqs(self):
        return self._nfreqs

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def order(self):
        return self._order

    @property
    def int_phz(self):
        return self._int_phz

    def _create_prototype_filter(self, order, f_cut, fs, N):
        """
        Create the prototype filter, which is the only filter require for windowing in the frequency
        domain of the signal. This filter is a lowpass filter.
        """
        _, filts = self.create_filter(order, f_cut, fs/2., N, shift=True)

        return filts

    def _get_indices_for_frequency_shifts(self):
        """
        Get the indices for properly shifting the fft of signal to DC, and the indices for shifting
        the fft of signal back to the correct frequency indices for ifft.
        """
        fois_ix_ = np.asarray(self.foi * self._int_phz, dtype=np.int64)

        # Get indices for filter coeffiecients
        self.fidx = np.zeros((self.nfreqs, int(self.bandwidth * 2 * self._int_phz)), dtype=np.int64)
        cf0 = self.binsize // 2
        for ix, f_ix in enumerate(fois_ix_):
            if f_ix[0] <= self.bandwidth:
                l_bound = cf0 - int(self._int_phz * self.bandwidth // 4) - 1
            else:
                l_bound = cf0 - int(self._int_phz * self.bandwidth) - 1

            diff = self.fidx[ix,:].shape[-1] - np.arange(l_bound, l_bound + self.bandwidth * 2 * self._int_phz).size

            self.fidx[ix,:] = np.arange(l_bound, l_bound + self.bandwidth * 2 * self._int_phz + diff)

        # Code 1: Does the same thing as below
        x = np.arange(0, int(self.bandwidth * 2 * self._int_phz))
        y = np.arange(0, self.nfreqs)
        index1, index2 = np.meshgrid(x, y)
        index1 += np.atleast_2d(fois_ix_[:,0]).T
        self.idx1 = np.asarray(index1, dtype=np.int64)
        self.idx2 = np.asarray(index2, dtype=np.int64)

    def _get_all_frequencies(self):
        """
        Ensures all frequencies, fois, cf, and bw.
        """
        if (self.center_f, self.bandwidth, self.foi) is (None, None, None):
            raise ValueError("Must enter one of the following arguments: 'cf', 'bw', 'fois.")

        if self.foi is None:
            self._foi = self.get_frequency_of_interests(self.center_f, self.bandwidth)

        if self.center_f is None or self.bandwidth is None:
            self._center_f, self._bandwidth = self.get_center_frequencies(self.foi)

    @staticmethod
    def get_center_frequencies(fois):
        """
        Provide an array-like frequencies of interests (foi), return the center frequencies (cf) and the bandwidth (bw).
        The units must be consistant, either all in Hz or all normalized.
        """
        foi = np.asarray(fois)
        if fois.shape[0] == 2:
            fois = fois.T

        cf = np.atleast_2d(fois.mean(axis=-1)).T
        bw = np.diff(fois, axis=-1)

        if not np.diff(bw):
            bw = float(bw.mean())

        return cf, bw

    @staticmethod
    def get_frequency_of_interests(cf, bw):
        """
        Provide an array-like center frequencies (cf) and bandwidth(s) (bw), return an array of frequency bands.
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
    def create_filter(order, cutoff, nyquist, N, ftype='fir', output='freq', shift=True):
        """
        Create a prototype filter.
        """
        if order > N:
            raise ValueError("The order of the filter should not be longer than the length for FFT (binsize).")

        if cutoff >= nyquist:
            raise ValueError("The cutoff frequency must be at least 2 times smaller than the Nyquist rate.")

        h = firwin(order, cutoff, nyq=nyquist)

        if output == 'freq':
            w = fft.fftfreq(N)
            w *= (nyquist*2)

            H = fft.fft(h, n=N, axis=-1, planner_effort='FFTW_ESTIMATE')

            if shift:
                return fft.fftshift(w), fft.fftshift(H)
            else:
                return w, H

        else:
            return h

    @staticmethod
    def _reshape_data(data, axis=-1):

        if data.ndim > 2 or axis > 1:
            raise ValueError("The data should only be in 2 dimensional. \
                    The support for 3 dimensional have not been implemented yet.")
        if data.ndim == 1:
            data = np.atleast_2d(data)

        if axis != 0:
            return data
        else:
            return data.T
