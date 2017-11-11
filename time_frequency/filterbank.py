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

    def __init__(self, data, binsize=1024, overlap_factor=.5,
                 bw=None, cf=None, foi=None,
                 order=None, sfreq=None, decimate_by=1):

        self.decimate_by = decimate_by
        self.binsize = binsize
        self.overlap_factor = overlap_factor

        # Organize frequencies of interests
        self._freqs = foi
        self._center_f = cf
        self._bandwidth = bw
        self._get_all_frequencies()

        self._nfreqs = self.freqs.shape[0]

        # Organize data
        self._data = self._reshape_data(data)
        self.nch, self.nsamp = self._data.shape

        self._sfreq = sfreq
        self._int_phz = self.binsize / self.sfreq # interval per Hz
        self._nsamp = int(self.bandwidth * 2 * self.int_phz)

        # Create indices for efficiently filtering the signal
        self._get_indices_for_frequency_shifts()

        # Create a prototype filter
        self._order = order
        self.filts = self._create_prototype_filter(self.order, self.bandwidth/2., self.sfreq, self.binsize)
        self.filts = np.atleast_2d(self.filts)

    def analysis(self, x, window='hanning', **kwargs):
        """
        Deconstruct the signal using STFT.
        """
        return stft.stft(x, binsize=self.binsize, overlap_factor=self.overlap_factor, window=window, **kwargs)

    def synthesis(self):
        """
        Construct using iSTFT.
        Full Reconstruction of the signal.
        Frequency band reconstruction of the signal.
        """
        return

    @property
    def freqs(self):
        return self._freqs

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

    def process(self, data, **kwargs):
        return self.filter_data(data, self.filts, **kwargs)

    def filter_data(self, data, filts, axis=-1, mode='hilbert', planner_effort='FFTW_ESTIMATE'):
        """
        Perform FFT filtering on the data, given the FFT of predefined filter coefficients.
        """
        _data = self._reshape_data(data, axis=axis)

        nsampd = self.nsamp // self.decimate_by
        data_f = fft.rfft(_data, axis=axis, planner_effort=planner_effort) / self.decimate_by
        if mode == 'hilbert':
            _X = np.zeros((self.nch, self.nfreqs, nsampd), dtype=np.complex64)
            _X[:, self._idx2, self._idx1] = data_f[:,self._idx1] * filts

            return fft.ifft(_X * 2, axis=axis, planner_effort=planner_effort)

        else:
            _X = np.zeros((self.nch, self.nfreqs, nsampd//2), dtype=np.complex64)
            _X[:, self._idx2, self._idx1] = data_f[:,self._idx1] * filts

            return fft.irfft(_X, axis=axis, planner_effort=planner_effort)

    @staticmethod
    def _reshape_data(data, axis=-1):

        if data.ndim > 2 or axis > 1:
            raise ("The data should only be in 2 dimensional. \
                    The support for 3 dimensional have not been implemented yet.")
        if data.ndim == 1:
            data = np.atleast_2d(data)

        if axis != 0:
            return data
        else:
            return data.T

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
        fois_ix_ = np.asarray(self.freqs * self._int_phz, dtype=np.int64)

        # Get indices for filter coeffiecients
        self.fidx = np.zeros((self.nfreqs, int(self.bandwidth * 2 * self._int_phz)), dtype=np.int64)
        cf0 = self.binsize // 2
        for ix, f_ix in enumerate(fois_ix_):
            if f_ix[0] <= self.bandwidth:
                l_bound = cf0 - int(self._int_phz * self.bandwidth // 4)
            else:
                l_bound = cf0 - int(self._int_phz * self.bandwidth)

            self.fidx[ix,:] = np.arange(l_bound, l_bound + self.bandwidth * 2 * self._int_phz-1)
            print(l_bound)
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
        if (self.center_f, self.bandwidth, self.freqs) is (None, None, None):
            raise "Must enter one of the following kwargs: 'cf', 'bw', 'fois."

        if self.freqs is None:
            self._freqs = self.get_frequency_of_interests(self.center_f, self.bandwidth)

        if self.center_f is None or self.bandwidth is None:
            self._center_f, self._bandwidth = self.get_center_frequencies(self.freqs)

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
