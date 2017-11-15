from __future__ import division
"""A module for filter bank.

"""
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
from copy import deepcopy

import numpy as np
from scipy.signal import get_window
from pyfftw.interfaces.numpy_fft import (rfft, irfft, ifft, fftfreq)

from . import stft
from .tools import (get_center_frequencies, get_frequency_of_interests, create_filter, reshape_data)
from ..utilities.parallel import Parallel
class FilterBank(object):

    def __init__(self, nch, nsamp, binsize=1024, overlap_factor=.5, hopsize=None, decimate_by=1,
                 bw=None, cf=None, foi=None, order=None, sfreq=None, nprocs=1):

        self.decimate_by = decimate_by
        # Organize data
        self.nch, self.nsamp = int(nch), int(nsamp)

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

        self.binsize_dec, self.hopsize_dec, self.nsamp_dec = self.binsize // self.decimate_by, self.hopsize // self.decimate_by, self.nsamp // self.decimate_by

        self.nprocs = nprocs
        if self.nprocs > 1:
            self.pfunc = Parallel(self._fft_procs,
                         nprocs=self.nprocs, axis=1,
                         ins_shape= [(self.nch, self.nsamp),self.win_idx.shape], ins_dtype=[np.float32, np.int32],
                         out_shape= (self.nch, self.win_idx.shape[0], self.nfreqs, self.binsize_dec), out_dtype=np.complex64,
                         binsize=self.binsize_dec, filt=True, domain='time', planner_effort='FFTW_ESTIMATE', window='hanning')

    def kill(self, opt=None): # kill the multiprocess
        self.pfunc.kill(opt=opt)

    def analysis(self, x, nsamp=None, filt=False, window='hanning', domain='time', **kwargs):
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
        nsamp = nsamp // self.decimate_by

        func = self.pfunc.result if self.nprocs > 1 else self._fft_procs
        x_ = func(x, self.win_idx, binsize=self.binsize_dec, window=window, filt=filt, domain=domain, **kwargs)
        x_ = np.asarray(x_, dtype=np.float32)

        # Reconstructing the signal using overlap-add
        _x = np.zeros((self.nch, self.nfreqs, self.nwin * self.binsize_dec))
        for ix in range(self.nwin):
            jx = (1-self.overlap_factor) * ix
            if int((jx+1)*self.binsize_dec) <= self.binsize_dec * self.nwin * (1-self.overlap_factor):
                _x[:,:,int(jx*self.binsize_dec):int((jx+1)*self.binsize_dec)] += x_[:,ix,:,:]

        if nsamp is not None:
            return _x[:,:,self.binsize_dec//2:nsamp+self.binsize_dec//2]
        else:
            return _x

    def synthesis(self, X, **kwargs):
        """
        Construct using iSTFT.
        Full Reconstruction of the signal.
        Frequency band reconstruction of the signal.
        """
        return stft.istft(X, nsamp=self.binsize_dec, binsize=self.binsize, overlap_factor=self.overlap_factor, **kwargs)

    def _fft_procs(self, x, win_idx, binsize=1024, filt=False, domain='time', axis=-1, **kwargs):
        X = stft.stft(x, win_idx=win_idx, **kwargs) / self.decimate_by

        _nwin, _nbins = win_idx.shape

        X_ = np.zeros((self.nch, _nwin, self.nfreqs, binsize//2), dtype=np.complex64)

        if filt:
            X_[:,:,self.idx2,self.idx1] = X[:,:,self.idx1] * self.filts[:,self.fidx][:,np.newaxis,:]
        else:
            X_[:,:,self.idx2,self.idx1] = X[:,:,self.idx1]

        if domain == 'freq':
            return X_
        elif domain == 'time':
            x_ = irfft(X_, n=binsize, axis=-1, planner_effort='FFTW_ESTIMATE')
            return x_

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
        _, filts = create_filter(order, f_cut, fs/2., N, shift=True)

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
            self._foi = get_frequency_of_interests(self.center_f, self.bandwidth)

        if self.center_f is None or self.bandwidth is None:
            self._center_f, self._bandwidth = get_center_frequencies(self.foi)
