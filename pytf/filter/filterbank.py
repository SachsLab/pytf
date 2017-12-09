from __future__ import division
"""A module for filter bank.
"""
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
import time
import numpy as np
from scipy.signal import get_window
from pyfftw.interfaces.numpy_fft import (rfft, irfft, ifft, fftfreq)

from .filter import (create_filter, get_center_frequencies, get_frequency_of_interests, get_all_frequencies)
from ..reconstruction.overlap import (overlap_add)
from ..time_frequency.stft import (_check_winsize, stft)
from ..utilities.parallel import Parallel

def _is_uniform_distributed_cf(cf):
    """
    Check if the provided center frequencies are uniformly distributed.
    """
    return np.any(np.diff(np.diff(cf))!=0)

class FilterBank(object):
    """

    """
    # __slots__ = ['center_frequencies', 'bandwidth', 'frequency_bands', 'sampling_rate',\
    #              'binsize', 'overlap_factor', 'hopsize', 'decimate_by']

    def __init__(self, binsize=1024, decimate_by=1,
                 bw=None, cf=None, foi=None, order=None, sfreq=None, filt=True, \
                 **kw_pfunc):

        self.decimate_by = decimate_by

        # Create indices for overlapping window
        self.binsize = binsize

        # The decimated sample size
        self.binsize_ = self.binsize // self.decimate_by

        # Organize frequencies
        self._center_f, self._bandwidth, self._foi = get_all_frequencies(cf, bw, foi)

        self._nfreqs = self.foi.shape[0]

        self._sfreq = sfreq
        self._int_phz = self.binsize / self.sfreq # interval per Hz

        # Create indices for efficiently filtering the signal
        self._get_indices_for_frequency_shifts()

        # Create a prototype filter
        self._order = order
        self.filts = self._create_prototype_filter(self.order, self.bandwidth/2., self.sfreq, self.binsize)\
                        if filt else None

        # Initializing for multiprocessing
        self.nprocs = 1
    #     kw_pfunc = ['nprocs', 'in_shape', 'out_shape', '_fft_procs_kwargs']
    #     self.pfunc = Parallel(self._fft_procs,
    #                  nprocs=self.nprocs, axis=1,
    #                  ins_shape=(self.nch, self.nsamp),
    #                  ins_dtype=np.float32,
    #                  out_shape=(self.nch, self.win_idx.shape[0], self.nfreqs, self.binsize_),
    #                  out_dtype=np.complex64,
    #                  binsize=self.binsize_, filt=True, domain='time', window='hanning')
    #
    # def kill(self, opt=None): # kill the multiprocess
    #     self.pfunc.kill(opt=opt)

    def analysis(self, x, nsamp=None, hilbert=False, **kwargs):
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
        ndtype = np.complex64 if hilbert else np.float32

        nsamp = x.shape[-1]
        nsamp //= self.decimate_by

        func = self.pfunc.result if self.nprocs > 1 else self._fft_procs
        indices =  (self._idx1,
                    self._idx2,
                    self._fidx)
        x_ = func(x, indices, dtype=ndtype, **kwargs)

        # Reconstructing the signal using overlap-add
        _x = overlap_add(x_, self.binsize_, overlap_factor=.5, dtype=ndtype)

        return _x[:,:,self.binsize_//2:nsamp+self.binsize_//2]

    def synthesis(self, x, **kwargs):
        """
        TODO: Reconstruct the signal from the analysis bank.
        """
        return

    def _fft_procs(self, x, indices, window='hamming', domain='freq', dtype=np.float32):
        """
        FFT filtering using STFT on the signal.

        Paramters:
        ----------
        x: ndarray (nch x nsamp)
            The signal to be analyzed.
        """
        t0 = time.time()
        nch, nsamp = x.shape
        X = stft(x, binsize=self.binsize, window=window, axis=-1, planner_effort='FFTW_ESTIMATE') / self.decimate_by
        X_ = np.zeros((nch, X.shape[1], self.nfreqs, self.binsize_//2), dtype=np.complex64)

        idx1 = indices[0]
        idx2 = indices[1]
        fidx = indices[2]
        if self.filts is None:
            X_[:,:,idx2,idx1] = X[:,:,idx1]
        else:
            X_[:,:,idx2,idx1] = X[:,:,idx1] * self.filts[fidx]

        if dtype == np.float32:
            _ifft = irfft
        else:
            X_[:,:,:,1:] *= 2
            _ifft = ifft

        if domain == 'freq':
            return X_

        elif domain == 'time':
            x_ = _ifft(X_, n=self.binsize_, axis=-1, planner_effort='FFTW_ESTIMATE')
            print('time: {} ms'.format((time.time() - t0)*1E3))
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
        self._fidx = np.zeros((self.nfreqs, int(self.bandwidth * 2 * self._int_phz)), dtype=np.int64)
        cf0 = self.binsize // 2
        for ix, f_ix in enumerate(fois_ix_):

            if f_ix[0] <= self.bandwidth:
                l_bound = cf0 - int(self._int_phz * self.bandwidth // 4) - 1
            else:
                l_bound = cf0 - int(self._int_phz * self.bandwidth) - 1

            diff = self._fidx[ix,:].shape[-1] - np.arange(l_bound, l_bound + self.bandwidth * 2 * self._int_phz).size

            self._fidx[ix,:] = np.arange(l_bound, l_bound + self.bandwidth * 2 * self._int_phz + diff)

        # Code 1: Does the same thing as below
        x = np.arange(0, int(self.bandwidth * 2 * self._int_phz))
        y = np.arange(0, self.nfreqs)
        index1, index2 = np.meshgrid(x, y)
        index1 += np.atleast_2d(fois_ix_[:,0]).T
        self._idx1 = np.asarray(index1, dtype=np.int64)
        self._idx2 = np.asarray(index2, dtype=np.int64)
