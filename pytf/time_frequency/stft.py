import numpy as np
from numpy.lib import stride_tricks

from scipy.signal import get_window
from pyfftw.interfaces.numpy_fft import (rfft, irfft, ifft, fftfreq)

from ..reconstruction.overlap import (overlap_add)
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def _check_winsize(binsize, overlap_factor=None, hopsize=None):
    """ Ensure all parameters for defining the windowing size of the signal aligns.

    Parameters:
    -----------
    binsize: int
        The length of chunk size.

    overlap_factor: float
        The ratio of overlapping between chuncks.

    hopsize: int
        The number of sample it takes to hop between rows.

    Retruns:
    --------
    The same as the input arguments, but altered values if necessary.
    """
    if overlap_factor is None and hopsize is None:
        raise ValueError("At least one of 'overlap_factor' or 'hopsize' has to have a value.")

    overlap_factor = hopsize / binsize if overlap_factor is None else overlap_factor
    hopsize = int(binsize * (1 - overlap_factor)) if hopsize is None else hopsize

    if overlap_factor in [0, 1]:
        hopsize = 0 if overlap_factor else binsize

    if np.abs(round(overlap_factor - hopsize / binsize, 3)) >= 5E-2:
        raise ValueError("The 'overlap_factor' calculated from hopsize/binsize does not match the input.")

    return binsize, overlap_factor, hopsize

def stft(x, binsize=1024, overlap_factor=.5, hopsize=None, window='hamming', **kwargs):
    """ STFT, Short-Term Fourier Transform.

    Parameters:
    -----------
    x: 2d-ndarray, (n_ch, n_samp)
        Multi-channel signal.

    binsize: int
        Window size for processing FFT on.

    overlap_factor: float
        The percentage of overlapping between consecutive windows.

    hopsize: int
        The sample size required to jump to the next row.

    window: str (default: 'hamming')
        The window used to create overlapping slices of the time domain signal.

    kwargs:
        The key-word arguments for rfft.

    Return:
    -------
    X: ndarray, (n_ch, n_win, binsize // 2)
    """
    # Sanity check
    if not np.isrealobj(x):
        raise ValueError("x is not a real valued array")

    if x.ndim > 2:
        raise ValueError("The dimension of the ndarray must be less than or equal to 2!")

    x = np.atleast_2d(x)
    n_ch, n_samp = x.shape

    if hopsize is not None:
        _overlap_factor = hopsize / binsize
        if overlap_factor != _overlap_factor:
            raise ValueError("The 'overlap_factor' calculated from hopsize/binsize does not match the input.")

    if overlap_factor in [0, 1] and binsize != hopsize != n_samp:
        binsize = n_samp
        hopsize = 0 if overlap_factor else binsize
    else:
        hopsize = int(binsize * (1 - overlap_factor)) if hopsize is None else hopsize

    if hopsize:
        n_win = n_samp / hopsize
        n_win = int(n_win + 1) if overlap_factor else int(n_win)
        length = hopsize
    else:
        n_win = 1
        length = binsize

    if overlap_factor in [0,1]:
        _x = np.zeros((n_ch, n_win * length))
    else:
        _x = np.zeros((n_ch, (n_win + 1) * length))

    _x[:,binsize//2:binsize//2+n_samp] = x

    # Process
    win_ = get_window(window, binsize)
    frames = stride_tricks.as_strided(_x, shape=(n_ch, n_win, binsize), strides=(_x.strides[0], _x.strides[1]*hopsize, _x.strides[1]))
    X = rfft(frames * win_, **kwargs)

    return X

def istft(X, nsamp=None, binsize=1024, overlap_factor=.5, hopsize=None):
    """ Inverse STFT.

    Parameters:
    -----------
    X: ndarray, (n_ch, n_win, binsize // 2) or (n_ch, n_win, n_fr, binsize // 2)

    binsize: int
        Window size for processing FFT on.

    overlap_factor: float
        The percentage of overlapping between consecutive windows.

    hopsize: int
        The sample size required to jump to the next row.

    Return:
    -------
    x: ndarray, (n_ch, n_fr, n_samp)
    """
    # Sanity check
    if X.ndim not in [3, 4]:
        raise ValueError("The dimension of 'X' is not valid as an output from stft! Double check 'X'.")

    if X.ndim == 3: # add a n_fr dimension
        X = X[:,:,np.newaxis,:]

    if X.shape[-1] != binsize:
        raise ValueError("The 'binsize' must match the length of X.shape[-1].")

    hopsize = binsize * (1 - overlap_factor) if hopsize is None else hopsize

    # Process
    x_ = irfft(X, n=binsize, axis=-1, planner_effort='FFTW_ESTIMATE')

    # Reconstructing the signal using overlap-add
    x = overlap_add(x_, binsize=binsize, overlap_factor=overlap_factor)

    # Clean up the signal
    if nsamp is not None:
        x = x[:,:,binsize//2:nsamp+binsize//2]

    return x
