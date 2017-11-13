from scipy.signal import get_window
from numpy.lib import stride_tricks
import numpy as np
from pyfftw.interfaces.numpy_fft import (rfft, irfft, ifft, fftfreq)

def create_time_idx(n_samp, args, create_indices=False, meshgrid=False):
    """
    All arguments provided to this function are related to each other for creating
    overlapping window of a given signal with the length 'nsamp'. This function takes
    the input argument 'args', and update their values. The actual indices for creating
    the window based on 'args' can be returned if 'create_indices = True'.

    Parameters:
    -----------
    n_samp: int
        The length of the total analysis sample size.

    args: list
        This is a list object containing the following arguments:
            binsize: int
                The length of chunk size.

            overlap_factor: float
                The ratio of overlapping between chuncks.

            hopsize: int
                The number of sample it takes to hop between rows.

    create_indices: bool
        Flag for returning indices.

    Return:
    -------
    indices (optional)
    """

    binsize, overlap_factor, hopsize, n_win = args
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

    args[0] = binsize
    args[1] = overlap_factor
    args[2] = hopsize
    args[3] = n_win

    if create_indices:
        if meshgrid:
            x = np.arange(0, binsize)
            y = np.arange(0, n_win)
            index1, index2 = np.meshgrid(x, y)
            index1 += np.atleast_2d(hopsize*np.arange(n_win)).T

            idx1 = np.asarray(index1, dtype=np.int64)
            idx2 = np.asarray(index2, dtype=np.int64)
            return idx1, idx2

        else:
            if overlap_factor in [0,1]:
                idx = np.arange(n_win * length)
            else:
                idx = np.arange((n_win + 1) * length)
            return stride_tricks.as_strided(idx, shape=(n_win, binsize), strides=(idx.strides[0]*hopsize, idx.strides[0])).copy()

def stft(x, win_idx=None, binsize=1024, overlap_factor=.5, hopsize=None, window='hamming', **kwargs):
    """
    STFT, Short-Term Fourier Transform.

    Parameters:
    -----------
    x: 2d-ndarray, (n_ch x n_samp)
        Multi-channel signal.

    win_idx: default None
        Create the indices of overlapping window.

    binsize: int
        Window size for processing FFT on.

    overlap_factor: float
        The percentage of overlapping between consecutive windows.

    Return:
    -------
    X: ndarray, (n_ch x n_win x binsize // 2)
    """
    if not np.isrealobj(x):
        raise ValueError("x is not a real valued array")

    if x.ndim == 1:
        x = np.atleast_2d(x)

    n_ch, n_samp = x.shape

    n_win = None

    binsize = binsize if win_idx is None else win_idx.shape[-1]
    hopsize = hopsize if win_idx is None else win_idx[1,0]

    args = [binsize, overlap_factor, hopsize, n_win]
    if win_idx is None:
        win_idx = create_time_idx(n_samp, args, create_indices=True)
    else:
        create_time_idx(n_samp, args)

    binsize, overlap_factor, hopsize, n_win = args

    win_ = get_window(window, binsize)

    length = hopsize if hopsize else binsize
    if overlap_factor in [0,1]:
        _x = np.zeros((n_ch, n_win * length))
    else:
        _x = np.zeros((n_ch, (n_win + 1) * length))

    _x[:,binsize//2:binsize//2+n_samp] = x

    X = rfft(_x[:, win_idx] * win_, **kwargs)

    return X

def istft(X, nsamp=None, win_idx=None, binsize=1024, overlap_factor=.5, hopsize=None, nfreqs=1, window='hamming', **kwargs):
    """
    Inverse STFT.

    Parameters:
    -----------
    X: ndarray, (n_ch x n_win x binsize // 2) or (n_ch x n_win x n_fr x binsize // 2)

    Return:
    -------
    x: ndarray, (n_ch x n_fr x n_samp)
    """

    if X.ndim == 3: # add a n_fr dimension
        X = X[:,:,np.newaxis,:]

    n_win = None

    if win_idx is None:
        binsize = binsize
        hopsize = binsize * (1 - overlap_factor) if hopsize is None else hopsize
    else:
        binsize = win_idx.shape[-1]
        hopsize = win_idx[1,0]
        overlap_factor = hopsize / binsize

    x_ = irfft(X, n=binsize, axis=-1, planner_effort='FFTW_ESTIMATE')

    _nch, _nwin, _nfreqs, _bin = x_.shape

    # Reconstructing the signal using overlap-add
    x = np.zeros((_nch, _nfreqs, _nwin * binsize))
    overlap_factor = .5
    for ix in range(_nwin):
        jx = (1-overlap_factor) * ix
        if int((jx+1)*binsize) <= binsize * _nwin * (1-overlap_factor):
            x[:,:,int(jx*binsize):int((jx+1)*binsize)] += x_[:,ix,:,:]

    if nsamp is not None:
        x = x[:,:,binsize//2:nsamp+binsize//2]

    return x
