import numpy as np
from scipy.signal import firwin
import pyfftw.interfaces.numpy_fft as fft
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def get_center_frequencies(fois):
    """
    Convert an array of frequency bands into center frequencies and a bandwidth.
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
    if fois.shape[0] == 2:
        fois = fois.T

    cf = np.atleast_2d(fois.mean(axis=-1)).T
    bw = np.diff(fois, axis=-1)
    bw = bw[0] if np.diff(bw) else bw.mean()
    print(bw)
    return cf, float(bw)

def get_frequency_of_interests(cf, bw):
    """
    Convert an array of center frequencies and a bandwidth into an array of frequency bands.
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

def reshape_data(data):
    """
    Reshaping the data such that the data has the shape of (nch x nsamp).

    Parameters:
    -----------
    data: ndarray
        The data of interest. It can currenly be 1d or 2d ndarray.

    Returns:
    --------
    data: ndarray
        The reshaped data.
    """
    if data.ndim > 2:
        raise ValueError("The data should only be in 2 dimensional. \
                The support for 3 dimensional have not been implemented yet.")
    if data.ndim == 1:
        data = np.atleast_2d(data)

    if axis != 0:
        return data
    else:
        return data.T
