import numpy as np
from scipy.signal import firwin
import pyfftw.interfaces.numpy_fft as fft
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

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

def reshape_data(data, axis=-1):

    if data.ndim > 2 or axis > 1:
        raise ValueError("The data should only be in 2 dimensional. \
                The support for 3 dimensional have not been implemented yet.")
    if data.ndim == 1:
        data = np.atleast_2d(data)

    if axis != 0:
        return data
    else:
        return data.T
