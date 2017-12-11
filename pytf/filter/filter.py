
import numpy as np
import pyfftw.interfaces.numpy_fft as fft
from scipy.signal import firwin
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def create_filter(order, cutoff, nyquist, N, ftype='fir', output='freq', shift=True):
    """ Create a lowpass FIR filter. This function is meant to create only the prototype filter,
    where highpass, bandpass, or bandstop can all be transformed from the lowpass filter.

    Parameters:
    -----------
    order: int
        The filter order. The number of taps for an FIR filter.

    cutoff: float
        The cutoff frequency of the filter.

    nyquist: int or float
        This parameter is half the sampling rate.

    ftype: str
        Declare the filter to be an FIR ('fir') or an IIR ('iir').

    output: str
        Declare the return of the function to be in 'time' domain or 'freq' domain.

    shift: bool
        Declare if fftshift is applied to the FFT filter coeffiecients or not.

    Returns:
    --------
    If output is 'freq'
        w: ndarray
            Frequencies corresponding to frequency components. The units is the
            same as the ones chosen for the Nyquist rate.

        H: ndarray, complex
            The values of the FFT of the filter coefficients.

    if output is 'time'
        h: ndarray
            The values of the filter coefficients
    """
    if order > N:
        raise ValueError("The order of the filter should not be longer than the length for FFT (binsize).")

    if cutoff >= nyquist:
        raise ValueError("The cutoff frequency must be at least 2 times smaller than the Nyquist rate.")

    if output not in ['time', 'freq']:
        raise ValueError("'output' must be either 'time' or 'freq'!")

    h = firwin(order, cutoff, nyq=nyquist)

    if output == 'freq':
        w = fft.fftfreq(N)
        w *= (nyquist*2)

        H = fft.fft(h, n=N, axis=-1, planner_effort='FFTW_ESTIMATE')

        if shift:
            return fft.fftshift(w), fft.fftshift(H)
        else:
            return w, H

    elif output == 'time':
        return h

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

def get_frequency_of_interests(cf, bw):
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

def get_all_frequencies(center_frequencies=None, bandwidth=None, frequency_bands=None):
    """ Ensures all frequencies, fois, cf, and bw.
    """
    if (center_frequencies, bandwidth, frequency_bands) is (None, None, None):
        raise ValueError("Must enter one of the following arguments: 'cf', 'bw', 'fois.")

    if frequency_bands is None:
        frequency_bands = get_frequency_of_interests(center_frequencies, bandwidth)

    if center_frequencies is None or bandwidth is None:
        center_frequencies, bandwidth = get_center_frequencies(frequency_bands)

    return center_frequencies, bandwidth, frequency_bands
