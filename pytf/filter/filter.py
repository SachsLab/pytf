
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
        return 1, h
