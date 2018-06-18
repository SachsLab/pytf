import numpy as np
try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import scipy.fftpack as fft

def logscale_normalization(spectra, srate=1., factor=20.):
    t_bins, f_bins = spectra.shape

    scale = np.linspace(0, 1, f_bins) ** factor
    scale *= (f_bins-1) / scale.max()
    scale = np.asarray(np.unique(np.round(scale)), dtype=np.int64)

    # create spectrogram with new freq bins
    spectra_ = np.zeros((t_bins, scale.size), dtype=np.complex128)
    for i in range(0, scale.size):
        if i < scale.size-1:
            spectra_[:,i] = np.sum(spectra[:,scale[i]:scale[i+1]], axis=1)
        else:
            spectra_[:,i] = np.sum(spectra[:,scale[i]:], axis=1)

    # list center freq of bins
    allfreqs = np.abs(fft.fftfreq(f_bins*2, 1./srate)[:f_bins+1])

    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return spectra_, np.asarray(freqs)
