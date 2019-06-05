import numpy as np
from numpy.lib.stride_tricks import as_strided

def frame(x, binsize, hopsize):
    """ Slice a time series into overlapping frames.

    Parameters:
    -----------
    x: 2d-ndarray, (n_ch, n_samp)
        Multi-channel signal.

    binsize: int
        Window size for processing FFT on.

    hopsize: int
        The sample size required to jump to the next row.

    Return:
    -------
    frames: ndarray, (n_ch, n_win, binsize / 2)
    """
    x = np.asarray(x)

    # Sanity check
    if x.ndim > 2:
        raise ValueError("The dimension of the ndarray must be less than or equal to 2! "
                             "Given x.ndim={}".format(x.ndim))
    else:
        x = np.atleast_2d(x)

    if hopsize < 1:
        raise ValueError('Invalid hopsize. Must be greater than 1.')

    nch, nsamps = x.shape

    # Compute the number of windows that will fit the length of the data.
    # The end may get truncated.
    nwin = 1 + int((nsamps - binsize) / hopsize)

    # Process
    return as_strided(
                x,
                shape=(nch, nwin, binsize),
                strides=(x.strides[0], x.strides[1]*hopsize, x.strides[1])
            )

def frames_to_samples(frames, hopsize):

    if hopsize < 1:
        raise ValueError('Invalid hopsize. Must be greater than 1.')

    if frames.ndim == 1:
        frames = np.atleast_2d(frames)
        nch, nwin = frames.shape
        binsize = 1
    elif frames.ndim == 2:
        frames = frames[np.newaxis,:,:] if frames.shape[0] != 1 else frames
        nch, nwin, binsize = frames.shape
    elif frames.ndim == 3:
        nch, nwin, binsize = frames.shape

    time_ix = np.arange(binsize)
    frame_ix = np.arange(nwin) * hopsize

    return np.array([time_ix, frame_ix], dtype=np.object)

def frames_to_time(frames, hopsize, srate):
    return frames_to_samples(frames, hopsize) / srate
