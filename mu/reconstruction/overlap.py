import numpy as np
import six
from ..core import frame
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)
# def ola(x, binsize, hopsize, dtype):
#     result = np.zeros(())
#     return

def ola(x, binsize, hopsize, n=None):
    overlap_factor = binsize / hopsize

    nsamps = int((x.size//x.shape[0]) / overlap_factor + binsize)
    result_xi = np.zeros((x.shape[0], nsamps+binsize), dtype=x.dtype)
    r_ = frame(result_xi, binsize, hopsize)
    for i in range(x.shape[1]):
        r_[:,i, :] += x[:,i, :]

    if n is None:
        return result_xi
    else:
        return result_xi[:,:n]

def overlap_add(x, binsize, overlap_factor=.5, dtype=np.float32):

    _nfreqs = 1
    if x.ndim == 3:
        _nch, _nwin, _ = x.shape
    elif x.ndim == 4:
        _nch, _nwin, _nfreqs, _ = x.shape

    # Reconstructing the signal using overlap-add
    x_= np.zeros((_nch, _nfreqs, _nwin * binsize), dtype=dtype)
    for ix in range(_nwin):
        jx = (1-overlap_factor) * ix
        if int((jx+1)*binsize) <= binsize * _nwin * (1-overlap_factor):
            x_[:,:,int(jx*binsize):int((jx+1)*binsize)] += x[:,ix,:,:]

    return x_

# def overlap_add(x, binsize, overlap_factor=.5, dtype=np.float32):
#
#     _nfreqs = 1
#     if x.ndim == 3:
#         _nch, _nwin, _ = x.shape
#     elif x.ndim == 4:
#         _nch, _nwin, _nfreqs, _ = x.shape
#
#     # Reconstructing the signal using overlap-add
#     def _ola(y, x):
#         z = np.concatenate((y, np.zeros((_nch,_nfreqs,binsize//2))), axis=-1)
#         print(z.shape)
#         z[:,:,-binsize:] += x
#         return y
#
#     x = six.moves.reduce(_ola, [x[:,i,:,:] for i in range(_nwin)])
#     return x[:,:,:_nwin*binsize]
