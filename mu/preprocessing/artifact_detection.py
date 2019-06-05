import numpy as np

from .utilities import group
from ..core import (frame, frames_to_samples)
# def artifact_burst_idx(x, condition, hopsize, duration=3):
#     # Create slices of indices from groups
#     _tmp = group(x, gap=hopsize)
#
#     prev_time = 0
#     idx_slices = []
#     for ix, _t in enumerate(_tmp):
#         tmp = (_t[0], _t[-1])
#
#         curr_time = tmp[0]
#         dur = np.abs(np.diff(tmp))[0] / srate
#         time_gap = (curr_time-prev_time) / srate
#         prev_time = tmp[-1]
#
#         if time_gap or not ix:
#             if np.logical_and(time_gap>0, time_gap < duration):
#                 tmp_ = idx_slices.pop()
#                 tmp = (tmp_.start, _t[-1])
#
#             idx_slices += [slice(*tmp)]
#
#     return idx_slices

def artifact_detect(x, cond, gap=1, replace_by_value=0, flag=0):
    """ Detect the artifact based on thresholding method. The thresholding condition is determined by 'cond'.
    The 'gap' specifies the duration between artifacts that are under the threshold but should be treated as artifact.
    """
    cond = list(np.where(cond)[0]) if cond.dtype == np.bool else list(cond)
    diff = np.diff(cond)
    for i, _gap in enumerate(diff):
        if _gap < gap:
            cond.extend(range(cond[i]+1, cond[i+1]))

    cond.sort()
    if flag:
        return np.asarray(cond)
    else:
        for i in cond:
            x[i,:] = replace_by_value
        return x

def segment_consecutive(x, step=100):
    """ Group data that has consecutive segments with increment < stepsize together.
    """
    return np.split(x, np.where(np.abs(np.diff(x)) >= step)[0]+1)

def segment_good_signals(x, idx_bad, good_dur=10, flag=0):
    """ Group good signals into a list.
    """
    idx = np.arange(x.shape[0])
    idx = np.delete(idx, idx_bad) # delete the indices that are artifacts
    idx_new = [ix for ix in segment_consecutive(idx, step=2) if len(ix) > good_dur]
    if flag:
        return idx_new
    else:
        return [x[ix] for ix in idx_new]

def segment_out_artifact(x, thres=None, timestamp=False):

    x_ = frame(np.atleast_2d(x), binsize, hopsize)[0,:,:]
    rms_ = rms_energy(x_).flatten()
    if thres is None:
        thres = rms_.std(axis=-1) * 3

    idx_ = frames_to_samples(rms_, hopsize)[-1]

    ix = artifact_detect(x_, rms_>thres, gap=20, flag=1)
    idxa = artifact_detect(0, idx_[ix], gap=1025, flag=1)

    seg2 = segment_good_signals(x, idxa, good_dur=4*srate, flag=0)
    out = seg2
    if timestamp:
        idx = np.arange(x.size)
        idx2 = segment_good_signals(idx, idxa, good_dur=4*srate, flag=0)

        out = (seg2, idx2)
    return out
