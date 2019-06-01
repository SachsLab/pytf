import numpy as np
from .utilities import group

def artifact_burst_idx(x, condition, hopsize, duration=3):
    # Create slices of indices from groups
    _tmp = group(x, gap=hopsize)

    prev_time = 0
    idx_slices = []
    for ix, _t in enumerate(_tmp):
        tmp = (_t[0], _t[-1])

        curr_time = tmp[0]
        dur = np.abs(np.diff(tmp))[0] / srate
        time_gap = (curr_time-prev_time) / srate
        prev_time = tmp[-1]

        if time_gap or not ix:
            if np.logical_and(time_gap>0, time_gap < duration):
                tmp_ = idx_slices.pop()
                tmp = (tmp_.start, _t[-1])

            idx_slices += [slice(*tmp)]

    return idx_slices
