import numpy as np

def group(x, gap=1):
    """ Group elements into subgroups based on the gap. The gap specifies the difference between two neighboring elements.
    """
    _diff = np.diff(x)

    _sub = []
    _group = []
    for i, _d in enumerate(_diff):
        _sub += [x[i]]
        if _d != gap:
            _group += [_sub]
            _sub = []

    if _diff[-1] == gap: # this allows the program to group the end part of the squence
        _sub += [_sub[-1]+gap]
        _group += [_sub]

    return _group
