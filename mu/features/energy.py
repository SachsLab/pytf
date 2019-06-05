import numpy as np
from ..core import (frame, frames_to_samples)

def rms_energy(x, axis=-1):
    return np.sqrt(np.nanmean(x**2, axis=axis))
