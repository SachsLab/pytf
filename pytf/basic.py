import numpy as np
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def reshape_data(data):
    """
    Reshaping the data such that the data has the shape of (nch x nsamp).

    Parameters:
    -----------
    data: ndarray
        The data of interest. It can currenly be 1d or 2d ndarray.

    Returns:
    --------
    data: ndarray
        The reshaped data.
    """
    if data.ndim > 2:
        raise ValueError("The data should only be in 2 dimensional. \
                The support for 3 dimensional have not been implemented yet.")
    if data.ndim == 1:
        data = np.atleast_2d(data)

    if axis != 0:
        return data
    else:
        return data.T
