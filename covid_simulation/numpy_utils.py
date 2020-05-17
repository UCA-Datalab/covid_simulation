import numpy as np


def dt_array_to_str_list(arr):
    """
    Transform an array of datetimes to a list of str
    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    ls : list
    """
    assert isinstance(arr, np.ndarray), f"Type of `arr` is not np.ndarray, " \
                                    f"but {type(arr)}"
    ls = np.datetime_as_string(arr, unit="D").tolist()
    return ls
