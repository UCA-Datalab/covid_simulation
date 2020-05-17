import numpy as np
import warnings


def create_masked_arr(array, mask_no_nan):
    """
    Create a copy of array, with nan's outside mask_no_nan.
    """
    fill_val = np.nan
    assert array.shape == mask_no_nan.shape
    array_masked = np.ones_like(array) * fill_val
    array_masked[mask_no_nan] = array[mask_no_nan]
    assert array.shape == array_masked.shape
    return array_masked


def union_masked_arrays(array_masked_1, array_masked_2):
    """
    Union of not-nan values of two masked arrays.
    """
    assert array_masked_1.shape == array_masked_2.shape
    array_masked = array_masked_1.copy()
    # substitute nans in 1 with values in 2
    mask_nans = np.isnan(array_masked)
    array_masked[mask_nans] = array_masked_2[mask_nans]
    assert array_masked.shape == array_masked_2.shape
    return array_masked


def from_times_to_counts(x, times):
    """
    Count cumulative number of cases with times <= to x, for each day in x

    Parameters
    ==========
    x : array-like
        shape : (ndays, )
        Array of days where to measure values
    times : numpy.array
        shape : (max_people, nruns)
        Array with entrance times of each person

    Returns
    -------
    cumulative_cases : np.array
        shape : (ndays, nruns)
        Cumulatine number of cases with times <= x
    """
    x = np.asarray(x)
    assert x.ndim == 1
    times = np.asarray(times)
    # reshape for broadcasting (n_days, max_people, nruns)
    x_bc = x.reshape([-1] + [1] * times.ndim)
    times_bc = np.expand_dims(times, 0)
    # There could be nans
    with warnings.catch_warnings():
        # RuntimeWarning: invalid value encountered in less_equal
        warnings.filterwarnings('ignore',
                                r'invalid value encountered in less_equal',
                                category=RuntimeWarning)
        # aggregate over max_people axis
        cumulative_cases = (times_bc <= x_bc).sum(axis=1)
    if times.ndim == 1:
        assert cumulative_cases.shape == x.shape
    else:
        assert cumulative_cases.shape == (x.shape[0], times.shape[1])
    return cumulative_cases


def _from_counts_to_times_1d(x, cum):
    """
    Generate array with one entry per event, i.e., each entrance
    is the time at which a person is entering.

    Parameters
    ----------
    x : array-like
        shape : (ndays,)
        Array of days where to measure values
    cum : array-like
        shape : (ndays,)
        Array with cumulative cases per day

    Returns
    -------
    times : numpy.array
        shape : (max_people,)
        Array with entrance times of each person
    """
    x = np.asarray(x)
    cum = np.asarray(cum)
    assert x.ndim == 1
    assert cum.ndim == 1

    times = np.ones(cum[-1], dtype='float')
    left = 0
    for day_idx, right in zip(x, cum):
        times[left:right] = day_idx
        left = right
    assert times.ndim == 1
    return times


def from_counts_to_times(x, cum):
    """
    Generate array with one entry per event, i.e., each entrance
    is the time at which a person is entering.

    Parameters
    ----------
    x : array-like
        shape : (ndays, )
        Array of days where to measure values
    cum : numpy.array
        shape : (ndays,nruns)
        Array with cumulative cases per day

    Returns
    -------
    times : numpy.array
        shape : (max_people, nruns)
        Array with entrance times of each person
    """
    x = np.asarray(x)
    assert x.ndim == 1

    if cum.ndim >= 2:
        # we need to know the maximum number of people for each run
        max_people = cum[-1].max()
        shape_output = list(cum.shape)
        shape_output[0] = max_people
        times = np.ones(shape=shape_output, dtype='float') * np.nan
        for i_run in range(cum.shape[1]):
            cum_run = cum[:, i_run]
            left = 0
            for day_idx, right in zip(x, cum_run):
                times[left:right, i_run] = day_idx
                left = right
    else:
        times = _from_counts_to_times_1d(x, cum)

    assert times.shape[1] == cum.shape[1]
    return times
