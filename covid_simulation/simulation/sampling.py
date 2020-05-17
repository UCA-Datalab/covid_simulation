'''
Functions that sample from distributions.
* noise_on_cases : sample the number of cases from a discrete distribution
* _sample_prob_uci_state : sample which people need and don't need uci
* propagate_times_hospital : sample length of stay in
                             hospital, hospital to uci and uci
'''

import numpy as np

from covid_simulation.config import PARAMS_PROB_UCI
from covid_simulation.simulation.distributions import distribution_daily_cases
from covid_simulation.simulation.distributions import hosp_after_uci_los
from covid_simulation.simulation.distributions import hosp_los
from covid_simulation.simulation.distributions import hosp_to_uci_los
from covid_simulation.simulation.distributions import uci_los

from covid_simulation.simulation.utils import create_masked_arr
from covid_simulation.simulation.utils import union_masked_arrays


def noise_on_cases(cum, nruns=None, seed=0):
    """
    Calculate cumulative cases with noise on daily cases.
    If nruns is not None, then broadcasts cum to axis=1.

    Parameters
    ----------
    cum : array-like
        shape : (ndays,)
        Vector with cumulative cases per day
    nruns : int or None, default=None
        Number of runs to sample.
        If None, then the shape is maintained
    seed : int, default=0
        Seed for numpy.random

    Returns
    -------
    cum_noise : numpy.array
        shape : (ndays, nruns)
        Array with cumulative cases per day, adding noise
    """
    cum = np.asarray(cum)
    np.random.seed(seed)
    if nruns is not None:
        assert isinstance(nruns, int)
        assert nruns > 0
        assert cum.ndim == 1

    # daily cases, keeping in 0 the first value
    daily = np.diff(cum, axis=0, prepend=0)
    if cum.ndim == 1:
        daily = np.expand_dims(daily, 1)
        daily = np.repeat(daily, nruns, axis=1)

    daily_noise = distribution_daily_cases(daily, seed=seed)

    # correct the first day, since it is not random
    daily_noise[0, :] = cum[0]
    cum_noise = daily_noise.cumsum(axis=0)
    if nruns is None:
        assert cum_noise.shape == cum.shape
    else:
        assert cum_noise.shape == (cum.shape[0], nruns)
    return cum_noise


def noise_on_times(times, seed=0):
    """
    Calculate noise to the time value, making it continuos.

    Parameters
    ----------
    times : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    seed : int, default=0
        Seed for numpy.random

    Returns
    -------
    times_noise : numpy.array
        shape : (max_people, nruns)
        Array with entrance times of each person, adding noise
    """
    times = np.asarray(times)
    np.random.seed(seed)
    times_noise = times + np.random.uniform(low=-1, high=0., size=times.shape)

    assert times.shape == times_noise.shape
    return times_noise


def _sample_prob_uci_state(times, p_uci, p_uci_now_given_uci, seed=0):
    """
    Sample cases and return masks of people into three categories:
    * uci_now : They are entering directly into the UCI
    * uci_later : They will stay in the normal beds, and later enter into UCI
    * no_uci : They will never pass through UCI

    Parameters
    ----------
    times : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    p_uci : float
        Probability for a peoson to go through the UCI
    p_uci : float
        Probability for a person to enter directly into the UCI,
        given that it goes through the UCI
    seed : int, default=0
        Seed for numpy.random

    Returns
    -------
    mask_uci_now : numpy.array
        shape : (max_people, nruns)
        Mask selecting uci_now cases on `times`
    mask_uci_later : numpy.array
        shape : (max_people, nruns)
        Mask selecting uci_later cases on `times`
    mask_no_uci : numpy.array
        shape : (max_people, nruns)
        Mask selecting no_uci cases on `times`
    """
    np.random.seed(seed)

    p_uci_now = p_uci_now_given_uci * p_uci
    p_uci_later = p_uci * (1 - p_uci_now_given_uci)
    p_no_uci = 1 - p_uci

    state = np.random.multinomial(1, [p_uci_now, p_uci_later, p_no_uci],
                                  size=times.shape)
    mask_finite = np.isfinite(times)
    mask_uci_now = np.logical_and((state[:, :, 0] == 1), mask_finite)
    mask_uci_later = np.logical_and((state[:, :, 1] == 1), mask_finite)
    mask_no_uci = np.logical_and((state[:, :, 2] == 1), mask_finite)

    assert times.shape == mask_uci_now.shape
    assert times.shape == mask_uci_later.shape
    assert times.shape == mask_no_uci.shape
    return mask_uci_now, mask_uci_later, mask_no_uci


def propagate_times_hospital(times, p_uci=None,
                             p_uci_now_given_uci=None,
                             seed=0):
    """
    Sample time spent into each step (hospital or uci), and returns the times
    in which each patient enters and leaves hospital and/or uci.
    No saturation is taken into account, i.e., the maximum number of beds
    in each category is infinity.

    **TODO:** work always with the masks, so we dont have to use masked arrays
    # this is, arr[mask] = arr[mask]+arr2[mask]

    Parameters
    ----------
    times : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    p_uci : float or None, default=None
        Probability of going to UCI from hospital.
        If None, it uses values from PARAMS_PROB_UCI
    p_uci_now_given_uci : float or None, default=None
        Probability of going to UCI directly, given you go to UCI.
        If None, it uses values from PARAMS_PROB_UCI
    seed : int, default=0
        Seed for numpy.random

    Returns
    -------
    times_hosp_in : numpy.array
        shape : (max_people, nruns)
        Array with entrance time in hospital, nan if never enters in hospital.
        Each index represents a person
    times_hosp_out : numpy.array
        shape : (max_people, nruns)
        Array with leaving time in hospital, nan if never enters in hospital.
        Each index represents a person
    times_uci_in : numpy.array
        shape : (max_people, nruns)
        Array with entrance time in uci, nan if never enters in uci.
        Each index represents a person
    times_uci_out : numpy.array
        shape : (max_people, nruns)
        Array with leaving time in uci, nan if never enters in uci.
        Each index represents a person

    Deleted Parameters
    ------------------
    mask_uci_now : numpy.array
        shape : (max_people, nruns)
        Mask selecting uci_now cases on `times`
    mask_uci_later : numpy.array
        shape : (max_people, nruns)
        Mask selecting uci_later cases on `times`
    mask_no_uci : numpy.array
        shape : (max_people, nruns)
        Mask selecting no_uci cases on `times`
    """
    if p_uci is None:
        p_uci = PARAMS_PROB_UCI['p_uci']
    if p_uci_now_given_uci is None:
        p_uci_now_given_uci = PARAMS_PROB_UCI['p_uci_now_given_uci']

    mask_uci_now, mask_uci_later, mask_no_uci = _sample_prob_uci_state(
        times, p_uci=p_uci, p_uci_now_given_uci=p_uci_now_given_uci, seed=seed)

    # times entering in hospital
    times_hosp_no_uci_in = create_masked_arr(times, mask_no_uci)
    times_uci_later_hosp_in = create_masked_arr(times, mask_uci_later)

    # calculate exit time from hospital people no uci
    times_hosp_no_uci_out = times_hosp_no_uci_in + hosp_los(
        size=times_hosp_no_uci_in.shape, seed=seed)
    # exit time hospital and entering time uci, people need uci later
    times_uci_later_hosp_out = times_uci_later_hosp_in + hosp_to_uci_los(
        size=times_uci_later_hosp_in.shape, seed=seed)

    # exit time uci, we union people entering uci now and later
    times_uci_now_in = create_masked_arr(times, mask_uci_now)
    times_uci_in = union_masked_arrays(times_uci_later_hosp_out,
                                       times_uci_now_in)

    times_uci_out = times_uci_in + uci_los(size=times_uci_in.shape, seed=seed)

    times_hosp_in = union_masked_arrays(times_hosp_no_uci_in,
                                        times_uci_later_hosp_in)
    times_hosp_out = union_masked_arrays(times_hosp_no_uci_out,
                                         times_uci_later_hosp_out)

    # the index is the user, so the nans must be the same along time
    assert (np.isfinite(times_hosp_in) == np.isfinite(times_hosp_out)).all
    assert (np.isfinite(times_uci_in) == np.isfinite(times_uci_out)).all

    assert times.shape == times_hosp_in.shape
    assert times.shape == times_hosp_out.shape
    assert times.shape == times_uci_in.shape
    assert times.shape == times_uci_out.shape
    return times_hosp_in, times_hosp_out, times_uci_in, times_uci_out


def propagate_times_hospital_after_uci(times_uci_in, times_uci_out, seed=0):
    """
    Sample time spent in hospital for patients who have left the UCI.
    No saturation is taken into account, i.e., the maximum number of beds
    in each category is infinity.

    **TODO:** work always with the masks, so we dont have to use masked arrays
    # this is, arr[mask] = arr[mask]+arr2[mask]

    Parameters
    ----------
    times_hosp2_in : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    seed : int, default=0
        Seed for numpy.random

    Returns
    -------
    times_hosp2_out : numpy.array
        shape : (max_people, nruns)
        Array with leaving time in hospital, nan if never enters in hospital.
        Each index represents a person
    """
    times_hosp2_out = times_uci_out + \
        hosp_after_uci_los(times_uci_out - times_uci_in, seed=seed)
    assert (np.isfinite(times_hosp2_out) == np.isfinite(times_uci_out)).all
    return times_hosp2_out
