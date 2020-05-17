'''
Simulation of the place filling procedure, using as inputs the a priori
times of arrival and leaving, and checking if the place is saturated or
'''
import numpy as np
import warnings

from covid_simulation.simulation.utils import from_times_to_counts


def apply_maxcapacity_constraint_1d(times_in, times_out,
                                    n0=0, nmax=None,
                                    tmax=None, fill_constrained=np.nan):
    """
    Correct times of entering and leaving applying saturation
    effects, i.e., that the room has a maximum cacity of nmax.

    *TODO:* Try to do the simulation more efficient.
    *TODO:* For consecutive events, like a person trying to enter
    twice in the room (before and after UCI), we need to generalize the
    algorithm to have 4 lists, and concatenate them since we have
    4 kinds of events (all connected by an index.)

    Parameters
    ----------
    times_in : array-like
        shape : (max_people, )
        Array with entrance times of each person
    times_out : array-like
        shape : (max_people, )
        Array with leaving times of each person
    n0 : int, default=0
        People already in the room at the beginning
    nmax : int or None, default=None
        Maximum number of people
    tmax : int or None, default=None
        Stop time for the algorithm, for efficiency reasons.
        We can neglect future times to save computational time
    fill_constrained : float, default:np.nan
        Value to fill with people who cannot enter.

    Returns
    -------
    times_constrained_in : array-like
        shape : (max_people, )
        Array with entrance times of each person, corrected
    times_constrained_out : array-like
        shape : (max_people, )
        Array with leaving times of each person, corrected
    """
    times_in = np.asarray(times_in)
    times_out = np.asarray(times_out)
    assert times_in.ndim == 1, '`times_in` must be 1 dimensional'
    assert times_in.shape == times_out.shape, '`times_in` and `times_out` must have same shape'
    assert np.isfinite(times_in).sum() == np.isfinite(times_out).sum()

    npeople = np.isfinite(times_in).sum()

    if nmax is None or (npeople + n0 <= nmax):
        return times_in, times_out
    else:
        assert n0 <= nmax

    npeople = times_in.size

    # we use two vectors to identify each event
    # event_times : time of event, mask_out : True if it goes out
    event_times = np.concatenate((times_in, times_out), axis=0)
    mask_out = np.ones(shape=(npeople * 2), dtype='bool')
    mask_out[:npeople] = False

    # we loop over the events in time order
    idx_sorted = np.argsort(event_times)

    # initialize variables
    nbeds = n0
    times_constrained_in = times_in.copy()
    times_constrained_out = times_out.copy()
    idx_person_no_enter = set()
    for idx_event, is_out, time in zip(idx_sorted, mask_out[idx_sorted],
                                       event_times[idx_sorted]):
        # we can neglect future times to save computational time
        if (tmax is not None) and (time > tmax):
            break
        # translate event index into person index
        if is_out:
            idx_person = idx_event - npeople
        else:
            idx_person = idx_event
        # want to enter
        if not is_out:
            # enters
            if nbeds < nmax:
                nbeds += 1
            # not enters
            else:
                idx_person_no_enter.add(idx_person)
                times_constrained_in[idx_person] = fill_constrained
                times_constrained_out[idx_person] = fill_constrained
        # wants to leave
        else:
            # if a person has not entered, it cannot leave
            if idx_person in idx_person_no_enter:
                continue
            else:
                nbeds -= 1

    assert times_constrained_in.shape == times_in.shape
    assert times_constrained_out.shape == times_out.shape
    # the number of people must be the same for entering and leaving
    assert np.isfinite(times_constrained_in).sum() == np.isfinite(
        times_constrained_out).sum()
    # after applying the constraing, less or equal people can enter
    n_original = np.isfinite(times_in).sum()
    n_constrained = np.isfinite(times_constrained_in).sum()
    assert n_original >= n_constrained
    return times_constrained_in, times_constrained_out


def apply_maxcapacity_constraint(times_in, times_out,
                                 n0=0, nmax=None,
                                 tmin=None, tmax=None, fill_constrained=np.nan,
                                 try_heuristic=True, verbose=False):
    """
    Correct times of entering and leaving applying saturation
    effects, i.e., that the room has a maximum cacity of nmax.

    *TODO:* Try to do the simulation more efficient.

    Parameters
    ----------
    times_in : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    times_out : array-like
        shape : (max_people, nruns)
        Array with leaving times of each person
    n0 : int, default=0
        People already in the room at the beginning
    nmax : int or None, default=None
        Maximum number of people
    tmin : float or None, default=None
        Starting time for the algorithm, for efficiency reasons.
        **TODO:** check its use
    tmax : float or None, default=None
        Stop time for the algorithm, for efficiency reasons.
        We can neglect future times to save computational time
    fill_constrained : float, default:np.nan
        Value to fill with people who cannot enter.

    Returns
    -------
    times_constrained_in : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person, corrected
    times_constrained_out : array-like
        shape : (max_people, nruns)
        Array with leaving times of each person, corrected
    """
    assert times_in.shape == times_out.shape
    npeople = np.isfinite(times_in).sum(axis=0).max()
    nruns = times_in.shape[1]
    # if nmax is None, then there is no constraint
    if nmax is None or (npeople + n0 <= nmax):
        times_constrained_in, times_constrained_out = times_in, times_out
        return times_constrained_in, times_constrained_out
    # time interval to check for saturation in minutes
    sat_check_freq = 15
    if tmax is not None:
        # maximum time we are interested in
        times_in = times_in.copy()
        times_out = times_out.copy()
        with warnings.catch_warnings():
            # RuntimeWarning: invalid value encountered in greater
            warnings.filterwarnings('ignore',
                                    r'invalid value encountered in greater',
                                    category=RuntimeWarning)
            mask_future_in = times_in > tmax

        times_in[mask_future_in] = np.inf
        times_out[mask_future_in] = np.inf

    if try_heuristic:
        # we check that the number of active cases is smaller than max_beds
        if tmin is not None:
            min_time = max((x.min(), times_in[np.isfinite(times_in)].min()))
        else:
            min_time = times_in[np.isfinite(times_in)].min()
        if tmax is not None:
            max_time = min((tmax, times_out[np.isfinite(times_out)].max()))
        else:
            max_time = times_out[np.isfinite(times_out)].max()

        assert max_time > min_time
        npoints_grid = (max_time - min_time) * 24 * (sat_check_freq / 60)
        x_grid = np.linspace(min_time,
                             max_time,
                             num=int(npoints_grid),
                             dtype='float')
        actives_grid = from_times_to_counts(
            x_grid, times_in) - from_times_to_counts(x_grid, times_out)
        saturated_runs_mask = actives_grid.max(axis=0) > nmax
        if verbose:
            nsatruns = saturated_runs_mask.sum()
            nsatmean = 100 * saturated_runs_mask.mean()
            print(f'Number of saturated runs: {nsatruns} ({nsatmean:.2f}%)')
    # assume there will be saturation in all runs
    else:
        saturated_runs_mask = np.ones((nruns, ), dtype='bool')

    # no one saturated -> all vectorized
    if np.alltrue(~saturated_runs_mask):
        times_constrained_in, times_constrained_out = times_in, times_out
        return times_constrained_in, times_constrained_out

    else:
        saturated_runs_indices = np.nonzero(saturated_runs_mask)[0]
        non_saturated_runs_indices = np.nonzero(~saturated_runs_mask)[0]
        list_tcons_in = [None] * nruns
        list_tcons_out = [None] * nruns

        # vectorize the non-saturated
        if saturated_runs_mask.mean() < 1:
            if verbose:
                print(f'Non saturated indices: {non_saturated_runs_indices}')
            # in order to use zip, we move run idx to first dimension
            times_in_not_sat = times_in[
                :, non_saturated_runs_indices].swapaxes(0, 1)
            times_out_not_sat = times_out[
                :, non_saturated_runs_indices].swapaxes(0, 1)

            # fill them into the list with the correct run index
            for i_run, vec_in_nosat, vec_out_nosat in zip(non_saturated_runs_indices,
                                                          times_in_not_sat, times_out_not_sat):
                list_tcons_in[i_run] = vec_in_nosat
                list_tcons_out[i_run] = vec_out_nosat

        # if verbose and can import tqdm, then show progress bar
        saturated_runs_indices_ = saturated_runs_indices
        if verbose:
            try:
                from tqdm import tqdm
                saturated_runs_indices_ = tqdm(saturated_runs_indices)
            except ModuleNotFoundError:
                pass

        # non-saturated cannot be vectorized
        for i_run in saturated_runs_indices_:
            times_in_1d = times_in[:, i_run]
            times_out_1d = times_out[:, i_run]
            times_in_1d_cons, times_out_1d_cons = apply_maxcapacity_constraint_1d(times_in_1d, times_out_1d,
                                                                                  n0=n0, nmax=nmax,
                                                                                  tmax=tmax, fill_constrained=fill_constrained)
            list_tcons_in[i_run] = times_in_1d_cons
            list_tcons_out[i_run] = times_out_1d_cons

        times_constrained_in = np.swapaxes(np.array(list_tcons_in), 0, 1)
        times_constrained_out = np.swapaxes(np.array(list_tcons_out), 0, 1)

        return times_constrained_in, times_constrained_out


def simulate_bed_filling_1d(x,
                            times_in,
                            times_out,
                            beds_0=0,
                            beds_max=None,
                            try_heuristic=False,
                            sat_check_freq=15,
                            verbose=False):
    """
    Simulate the people entering and leaving a place,
    taking into account that there is a maximum capacity.
    Retuns the cumulative cases entering, leaving
    and the number of active cases.
    Version for 1 dimensional times_in and times_out.

    Caution: the condition is checked every 15 minutes to save resources

    Parameters
    ----------
    x : array-like
        shape : (ndays, )
        Array of days where to measure values
    times_in : array-like
        shape : (max_people, )
        Array with entrance times of each person
    times_out : array-like
        shape : (max_people, )
        Array with leaving times of each person
    beds_0 : int, default=0
        Beds occupied at the beginning
    beds_max : int or None, default=None
        Maximum number of beds
    try_heuristic : bool, default=False
        Check if saturation happens a priori, if it is not the case
        the simulation is vectorized
    sat_check_freq : int or float, default=15
        Frequency to check for saturation in minutes.
        The smaller the more time and memory consumes
    verbose : bool, default=False
        Print messages

    Returns
    -------
    cum_in : numpy.array
        shape : (ndays, )
        Cumulative number of people entering
    cum_out : numpy.array
        shape : (ndays, )
        Cumulative number of people leaving
    actives : numpy.array
        shape : (ndays, )
        Number of people occuping bed at that day: cum_in-cum_out
    """
    x = np.asarray(x)
    assert x.ndim == 1
    assert times_in.ndim == 1
    assert times_out.ndim == 1
    assert times_in.shape == times_out.shape
    times_in = times_in.copy()
    times_out = times_out.copy()

    assert times_in[np.isfinite(times_in)].min() < times_out[
        np.isfinite(times_out)].min()
    assert times_in[np.isfinite(times_in)].max() < times_out[
        np.isfinite(times_out)].max()

    npeople = np.isfinite(times_in).sum()

    # maximum time we are interested in
    tmax = x.max()
    with warnings.catch_warnings():
        # RuntimeWarning: invalid value encountered in greater
        warnings.filterwarnings('ignore',
                                r'invalid value encountered in greater',
                                category=RuntimeWarning)
        mask_future_in = times_in > tmax
    times_in[mask_future_in] = np.inf
    times_out[mask_future_in] = np.inf

    # rm nans and infinites, i.e., people we are not interested in
    mask_finite = np.isfinite(times_in)
    times_in = times_in[mask_finite]
    times_out = times_out[mask_finite]

    if (beds_max is None) or (npeople + beds_0 <= beds_max):
        cum_in = from_times_to_counts(x, times_in)
        cum_out = from_times_to_counts(x, times_out)
        actives = cum_in - cum_out

        return cum_in, cum_out, actives

    if try_heuristic:
        # we check that the number of active cases is smaller than max_beds
        min_time = max((x.min(), times_in[np.isfinite(times_in)].min()))
        max_time = min((x.max(), times_out[np.isfinite(times_out)].max()))
        assert max_time > min_time
        npoints_grid = (max_time - min_time) * 24 * (sat_check_freq / 60)
        x_grid = np.linspace(min_time,
                             max_time,
                             num=int(npoints_grid),
                             dtype='float')

        actives_grid = from_times_to_counts(
            x_grid, times_in) - from_times_to_counts(x_grid, times_out)
        if actives_grid.max() <= beds_max:
            if verbose:
                print('Time dimension vectorization is possible')
            cum_in = from_times_to_counts(x, times_in)
            cum_out = from_times_to_counts(x, times_out)
            actives = cum_in - cum_out

            assert actives.max() <= beds_max
            assert cum_in.shape == cum_out.shape
            assert cum_in.shape == actives.shape
            assert cum_in.shape == x.shape
            return cum_in, cum_out, actives

    if verbose:
        print('Time dimension vectorization is not possible')

    times_real_in, times_real_out = apply_maxcapacity_constraint_1d(
        times_in, times_out, n0=beds_0, nmax=beds_max)

    cum_in = from_times_to_counts(x, times_real_in)
    cum_out = from_times_to_counts(x, times_real_out)
    actives = cum_in - cum_out

    assert actives.max() <= beds_max
    assert cum_in.shape == cum_out.shape
    assert cum_in.shape == actives.shape
    assert cum_in.shape == x.shape
    return cum_in, cum_out, actives


def simulate_bed_filling(x,
                         times_in,
                         times_out,
                         beds_0=0,
                         beds_max=None,
                         try_heuristic=False,
                         sat_check_freq=15,
                         verbose=False):
    """
    Simulate the people entering and leaving a place,
    taking into account that there is a maximum capacity.
    Retuns the cumulative cases entering, leaving
    and the number of active cases.

    Caution: the saturation condition is checked every 15 minutes
    to save computational resources.

    Parameters
    ----------
    x : array-like
        shape : (ndays, )
        Array of days where to measure values
    times_in : array-like
        shape : (max_people, nruns)
        Array with entrance times of each person
    times_out : array-like
        shape : (max_people, nruns)
        Array with leaving times of each person
    beds_0 : int, default=0
        Beds occupied at the beginning
    beds_max : int or None, default=None
        Maximum number of beds
    try_heuristic : bool, default=False
        Check if saturation happens a priori, if it is not the case
        the simulation is vectorized
    sat_check_freq : int or float, default=15
        Frequency to check for saturation in minutes.
        The smaller the more time and memory consumes
    verbose : bool, default=False
        Print messages and progress bar

    Returns
    -------
    cum_in : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people entering
    cum_out : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people leaving
    actives : numpy.array
        shape : (ndays, nruns)
        Number of people occuping bed at that day: cum_in-cum_out
    """
    assert x.ndim == 1
    assert times_in.shape == times_out.shape
    npeople = np.isfinite(times_in).sum(axis=0).max()
    nruns = times_in.shape[1]
    times_in = times_in.copy()
    times_out = times_out.copy()
    # maximum time we are interested in
    tmax = x.max()
    with warnings.catch_warnings():
        # RuntimeWarning: invalid value encountered in greater
        warnings.filterwarnings('ignore',
                                r'invalid value encountered in greater',
                                category=RuntimeWarning)
        mask_future_in = times_in > tmax
    times_in[mask_future_in] = np.inf
    times_out[mask_future_in] = np.inf

    # time interval to check for saturation in minutes
    sat_check_freq = 15
    # no saturation is possible, no need to run bed filling simulation
    if (beds_max is None) or (npeople + beds_0 <= beds_max):
        cum_in = from_times_to_counts(x, times_in)
        cum_out = from_times_to_counts(x, times_out)
        actives = cum_in - cum_out
        return cum_in, cum_out, actives

    if try_heuristic:
        # we check that the number of active cases is smaller than max_beds
        min_time = max((x.min(), times_in[np.isfinite(times_in)].min()))
        max_time = min((x.max(), times_out[np.isfinite(times_out)].max()))
        assert max_time > min_time
        npoints_grid = (max_time - min_time) * 24 * (sat_check_freq / 60)
        x_grid = np.linspace(min_time,
                             max_time,
                             num=int(npoints_grid),
                             dtype='float')
        actives_grid = from_times_to_counts(
            x_grid, times_in) - from_times_to_counts(x_grid, times_out)
        saturated_runs_mask = actives_grid.max(axis=0) > beds_max
        if verbose:
            nsatruns = saturated_runs_mask.sum()
            nsatmean = 100 * saturated_runs_mask.mean()
            print(f'Number of saturated runs: {nsatruns} ({nsatmean:.2f}%)')
    # assume there will be saturation in all runs
    else:
        saturated_runs_mask = np.ones((nruns, ), dtype='bool')

    # no one saturated -> all vectorized
    if np.alltrue(~saturated_runs_mask):
        cum_in = from_times_to_counts(x, times_in)
        cum_out = from_times_to_counts(x, times_out)
        actives = cum_in - cum_out
        return cum_in, cum_out, actives

    else:
        saturated_runs_indices = np.nonzero(saturated_runs_mask)[0]
        non_saturated_runs_indices = np.nonzero(~saturated_runs_mask)[0]
        list_cum_in = [None] * nruns
        list_cum_out = [None] * nruns
        list_actives = [None] * nruns

        # vectorize the non-saturated
        if saturated_runs_mask.mean() < 1:
            times_in_not_sat = times_in[:, non_saturated_runs_indices]
            times_out_not_sat = times_out[:, non_saturated_runs_indices]
            # in order to use zip, we move run idx to first dimension
            cum_in = np.swapaxes(from_times_to_counts(x, times_in_not_sat), 0,
                                 1)
            cum_out = np.swapaxes(from_times_to_counts(x, times_out_not_sat),
                                  0, 1)
            actives = cum_in - cum_out
            # fill them into the list with the correct run index
            for i_run, c_in, c_out, act in zip(non_saturated_runs_indices,
                                               cum_in, cum_out, actives):
                list_cum_in[i_run] = c_in
                list_cum_out[i_run] = c_out
                list_actives[i_run] = act

        # if verbose and can import tqdm, then show progress bar
        saturated_runs_indices_ = saturated_runs_indices
        if verbose:
            try:
                from tqdm import tqdm
                saturated_runs_indices_ = tqdm(saturated_runs_indices)
            except ModuleNotFoundError:
                pass

        # non-saturated cannot be vectorized
        for i_run in saturated_runs_indices_:
            times_in_1d = times_in[:, i_run]
            times_out_1d = times_out[:, i_run]
            cum_in, cum_out, actives = simulate_bed_filling_1d(
                x,
                times_in_1d,
                times_out_1d,
                beds_max=beds_max,
                try_heuristic=False)
            list_actives[i_run] = actives
            list_cum_in[i_run] = cum_in
            list_cum_out[i_run] = cum_out

        cum_in = np.swapaxes(np.array(list_cum_in), 0, 1)
        cum_out = np.swapaxes(np.array(list_cum_out), 0, 1)
        actives = np.swapaxes(np.array(list_actives), 0, 1)
        return cum_in, cum_out, actives
