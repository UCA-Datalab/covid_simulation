"""
This script serves as a pipeline that implements
all the other functions contained in this folder

Attributes
----------
dic_cmap : TYPE
    Description
dic_names : TYPE
    Description
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy.optimize import minimize_scalar

from covid_simulation.config import PARAMS_PROB_UCI
from covid_simulation.exceptions import FailedOptimizationError
from covid_simulation.fit.utils import ser_to_numpy
from covid_simulation.noise.noise import get_quantiles
from covid_simulation.pandas_utils import assert_filled_in_dates
from covid_simulation.plot_utils import plot_prediction_with_uncertainty
from covid_simulation.simulation.sampling import noise_on_cases
from covid_simulation.simulation.sampling import noise_on_times
from covid_simulation.simulation.sampling import propagate_times_hospital
from covid_simulation.simulation.sim_filling import simulate_bed_filling
from covid_simulation.simulation.utils import from_counts_to_times

dic_names = {"uci_in_diary": "Ingresos UCI diarios",
             "uci_in": "Ingresos UCI acumulados",
             "uci_active": "Camas ocupadas en UCI",
             "hosp_in": "Ingresos no UCI acumulados",
             "hosp_in_diary": "Ingresos no UCI diarios",
             "hosp_active": "Camas ocupadas no UCI"
             }

dic_cmap = {"uci_in_diary": "Oranges",
            "uci_in": "Reds",
            "uci_active": "Greens",
            "hosp_in": "Greens",
            "hosp_in_diary": "Blues",
            "hosp_active": "Purples"}


def run_simulation(ser_admitted, nruns, max_beds_uci,
                   p_uci=None,
                   try_heuristic_uci=True,
                   calculate_hosp=True,
                   max_beds_hosp=None,
                   try_heuristic_hosp=True,
                   verbose=True,):
    """
    Do the whole simulation process:
        - Add noise (different to each simulation)
        - Compute the times and LOS for each simulation
        - Simulate bed filling both in hospital and UCI

    Parameters
    ----------
    ser_admitted : pandas.Series
        Series with the cumulative cases entering in the hospital per day.
        Its index must be the date.
    nruns : int
        Number of runs to sample.
    max_beds_uci : int or None
        Maximum number of beds in UCI. If None, there is no limit
    p_uci : float or None, default=None
        Probability of going to UCI from hospital.
        If None, it uses values from PARAMS_PROB_UCI
    try_heuristic_uci : bool, default=True
        Check if saturation happens a priori, if it is not the case
        the simulation is vectorized
    calculate_hosp : bool, default=True
        Calculate quantities related to hospital beds
    max_beds_hosp : int or None, default=None
        Maximum number of beds in hospital. If None, there is no limit
    try_heuristic_hosp : bool, default=True
        Check if saturation happens a priori, if it is not the case
        the simulation is vectorized
    verbose : bool, optional
        Print comments on every step of the simulation

    Returns
    -------
    uci_in : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people entering UCI
    uci_out : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people leaving UCI
    uci_active : numpy.array
        shape : (ndays, nruns)
        Number of people occuping bed at UCI that day: cum_in-cum_out
    hosp_in : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people entering hospital
    hosp_out : numpy.array
        shape : (ndays, nruns)
        Cumulative number of people leaving hospital
    hosp_active : numpy.array
        shape : (ndays, nruns)
        Number of people occuping bed at hospital that day: cum_in-cum_out
    """
    assert isinstance(
        ser_admitted, pd.Series), 'ser_admitted must be a pandas.Series object'
    assert ser_admitted.is_monotonic_increasing, 'ser_admitted must be monotonic increasing'
    # check that all in-between dates are filled
    ser_admitted = ser_admitted.copy().dropna()
    assert_filled_in_dates(ser_admitted)

    # transform into numpy arrays with integers
    x, y = ser_to_numpy(ser_admitted, start_date=None)

    if verbose:
        print('\nSampling from number of people and random arrival distributions...')
    y_noise = noise_on_cases(y, nruns=nruns)
    times = from_counts_to_times(x, y_noise)
    times = noise_on_times(times)

    if verbose:
        print('\nSampling from length of stay distributions...')
    p_uci_now_given_uci = PARAMS_PROB_UCI['p_uci_now_given_uci']
    (times_hosp_in, times_hosp_out, times_uci_in,
     times_uci_out) = propagate_times_hospital(times,
                                               p_uci=p_uci,
                                               p_uci_now_given_uci=p_uci_now_given_uci)

    if verbose:
        print('\nImposing max beds contraint in UCI...')
    uci_in, uci_out, uci_active = simulate_bed_filling(x,
                                                       times_uci_in,
                                                       times_uci_out,
                                                       beds_max=max_beds_uci,
                                                       verbose=verbose,
                                                       try_heuristic=try_heuristic_uci)

    if not calculate_hosp:
        return uci_in, uci_out, uci_active
    else:
        if verbose:
            print('\nImposing max beds contraint in hospital...')
        hosp_in, hosp_out, hosp_active = simulate_bed_filling(x,
                                                              times_hosp_in,
                                                              times_hosp_out,
                                                              beds_max=max_beds_hosp,
                                                              verbose=verbose,
                                                              try_heuristic=try_heuristic_hosp)
        return uci_in, uci_out, uci_active, hosp_in, hosp_out, hosp_active


def _loss_p_uci_no_saturation(p_uci, p_uci_0, y_target, y_pred_0):
    """
    Loss function for p_uci assuming no saturation.
    Auxiliar function for `fit_p_uci_no_saturation`

    Smaller is better.

    Parameters
    ----------
    p_uci : float
        `p_uci` input for loss function
    p_uci_0 : float
        `p_uci` used to calculate `y_pred_0`
    y_target : numpy.array
        Real values
    y_pred_0 : numpy.array
        Predicted values

    Returns
    -------
    loss : float
        Aggregated loss function
    """
    # assume proportionality
    y_pred = (p_uci / p_uci_0) * y_pred_0
    # l1 loss function
    loss = np.mean(np.abs(y_pred - y_target))
    return loss


def fit_p_uci_no_saturation(ser_admitted, ser_uci, size_val=5,
                            p_uci_0=.15, nruns=200,
                            verbose=False):
    """
    Fit the best parameter `p_uci` according to a given lost function.
    No saturation is asummed to simplify calculation, i.e., we assume
    that the number of people in uci is directly propotional to `p_uci`.
    This way, we avoid the calculation of several simulations.

    **TODO:** allow for using this assumption.

    Parameters
    ----------
    ser_admitted : pandas.Series
        Description
    ser_uci : pandas.Series
        Description
    size_val : int, default=5
        number of points to use for validation.
        it selects the last `size_val` points
    p_uci_0 : float, default=.15
        Starting value of `p_uci`.
    nruns : int, default=200
        Number of runs to sample.
    verbose : bool, optional
        Print comments on every step of the simulation

    Returns
    -------
    p_uci_best : float
        Value of `p_uci` minimizing loss function.

    Deleted Parameters
    ------------------
    ser_admitted : pandas.Series
        Series with the cumulative cases entering in the hospital per day.
        Its index must be the date.
    df : pandas.DataFrame
        data to fit and with number of people admitted in hospital
    col_admitted : str, default='admitted'
        df column with admitted cases to run the simulation
    col_target : str, default='uci_in'
        df column with quantity to fit

    Raises
    ------
    FailedOptimizationError
        Description
    """
    assert isinstance(size_val, int)
    assert size_val > 0, '`size_val` must be greater than zero'
    assert isinstance(ser_admitted, pd.Series)
    assert isinstance(ser_uci, pd.Series)
    assert ser_admitted.is_monotonic_increasing
    assert ser_uci.is_monotonic_increasing
    # check that all in-between dates are filled
    ser_admitted = ser_admitted.copy().dropna()
    assert_filled_in_dates(ser_admitted)
    ser_uci = ser_uci.copy().dropna()

    # fist date comming from admitted, last date comming from uci
    # we are not predicting, just validating values in ser_uci
    first_date = ser_admitted.index.min()
    last_date = ser_uci.index.max()
    _, y_uci = ser_to_numpy(ser_uci, start_date=first_date)
    y_val = y_uci[-size_val:]

    # remove entries with 'date' greater than the ones in `col_target`
    mask_no_future = ser_admitted.index <= last_date
    ser_admitted = ser_admitted[mask_no_future]

    (
        uci_in, _, _,
    ) = run_simulation(ser_admitted, nruns=nruns,
                       max_beds_uci=None,
                       calculate_hosp=False,
                       p_uci=p_uci_0,
                       verbose=verbose)
    y_pred = np.median(uci_in[-size_val:], axis=1)

    # minimize loss function w.r.t. p_uci
    f = lambda p_uci: _loss_p_uci_no_saturation(
        p_uci, p_uci_0, y_target=y_val, y_pred_0=y_pred)
    res = minimize_scalar(f, bounds=(0, 1), method='bounded')
    if res.success:
        p_uci_best = float(res.x)
    else:
        raise FailedOptimizationError(res.message)

    return p_uci_best


def perform_simulations_and_store(ser_admitted, nruns,
                                  max_beds_uci, max_beds_hosp,
                                  path_csv, plots_title=None,
                                  p_uci=None,
                                  path_plots=None,
                                  ser_uci=None,
                                  verbose=True):
    """
    Do the whole simulation process:
        - Convert daily cases to acumulate
        - Add noise (different to each simulation)
        - Compute the times and LOS for each simulation
        - Simulate bed filling both in hospital and UCI
        - Store cases and metrics in csv
        - Store plots

    **TODO:** remove region, sice it is only used for title and for assert
    instead, use variable title

    Parameters
    ----------
    ser_admitted : pd.Series
        cumulative number of admitted people in hospital
    nruns : int
        Number of runs to sample.
    max_beds_uci : int or None
        Maximum number of beds in UCI. If None, there is no limit
    max_beds_hosp : int or None
        Maximum number of beds in hospital. If None, there is no limit
    path_csv : str
        Path where to store the output aggregated quantities
    plots_title : str, default=None
        title for plots
    p_uci : float or None, default=None
        Probability of going to UCI from hospital.
        If None, it uses values from PARAMS_PROB_UCI
    path_plots : str, Path or None, default = None
        Path where to store the output plots.
        If None, no plot is outputted.
    ser_uci : None, default=None
        series of cumulative cases of uci patients to plot with prediction
    verbose : bool, optional
        Description
    verbose : bool, optional

    Deleted Parameters
    ------------------
    dataset : pandas.DataFrame
        size : (ndays, *)
        Dataframe containing the number of different cases by day.
        It must contain an "ingreso" (admitted) column.
        It must also contain a "region" (region) column.
        * = Number of columns. They are be types of cases (admitted, uci...)
    region : str
        Region of interest (Autonomous Community or province)

    No Longer Returned
    ------------------
    None

    Returns
    -------
    TYPE
        Description
    """
    if ser_uci is not None:
        assert isinstance(ser_uci, pd.Series)

    # Compute first and last date
    first_date = ser_admitted.index.min()
    last_date = ser_admitted.index.max()

    x, _ = ser_to_numpy(ser_admitted, start_date=first_date)

    (
        uci_in, uci_out, uci_active,
        hosp_in, hosp_out, hosp_active
    ) = run_simulation(ser_admitted, nruns=nruns,
                       max_beds_uci=max_beds_uci,
                       max_beds_hosp=max_beds_hosp,
                       p_uci=p_uci,
                       verbose=verbose)

    # create path for plots if needed
    if path_plots is not None:
        if not os.path.exists(path_plots):
            os.mkdir(path_plots)

    # save results into dictionary
    dic_results_raw = {"uci_in": uci_in,
                       "uci_in_diary": np.diff(uci_in,
                                               axis=0, prepend=0),
                       "uci_active": uci_active,
                       "hosp_in": hosp_in,
                       "hosp_in_diary": np.diff(hosp_in,
                                                axis=0, prepend=0),
                       "hosp_active": hosp_active}

    # compute median and quantiles
    # Initialize a dictionary containing them
    dic_results_agg = {}
    quantiles = (.05, .25, .75, .95)

    if path_plots is not None:
        if verbose:
            print(f'\nGenerating plots...')

    for name, quantity in dic_results_raw.items():
        quan, median = get_quantiles(quantity, axis=1,
                                     quantiles=quantiles)

        for i in range(len(quantiles)):
            q_name = f"q{int(100*quantiles[i])}"
            dic_results_agg[(name, q_name)] = quan[i, :]
        dic_results_agg[(name, "median")] = median

        if path_plots is not None:
            with plt.style.context('seaborn-whitegrid', after_reset=True), plt.rc_context(rc={'font.size': 10}):
                # Plot them
                plt.figure(dpi=180)
                plot_prediction_with_uncertainty(x, median, quan,
                                                 start_date=first_date,
                                                 color="black",
                                                 cmap=plt.get_cmap(dic_cmap[name]))

                # Add ground truth data points
                if (ser_uci is not None) and (name == "uci_in"):
                    plt.plot(ser_uci.index, ser_uci.values, "k.", markersize=2)
                # Add ground truth data points
                if (ser_uci is not None) and (name == "uci_in_diary"):
                    plt.plot(ser_uci.index[1:], np.diff(
                        ser_uci.values), "k.", markersize=2)

                if plots_title is not None:
                    plt.title(plots_title)
                plt.xlabel('Fecha')
                plt.ylabel(dic_names[name])
                plt.savefig(os.path.join(path_plots, name + ".png"))
                plt.close()

    print('\nGenerating aggregated results csv file...')
    # Turn dictionary into a dataframe
    df_results_agg = pd.DataFrame(dic_results_agg)
    # round results to two decimals
    df_results_agg = round(df_results_agg, 2)
    # Add dates
    df_results_agg["date"] = pd.date_range(
        start=first_date, end=last_date, freq="D")
    df_results_agg.set_index("date", inplace=True)
    df_results_agg.to_csv(path_csv)

    return None
