import numpy as np
import pandas as pd
import datetime as dt
import warnings
from covid_simulation.fit.fit import fit_model_params
from covid_simulation.fit.utils import ser_to_numpy
from covid_simulation.fit.models_utils import add_model_to_dic


def predict_and_append(ser, start_date,
                       dic_model,
                       pred_size,
                       fit_model=False):
    """
    Append values predicted with model to the series
    it calculates the predicted cumulative values, and from
    that the daily values. From which you calculate the predicted
    cumulative values.

    This uses as ground truth the predicted daily values and not
    the cumulative values, to avoid the series to stop
    being monotonic increasing.

    Parameters
    ----------
    ser : pandas.Series
        Series with values
    start_date : datetime.datetime
        date with index 0
    dic_model : dict
        dictionary with name and parameters of the model
    pred_size : int
        number of values ahead to predict
    fit_model : bool, optional
        if True, fit the model to ser using params as initial parameters

    Returns
    -------
    ser : pandas.Series
        Series with values and predicted values appended

    Raises
    ------
    ValueError
        Description
    """
    assert isinstance(ser, pd.Series)
    assert isinstance(dic_model, dict)
    assert isinstance(pred_size, int)
    assert ser.is_monotonic_increasing, f'ser must be monotonic increasing:\n{ser}'
    assert pred_size > 0

    if (not fit_model) and ('params' not in dic_model.keys()):
        warnings.warn(f'predicting with "params0" in {dic_model}')

    ser = ser.copy()
    x, y = ser_to_numpy(ser, start_date)

    # add model function to dic_model
    add_model_to_dic(dic_model, inplace=True)
    try:
        params = dic_model['params']
    except KeyError:
        params = dic_model['params0']
    model = dic_model['model']

    if fit_model:
        best_params = fit_model_params(
            model, x, y, params0=params)
    else:
        best_params = params


    # we predict the number of dayly cases, and use it to append them
    # this way, we ensure that it always increasing

    # the first value is the last day recorded
    x_pred = np.array(range(pred_size+1)) + x[-1]
    y_pred = model(x_pred, *best_params)
    # one point less, first day is the first predicted day

    dydx_pred = np.diff(y_pred)

    x_pred_corrected = x_pred[1:]
    y_pred_corrected = (y[-1]+dydx_pred.cumsum()).astype(np.int32)

    x_days = start_date + x_pred_corrected * dt.timedelta(days=1)

    # append predictions to ser
    ser_pred = pd.Series(data=y_pred_corrected, index=x_days, dtype=np.int32)
    ser = ser.append(ser_pred, verify_integrity=True).astype('uint32')

    assert ser.is_monotonic_increasing, f'ser must be monotonic increasing:\n{ser}'
    return ser
