import numpy as np
import warnings

import matplotlib.pyplot as plt

from inspect import signature
from scipy.optimize import curve_fit

'''
functions used to fit the total number of cases (cumulative)
for a given variable (deaths, infected, ...)
'''


def _validation_metric(y_pred, y_true):
    """
    Metric to use for validation.
    The smaller the better.

    Now: relative MAE
    """
    assert len(y_pred) == len(y_true), 'y_pred and y_true must have same len'

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert (y_true == 0).sum() == 0, 'y_true cannot contain zero values'
    return np.mean(np.abs((y_pred - y_true) / y_true))


def fit_model_params(model, x, y, params0=None, params1=None):
    """
    Obtain best parameters for model fitting to x,y data
    It uses params0 (can be None) as initial value for the parameters,
    if it cannot fit with those, then it tries to fit using params1.
    If it cannot fit any of them, returns None.

    Parameters
    ==========
    model : callable
        function with the model to fit
    x : Union[list, np.array]
    y : Union[list, np.array]
    params0 : Union[list, np.array, None]
        initial value of parameters for gradient descent
    params1 : Union[list, np.array, None]
        initial value of parameters for gradient descent

    Returns
    =======
    best_params : Union[list, None]
        best fit parameters, or None if not found
    """
    assert callable(model), 'model must be a function'
    assert len(x) == len(y), 'x and y must have the same number of points'
    if params0 is not None:
        n_params = len(signature(model).parameters) - 1
        msg = 'params0 must have the same number of parameters as model'
        assert (len(params0) == n_params), msg
    if params1 is not None:
        n_params = len(signature(model).parameters) - 1
        msg = 'params1 must have the same number of parameters as model'
        assert (len(params1) == n_params), msg

    # sometimes the method does not converge
    try:
        best_params, _ = curve_fit(model, x, y, p0=params0)
    except RuntimeError:
        try:
            warnings.warn('parameters not found using params0, params1 used')
            best_params, _ = curve_fit(model, x, y, p0=params1, method='trf')
        except RuntimeError:
            msg = "parameters not found, try to change initial parameters"
            warnings.warn(msg)
            best_params = None

    return best_params


def _fit_model_and_predict(model, x, y, n_pred=5, params0=None, params1=None):
    """
    fit model to data and predict n_pred days ahead

    Parameters
    ==========
    model : callable
        function with the model to fit
    x : Union[list, np.array]
    y : Union[list, np.array]
    n_pred : int, default=5
        number of days to predict in the future
    params0 : Union[list, np.array, None]
        initial value of parameters for gradient descent
    params1 : Union[list, np.array, None]
        initial value of parameters for gradient descent

    Returns
    =======
    y_pred = np.array
        predicted points for n_pred days ahead
    """
    assert isinstance(n_pred, int), 'n_pred must be an int'
    x_pred = np.arange(max(x) + 1, max(x) + 1 + n_pred, 1, dtype='int')
    assert len(x_pred) == n_pred

    best_params = fit_model_params(model, x, y, params0=params0, params1=params1)

    if best_params is None:
        warnings.warn('model not fitted, impossible to predict')

    y_pred = model(x_pred, *best_params)

    assert len(y_pred) == n_pred
    return y_pred


def temporal_validation_metric(model,
                               x,
                               y,
                               val_size=5,
                               min_train_size=10,
                               params0=None,
                               plot=False,
                               plot_kwargs=None):
    """
    Perform temporal cross-validation using fixed val_size.
    The smaller the better.

    It uses at least min_train_size points as train set, and
    validate on the next val_size points.
    Then it aggregates the matric for all the validation sets.

    Parameters
    ==========
    model : callable
        function with the model to fit
    x : Union[list, np.array]
    y : Union[list, np.array]
    val_size : int, default=5
        number of days to predict in the future
    min_train_size : int, default=10
        min number of points to use as training
    params0 : Union[list, np.array, None]
        initial value of parameters for gradient descent
    plot : bool, default=False
        plot the metric vs train_size
    plot_kwargs : dict
        Parameters for the plot function


    Returns
    =======
    metric_agg : float
        aggregated metric for all validation sets
    """
    assert len(x) >= min_train_size + val_size
    max_train_size = len(x) - val_size
    # fit to all data to determine good parameters to initialize gdp
    params_all = fit_model_params(model, x, y, params0=params0, params1=None)

    train_size_list = list(range(min_train_size, max_train_size + 1))
    val_metric_list = []
    for train_size in train_size_list:
        # extract train and validation
        x_train, y_train = x[:train_size], y[:train_size]
        y_val = y[train_size:train_size + val_size]

        y_pred = _fit_model_and_predict(
            model, x_train, y_train, n_pred=val_size,
            params0=params_all, params1=params0)

        metric = _validation_metric(y_pred, y_val)
        val_metric_list.append(metric)

    metric_agg = np.mean(val_metric_list)

    if plot:
        label = f"{model.__name__}: {metric_agg:.2f}"
        orig_kwargs = {'linestyle': 'dashed', "marker": 'o',
                       "label": label}
        if plot_kwargs is not None:
            orig_kwargs.update(plot_kwargs)
        plt.plot(train_size_list, val_metric_list, **orig_kwargs)

        plt.ylim(ymin=0)
        plt.legend()
        plt.xlabel('Size training set')
        plt.ylabel('Error metric')

    return metric_agg
