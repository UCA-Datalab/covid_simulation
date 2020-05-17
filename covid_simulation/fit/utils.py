import datetime as dt
import pandas as pd

from covid_simulation.fit.models import *


def ser_to_numpy(ser, start_date=None):
    """
    Transform series with datetime index into numpy arrays

    Parameters
    ----------
    ser : pd.Series
        series with datetime index
    start_date : datetime.datetime
        date to assign index 0

    Returns
    -------
    x : np.array
        dates index transformed into integers
    y : np.array
        values
    """
    assert isinstance(ser, pd.Series)

    # creates the numpy arrays, translate date to int
    # x = días (en entero)
    if start_date is not None:
        x = (ser.index - start_date).days.to_numpy()
    else:
        x = ser.reset_index().index
    y = ser.to_numpy()
    return x, y


def df_to_arrays(df, region, start_date, case="admitted", min_cases=0):
    """
    Transform the df into numpy arrays.

    Returns
    =======
    x : np.array
        dates transformed into integers
    y : np.array
        cumulative cases
    dydx : np.array
        daily cases (difference), it has one point less
    """
    # No queremos trabajar en el df original
    df = df.copy()
    df = df.sort_index()

    # Listamos los casos por dia
    ser = df[df.region == region][case]

    # queremos que haya al menos 10 casos
    mask = ser > min_cases
    ser = ser[mask]

    # creates the numpy arrays, translate date to int
    # x = días (en entero)
    x = (ser.index - start_date).days.to_numpy()
    # y = casos acumulados
    y = ser.to_numpy()
    # dydx = casos por día
    dydx = np.diff(y)

    return x, y, dydx


def _predict_with_str_dict(model, params, x_pred):
    """
    Load a model using its name and parameters contained in a dictionary.
    Predict over x_pred.

    See predict_with_model for a in-depth explanation
    """
    # Check we are using a valid model
    if model not in dic_models.keys():
        raise ValueError(f"{model} is not a valid model name. Try:\n{dic_models.keys()}")

    # Check all the parameters are valid
    missing_params = [p for p in dic_models[model] if p not in params.keys()]
    if len(missing_params) > 0:
        raise ValueError(f"Model {model} is missing the following parameters:\n{missing_params}")
    excessive_params = [p for p in params.keys() if p not in dic_models[model]]
    if len(excessive_params) > 0:
        raise ValueError(f"The following parameters are not required for model {model}:\n{excessive_params}")

    # Realizamos la predicción
    y_pred = eval(model)(x_pred, **params)
    return y_pred


def predict_with_model(model, params, x_pred):
    """
    Uses given model and parameters to perform a fit on x_pred days.

    Parameters
    ----------
    model : str or callable
        Name of the model to use. Examples: tanh_cdf, gompertz_simple
        In this case, `params` must be a dictionary.
        -OR-
        The model function to be used.
        I this case, `params` must be an array
    params : dict or array (see above)
        Dictionary containing the parameters needed for the model.
        Example por gompertz_simple: params = {"a": 10000, "u": 1000, "d": 20}
        -OR-
        An array containing the values of those parameters, in the order
        asked by the model function.
    x_pred : array
        Array of days (integers or floats) to be predicted on.

    Returns
    -------
    y_pred : array
        Array of the same length as x_pred, containing the predictions
        for the days in x_pred.
    """
    # Option 1
    if callable(model) and type(params) == list:
        y_pred = model(x_pred, *params)
    # Option 2
    elif type(model) == str and type(params) == dict:
        y_pred = _predict_with_str_dict(model, params, x_pred)
    else:
        raise ValueError(
            "Variable types are not compatible. Please read the documentation.")
    return y_pred


def include_predictions_df(df, x_pred, y_pred, start_date, region, case="admitted"):
    """
    Add the fit arrays to the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
    x_pred : numpy.array
    y_pred : numpy.array
    start_date : date
    region : str
    case : str, default="admitted"

    Returns
    -------
    df : pandas.DataFrame
        Original dataframe with the predicted days included

    """
    # No queremos trabajar en el df original
    df = df.copy()

    # Creamos las fechas
    dates = [start_date + dt.timedelta(int(d)) for d in x_pred]

    # Creamos el dataframe con las predicciones
    df_pred = pd.DataFrame(y_pred, columns=[case], index=dates)
    df_pred = df_pred.astype(int)
    df_pred["region"] = region

    # Unimos al original
    df = pd.concat([df, df_pred])

    return df
