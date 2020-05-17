import numpy as np

from covid_simulation.exceptions import DicModelError
from covid_simulation.exceptions import ModelNotDefinedError
from covid_simulation.exceptions import ParameterNumberError
from inspect import signature


from covid_simulation.fit.models import gompertz_cdf
from covid_simulation.fit.models import exp_cdf_nonorm
from covid_simulation.fit.models import gompertz_simple
from covid_simulation.fit.models import tanh_KM_cdf


# list_params_names = list((signature(model).parameters).keys())[1:]
dic_models = {"tanh_cdf": ["a", "b", "c"],
              "tanh_KM_cdf": ["a", "b", "c"],
              "gompertz_cdf_nonorm": ["a", "u", "d", "y0"],
              "exp_cdf_nonorm": ["a", "b", "c"],
              "gompertz_simple": ["a", "u", "d"]}


# dictionary associating model names to the model function
modelname2model = {"gompertz": gompertz_simple,
                   "tanh": tanh_KM_cdf,
                   'exp': exp_cdf_nonorm}


def add_model_to_dic(dic_model, inplace=False):
    """
    add model to dic with model name.
    if 'params' or 'params0' in keys, then it also checks
    that the number is correct and adds
    'nparams' to the dictionary.

    Parameters
    ----------
    dic_model : dict
        dictionary containing the model name
    inplace : bool, default=False

    Raises
    ------
    DicModelError
        dic_model has no defined parameters
    ModelNotDefinedError
        'model_name' has no associated model defined
    ParameterNumberError
        the number of parameters does not coincide

    Returns
    -------
    None or dict
    """
    assert isinstance(dic_model, dict), "`dic_model` must be a dict object"
    assert 'model_name' in dic_model.keys(), "`dic_model` has no key 'model_name'"
    if not inplace:
        dic_model = dic_model.copy()
    model_name = dic_model['model_name'].strip().lower()
    try:
        model = modelname2model[model_name]
    except KeyError:
        raise ModelNotDefinedError(f"model_name: '{model_name}' not defined")

    dic_model['model'] = model
    nparams = len((signature(model).parameters).keys()) - 1

    if 'params' in dic_model.keys():
        params = dic_model['params']
        if not len(params) == nparams:
            raise ParameterNumberError(f"params: {params} must have len {nparams}")
    if 'params0' in dic_model.keys():
        params = dic_model['params0']
        # params0 can be None
        if (not len(params) == nparams) and (params is not None):
            raise ParameterNumberError(f"params0: {params} must have len {nparams}")

    if {'params', 'params0'}.isdisjoint(set(dic_model.keys())):
        raise DicModelError(f"'params' or 'params0' must be defined in `dic_model`: {dic_model}")

    dic_model['nparams'] = nparams

    if not inplace:
        return dic_model
    else:
        return None
