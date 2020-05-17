import numpy as np

from covid_simulation.config import PARAMS_HOSP_TO_ALTA
from covid_simulation.config import PARAMS_HOSP_TO_UCI
from covid_simulation.config import PARAMS_UCI_TO_ALTA

from covid_simulation.simulation.distributions import hosp_to_uci_los_dist
from covid_simulation.simulation.distributions import hosp_los_dist
from covid_simulation.simulation.distributions import uci_los_dist

from copy import deepcopy

DIC_DISTRIBUTION_FUNCTIONS = {"hosp_los": hosp_los_dist,
                              "uci_los": uci_los_dist,
                              "hosp_to_uci": hosp_to_uci_los_dist}

DIC_DISTRIBUTION_PARAMS = {"hosp_los": PARAMS_HOSP_TO_ALTA,
                           "uci_los": PARAMS_UCI_TO_ALTA,
                           "hosp_to_uci": PARAMS_HOSP_TO_UCI}

dic_distribution_title = {"hosp_los": "Tiempo de estancia en el hospital para pacientes no UCI",
                          "uci_los": "Tiempo de estancia en UCI",
                          "hosp_to_uci": "Tiempo entre llegada al hospital y entrada en UCI"}

dic_offset = {"hosp_los": 50,
              "uci_los": 60,
              "hosp_to_uci": 20}


def _plot_without_data(distribution_type, offset_max=60):
    """
    Parameters
    ----------
    distribution_type : str
        Distribution to plot (hospital LOS, UCI LOS...)

     Returns
    ----------
    dic_plot : dictionary of strings and arrays
        Dictionary containing all the information to be included in a plot.
        Refer to generate_distribution_data for the whole explanation.
    """

    # Get model name and parameters
    dic_model_params = deepcopy(DIC_DISTRIBUTION_PARAMS[distribution_type])
    assert "distribution" in dic_model_params.keys(), f"key 'distribution' missing in {dic_model_params}"
    model_name = dic_model_params["distribution"]

    # Homogenize parameters
    if "c" in dic_model_params.keys():
        dic_model_params['shape'] = dic_model_params.pop("c")
    # format parameters numbers into str, and filter non-numbers out
    # **TODO:** proper filter with try: float() except None
    dic_model_params = {k: f'{v:.2f}' for k, v in dic_model_params.items() if isinstance(v, (float, int))}

    # Get values to plot
    dist = DIC_DISTRIBUTION_FUNCTIONS[distribution_type]
    step = .1
    x = np.arange(0, offset_max + step, step)
    y = 100*dist.pdf(x)
    dic_line = {"x": x.tolist(),
                "y": y.tolist()}

    # add binned distribution
    step = 1
    x_int = np.arange(0, offset_max + step, step, dtype='int')
    cum = dist.cdf(x_int)
    y_int = 100*np.diff(cum)
    # displace for the center of the bar to be in the center of the interval
    x_int = x_int[:-1] + .5
    assert x_int.shape == y_int.shape

    dic_bars = {"x": x_int.tolist(),
                "y": y_int.tolist(),
                'width':1}

    # Initialize dictionary
    dic_plot = {"xlabel": "Días",
                "ylabel": "Frecuencia relativa (%)",
                "title": dic_distribution_title[distribution_type],
                'mean': dist.mean(),
                'variance': dist.var(),
                'line': dic_line,
                'model': model_name,
                'model_params': dic_model_params,
                'bars': dic_bars}

    return dic_plot


def generate_distribution_data(distribution_type):
    """
    Parameters
    ----------
    distribution_type : str
        Distribution to plot (hosp_los, uci_los, hosp_to_uci)


     Returns
    ----------
    dic_plot : dictionary of strings and arrays
        Dictionary containing all the information to be included in a plot.
        It's keys are:
        title: str
            Title of the plot
        xlabel : str
            Label for the x-axis
        ylabel : str
            Label for the y-axis
        mean : float
            Mean value of the distribution
        variance : float
            Variance of the distribution
        line : dict
            Line coordinates {"x": list, "y": list}
            Show the pdf of the distribution
        bars : dict
            Bar coordinates {"x": list, "y": list}
            Show the pdf of the distribution
        model : str
            Name of the distribution
        model_params : dict
            Distribution parameter values {"param": float}
    """

    # Check distribution type is valid
    if distribution_type not in dic_distribution_title.keys():
        raise ValueError(f"{distribution_type} not accepted.\n`distribution_type` must be one of the following:\n{dic_distribution_title.keys()}")

    # Get dictionary info
    dic_plot = _plot_without_data(
        distribution_type, offset_max=dic_offset[distribution_type])

    return dic_plot
