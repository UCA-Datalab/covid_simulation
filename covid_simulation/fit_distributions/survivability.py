import matplotlib.pyplot as plt
import numpy as np
import scipy

from covid_simulation.config import PARAMS_HOSP_TO_ALTA
from covid_simulation.config import PARAMS_HOSP_TO_UCI
from covid_simulation.config import PARAMS_PROB_UCI


def get_event_proba(ndays=50):
    """
    Computes different probabilities for events occurring
    to a patient given his length of stay in hospital
        - Leaving the hospital (discharge)
        - Needing intensive care (UCI)
        - Staying in hospital

    Parameters
    ----------
    ndays : integer, default 50
        Maximum number of days to take into account

    Returns
    -------
    x : numpy array
        The days
    p_alta_x : numpy array
        Probabily of being discharged by day
    p_uci_x : numpy array
        Probability of needing UCI by day
    p_planta_x : numpy array
        Probability of staying at the hospital by day
    """

    # The probability of needing UCI is hard-coded by us
    # It was estimated by
    # TODO: insert reference here
    p_uci = PARAMS_PROB_UCI['p_uci']
    p_uci_now_given_uci = PARAMS_PROB_UCI['p_uci_now_given_uci']

    # Parameters for Weibull distributions
    for dic in [PARAMS_HOSP_TO_UCI, PARAMS_HOSP_TO_ALTA]:
        if 'shape_a' in dic.keys():
            dic['c'] = dic.pop('shape_a')

    # Probability of entering UCI
    dist_host_to_uci = scipy.stats.weibull_min(scale=PARAMS_HOSP_TO_UCI["scale"],
                                               c=PARAMS_HOSP_TO_UCI["shape"])
    # Probability of being discharged
    dist_host_to_alta = scipy.stats.weibull_min(scale=PARAMS_HOSP_TO_ALTA["scale"],
                                                c=PARAMS_HOSP_TO_ALTA["shape"])

    # Array of days
    x = np.linspace(0, ndays, ndays * 2)
    # Discharge probability by day
    p_alta_x = (1 - p_uci) * dist_host_to_alta.cdf(x)
    # UCI probability by day
    p_uci_x = p_uci * (p_uci_now_given_uci +
                       ((1 - p_uci_now_given_uci)) * dist_host_to_uci.cdf(x))
    # Staying in hospital probability by day
    p_planta_x = 1 - p_alta_x - p_uci_x

    return x, p_alta_x, p_uci_x, p_planta_x


def plot_event_proba(x, p_alta_x, p_uci_x):
    """
    Plot the different event probabilities against
    the length of stay

    Parameters
    ----------
    x : numpy array
        The days
    p_alta_x : numpy array
        Probabily of being discharged by day
    p_uci_x : numpy array
        Probability of needing UCI by day
    """

    y0 = 0
    y1 = 100 * p_uci_x
    y2 = 100 * (1 - p_alta_x)
    y3 = 100

    plt.figure(dpi=100)

    plt.fill_between(x, y0, y1, label='UCI', alpha=.7)
    plt.fill_between(x, y1, y2, label='Planta', alpha=.7)
    plt.fill_between(x, y2, y3, label='Alta', alpha=.7)
    plt.legend()
    plt.ylim(0, 100)
    plt.xlabel('Dias ingresado')
    plt.ylabel('Probabilidad (%)')
