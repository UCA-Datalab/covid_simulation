'''
Distributions used to sample different random distributions of the simulation
Except daily cases, which is a relative noise, the rest have as input
size and seed.

Parameters are set here.
'''

import numpy as np
import scipy.stats


from covid_simulation.config import PARAMS_DAILY_CASES
from covid_simulation.config import PARAMS_HOSP_TO_ALTA
from covid_simulation.config import PARAMS_HOSP_TO_UCI
from covid_simulation.config import PARAMS_PROB_UCI
from covid_simulation.config import PARAMS_UCI_TO_ALTA
from covid_simulation.exceptions import DistributionError


def define_dist(distribution, **kwargs):
    """Define scipy distribution from dictionary

    Parameters
    ----------
    distribution : str
        Name of the distribution
    **kwargs : dict
        Parameters of the distribution

    Returns
    -------
    dist : scipy.stats._distn_infrastructure.rv_continuous
        Scipy distribution

    Raises
    ------
    DistributionError
        Use a non-defined distribution
    """
    if distribution.strip().lower() == 'weibull':
        assert set(kwargs.keys()) >= {
            'scale', 'shape'}, "weibull distribution needs 'scale' and 'shape' parameters"
        dist = scipy.stats.weibull_min(
            c=kwargs['shape'], scale=kwargs['scale'], loc=0)
    else:
        raise DistributionError(f'{distribution} is not a defined distribution')

    return dist


hosp_los_dist = define_dist(**PARAMS_HOSP_TO_ALTA)
hosp_to_uci_los_dist = define_dist(**PARAMS_HOSP_TO_UCI)
uci_los_dist = define_dist(**PARAMS_UCI_TO_ALTA)


def hosp_los_params(scale, shape_a, size, seed=0):
    np.random.seed(seed)
    return scale * np.random.weibull(shape_a, size)


def hosp_to_uci_los_params(scale, shape_a, size, seed=0):
    np.random.seed(seed)
    return scale * np.random.weibull(shape_a, size)


def uci_los_params(scale, shape_a, size, seed=0):
    np.random.seed(seed)
    return scale * np.random.weibull(shape_a, size)


#################
# These are the ones loaded for the simulation
#################

def distribution_daily_cases(x, seed=0):
    '''
    Noise to add to predicted daily cases.
    '''
    np.random.seed(seed)
    gauss = np.random.normal(scale=PARAMS_DAILY_CASES['std'], size=x.shape)
    mean_pois = np.maximum(0, x * (1 + gauss))
    poiss = np.random.poisson(mean_pois)
    return poiss


def hosp_los(size=None, seed=0):
    '''
    Lenght of stay in the hospital if you don't need UCI.
    '''
    return hosp_los_dist.rvs(size=size, random_state=seed)


def hosp_to_uci_los(size=None, seed=0):
    '''
    Lenght of stay in the hospital if you then go to UCI.
    '''
    return hosp_to_uci_los_dist.rvs(size=size, random_state=seed)


def uci_los(size=None, seed=0):
    '''
    Lenght of stay in the UCI.
    '''
    return uci_los_dist.rvs(size=size, random_state=seed)


def hosp_after_uci_los(mean_time, seed=0):
    '''
    Lenght of stay in the hospital after leaving the UCI.
    We assume that the los is similar to the UCI one, with gaussian
    noise of std = 1.
    '''
    np.random.seed(seed)
    return mean_time + np.random.normal(size=mean_time.shape)
