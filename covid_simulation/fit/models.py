import numpy as np


def rename(newname):
    """
    Decorator to set function's `__name__` attribute.
    """
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


'''
All models must have the input as the first variable, and then the parameters.
'''


def gompertz_cdf(t, t0, logb, logeta):
    eta = np.exp(logeta)
    b = np.exp(logb)
    ptilda = np.exp(eta * (1 - np.exp((t - t0) * b)))
    p = 1 - ptilda
    return p


def tanh_cdf(x, a, b, c):
    return a * (np.tanh(b * x + c) + 1.)


@rename('Tanh')
def tanh_KM_cdf(x, a, b, c):
    """
    It is not really a cdf, since it can be negative, depending on c.
    Reference
    =========
    A Contribution to the Mathematical Theory of Epidemics.
    By W. 0. Kermack and A. G. McKendrick
    """
    return (a / b) * (np.tanh(c) + np.tanh(b * x - c))


def gompertz_cdf_nonorm(x, a, u, d, y0):
    """
    Gompertz growth model.
    Proposed in Zwietering et al., 1990 (PMID: 16348228)
    """
    u = np.abs(u)  # positive constraint, works better numerically
    y = (a * np.exp(-np.exp(((u / a) * (d - x)) + 1))) + y0
    return y


@rename('Gompertz')
def gompertz_simple(x, a, u, d):
    """
    it is better not to consider the y0 parameters, and set it to zero
    params0 = [10000, 1000, 15]
    """
    return gompertz_cdf_nonorm(x, a, u, d, y0=0)


@rename('Exp')
def exp_cdf_nonorm(x, a, b, c):
    """
    Gompertz growth model.
    Proposed in Zwietering et al., 1990 (PMID: 16348228)
    """
    y = a * np.exp(b * x) + c
    return y
