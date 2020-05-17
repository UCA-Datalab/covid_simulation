import functools
import numpy as np


def cast_to_float64(kwarg_list=None):
    """
    Decorator that casts all args and kwargs in `kwarg_list`
    into a float64 numpy.array.

    **TODO:** add args_list as argument. if 'all' apply to all.
    Parameters
    ----------
    kwarg_list : None, iterable
        List of kwargs to be castd

    Returns
    -------
    decorator_cast : callable
    """
    # extra decorator to can have a decorator with variables
    def decorator_cast(func):
        @functools.wraps(func)
        def wrapper_cast(*args, **kwargs):
            casted_args = [np.asarray(arg, 'float64') for arg in args]
            casted_kwargs = {k: np.asarray(
                v, 'float64') if k in kwarg_list else v for k, v in kwargs.items()}
            return func(*casted_args, **casted_kwargs)
        return wrapper_cast
    return decorator_cast


@cast_to_float64(['x0'])
def weibull(scale, shape_a, x0=0, size=None, seed=0):
    """
    Sample from Weibull distribution conditioned to `x>x0`, i.e.,
    p(x) = q(x|x>x0), where q is Weibull with `scale` and `shape_a` parameters.

    Parameters
    ----------
    scale : float or np.array
        Scale parameter of Weibull
    shape_a : float or np.array
        Shape parameter of Weibull
    x0 : float or np.array
        Minimum value of output
    size : interable, default=None
        Shape of the output
    seed : int, default=0
        Random seed for numpy random

    Returns
    -------
    x : numpy.array
        shape : size

    Note
    ----
    This is a numerically stable solution, in which
    for x0>scale and shape_a>>1 it produce non-infinite results.
    It avoids taking logs of exps.
    """
    assert np.all(x0 >= 0)
    assert np.all(scale > 0)
    assert np.all(shape_a > 0)

    np.random.seed(seed)

    v = np.random.uniform(0, 1, size=size)
    x = scale * ((x0 / scale)**shape_a - np.log(1 - v))**(1 / shape_a)
    return x


@cast_to_float64(['x0'])
def weibull_pdf(x, scale, shape_a, x0=0):
    """
    Probbility density function of Weibull distribution
    conditioned to `x>x0`, i.e., p(x) = q(x|x>x0),
    where q is Weibull with `scale` and `shape_a` parameters.

    Parameters
    ----------
    x : float or np.array
        Input for the pdf
    scale : float or np.array
        Scale parameter of Weibull
    shape_a : float or np.array
        Shape parameter of Weibull
    x0 : float or np.array
        Minimum value of output

    Returns
    -------
    p : float or np.array
        Probability density function
    """
    assert np.all(x >= 0)
    assert np.all(x0 >= 0)
    assert np.all(scale > 0)
    assert np.all(shape_a > 0)

    # normalize input
    t = x / scale
    t0 = x0 / scale

    p = (1 / scale) * shape_a * (t**(shape_a - 1)) * \
        np.exp(-t**shape_a + t0**shape_a)
    # step function
    if isinstance(p, np.ndarray):
        p[x < x0] = 0
    elif x < x0:
        p = 0

    return p
