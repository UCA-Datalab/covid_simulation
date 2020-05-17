import numpy as np


def sample_noise(x, std=.05, n=150):
    x = x.copy()
    x = x[:, np.newaxis]
    gaussian_noise = np.random.normal(scale=std, size=(x.shape[0], n))
    x_gaussian = x * (1 + gaussian_noise)
    x_poisson = np.random.poisson(x_gaussian)
    return x_gaussian


def get_quantiles(x, axis,
                  quantiles=(.05, .25, .75, .95),
                  add_median=True):
    if add_median:
        quantiles = [.5] + list(quantiles)
        median, *quan = np.quantile(x, quantiles, axis=axis)
        quan = np.asarray(quan)
        return quan, median
    else:
        quan = np.quantile(x, quantiles, axis=axis)
        return quan


def add_noise_and_get_quantiles(x,
                                sample_noise,
                                quantiles=(.05, .25, .75, .95),
                                add_median=True):
    x_noise = sample_noise(x)
    if add_median:
        quantiles = [.5] + list(quantiles)
        median, *quan = np.quantile(x_noise, quantiles, axis=1)
        quan = np.asarray(quan)
        return quan, median
    else:
        quan = np.quantile(x_noise, quantiles, axis=1)
        return quan
