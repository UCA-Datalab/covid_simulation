import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import lognorm, weibull_min, kstest


def fit_lognorm(series, s=0, floc=0, scale=1, title=""):
    """
    Fits a lognormal distribution, initialized with
    given parameters. Plots the fitted distribution
    against the ground-truth data.
    
    Params
    ------
    series: Pandas series
    Pandas series containing the ground-truth values
    
    Returns
    -------
    params : dictionary
    Contains the fitted parameters
    """
    # We dont want to work on the original series
    series = series.copy()
    # Lognorm needs values > 0
    series = series[series > 0]
    
    # Fit distribution
    (s, loc, scale) = lognorm.fit(series, s, floc=floc, scale=scale)
    
    # Plot
    ax = plt.figure(figsize=(12,6)).gca()
    # the histogram of the data
    # Set as many bins as days
    bins = int(series.max() - series.min() + 1)
    n, bins, patches = plt.hist(series, bins, facecolor='green', alpha=1, density=True)
    # add a 'best fit' line
    y = lognorm.pdf(bins, s, floc, scale)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel("Días")
    plt.title(title)
    # Only integer days
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Store parameters
    params = {"s":s, "scale":scale}
    return params


def fit_weibull(series, c=0, floc=0, scale=1, title=""):
    """
    Fits a Weibull distribution, initialized with
    given parameters. Plots the fitted distribution
    against the ground-truth data.
    
    Params
    ------
    series: Pandas series
    Pandas series containing the ground-truth values
    
    Returns
    -------
    params : dictionary
    Contains the fitted parameters
    """
    # Fit distribution
    (c, loc, scale) = weibull_min.fit(series, c, floc=floc, scale=scale)
    
    # Plot
    ax = plt.figure(figsize=(12,6)).gca()
    # the histogram of the data
    # Set as many bins as days
    bins = int(series.max() - series.min() + 1)
    n, bins, patches = plt.hist(series, bins, facecolor='green', alpha=1, density=True)
    # add a 'best fit' line
    y = weibull_min.pdf(bins, c, floc, scale)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel("Días")
    plt.title(title)
    # Only integer days
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Store parameters
    params = {"c":c, "scale":scale}
    return params


def non_parametric_test(series, dist, params, sig=0.05):
    """
    Performs a non parametric test (Kolmogorov-Smirnov)
    Null hypothesis: both distributions are the same.
    """
    # Goodness of fit
    d, pvalue = kstest(series, dist, params)
    print(f"Hipótesis nula: los datos provienen de una distribución {dist}")
    if pvalue < sig:
        print(f"Con un pvalor de {round(pvalue, 6)} rechazamos H0 a un nivel de significación de {sig}")
        print(f"Los datos no se ajustan a la {dist} dada")
    else:
        print(f"Con un pvalor de {round(pvalue, 6)} no podemos rechazar H0 a un nivel de significación de {sig}")
        print(f"Los datos podrían provenir de una {dist}")
