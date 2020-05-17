import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def distplot(series, title=""):
    """
    Plots the distribution of given series
    """
    ax = plt.figure(figsize=(12,6)).gca()
    # Set as many bins as days
    bins = int(series.max() - series.min() + 1)
    n, bins, patches = plt.hist(series, bins, facecolor='b', alpha=1, density=True)
    plt.ylabel("Densidad de probabilidad")
    plt.xlabel("Dias")
    plt.title(title)
    # Only integer days
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
