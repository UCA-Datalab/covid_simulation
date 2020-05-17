import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np


def format_date_xaxis(fmt='%b %d', ax=None, interval=1):
    if ax is None:
        ax = plt.gca()
    ax.xaxis_date()
    # set ticks every week
    ax.xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=6, interval=interval))
    # set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def plot_prediction_with_uncertainty(x,
                                     y_center,
                                     y_quantiles,
                                     quantiles=(.05, .25, .75, .95),
                                     start_date=None,
                                     color='green',
                                     cmap=plt.get_cmap('Greens')):
    if start_date is not None:
        start_date_dt = dt.datetime.fromisoformat(start_date.isoformat())
        x_date = start_date_dt + x * dt.timedelta(days=1)
        x_plot = x_date

    else:
        x_plot = x

    a = np.arange(0, len(y_quantiles))
    per_quantiles = list(0.7 * (np.abs(a - a.mean()) / ((len(a)) // 2)))
    del per_quantiles[len(y_quantiles) // 2]
    color_quantiles = [cmap(1 - p) for p in per_quantiles]

    # Plot center
    plt.plot(x_plot, y_center, '-', color=color, linewidth=1)

    # Add ribbons
    for i, c_q in zip(range(len(y_quantiles) - 1), color_quantiles):
        q0 = y_quantiles[i]
        q1 = y_quantiles[i + 1]
        plt.fill_between(
            x_plot,
            q0,
            y2=q1,
            color=c_q,
            alpha=.7,
            linewidth=0)

    if start_date is not None:
        format_date_xaxis()
