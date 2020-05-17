from inspect import signature

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from covid_simulation.noise.noise import get_quantiles
from covid_simulation.fit.fit import fit_model_params


def format_date_xaxis(fmt='%b %d', ax=None, interval=1):
    if ax is None:
        ax = plt.gca()
    ax.xaxis_date()
    # set ticks every week
    ax.xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=6, interval=interval))
    # set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def plot_results_fit(model,
                     params,
                     x,
                     y,
                     start_date=None,
                     predicted_days=5,
                     implicit_plot=False):
    """
    Plot model fit and data points:
        * Cumulative cases vs t
        * Daily cases vs t
        * Daily cases vs Cumulative cases (Log Log scale)
    """
    assert len(params) == len(signature(model).parameters) - 1

    nplots = 3 if implicit_plot else 2

    dydx = np.diff(y)
    x_pred = np.arange(x.min() - 1, x.max() + predicted_days + 1, 1)
    y_pred = model(x_pred, *params)
    dydx_pred = np.diff(y_pred)

    x_next_day = int(x.max() + 1)
    y_next_day = int(y_pred[x_pred == x_next_day])
    delta_cases_next_day = y_next_day - int(y[-1])

    # we calculate the value in 3 months
    y_asymp = int((model(30 * 3, *params)))

    if start_date is not None:
        x_date = start_date + x * dt.timedelta(days=1)
        x_pred_date = start_date + x_pred * dt.timedelta(days=1)
        fecha_next_day = (start_date + x_next_day *
                          dt.timedelta(days=1)).strftime('%b %d')
        title = f"{fecha_next_day}: Nuevos {delta_cases_next_day} / Total {y_next_day}\nCasos asintóticos: {y_asymp:.5g}"
    else:
        fecha_next_day = 'Día siguiente'
        title = f"{fecha_next_day}: Nuevos {delta_cases_next_day} / Total {y_next_day}\nCasos asintóticos: {y_asymp:.5g}"

    plt.figure(dpi=100, figsize=(12, 4))
    plt.suptitle(title)

    # cumulativa
    plt.subplot(1, nplots, 1)
    if start_date is not None:
        plt.scatter(x_date, y)
        plt.plot(x_pred_date, y_pred, color="red")
        format_date_xaxis()
    else:
        plt.scatter(x, y)
        plt.plot(x_pred, y_pred, color="red")
    plt.xlabel(f"Dia")
    plt.ylabel("Casos acumulados")

    # casos por dia
    plt.subplot(1, nplots, 2)
    if start_date is not None:
        plt.scatter(x_date[1:], dydx)
        plt.plot(x_pred_date[1:], dydx_pred, color="red")
        format_date_xaxis()
    else:
        plt.scatter(x[1:], dydx)
        plt.plot(x_pred[1:], dydx_pred, color="red")
    plt.xlabel(f"Dia")
    plt.ylabel("Casos diarios")

    if implicit_plot:
        # curva implícita
        plt.subplot(1, nplots, 3)
        plt.scatter(y[1:], dydx)
        plt.plot(y_pred[1:], dydx_pred, color='red')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Casos totales")
        plt.ylabel("Casos diarios")


def fit_and_plot_validation(model, x, y, params0=None, val_days=5, color_val='red',
                            start_date=None, add_title=None, kwargs_pred=None):
    """
    Plot cumulative funtions for validation purposes, i.e.,
    it includes the original points
    and the and the predicted one
    on the same val_days last days.
    """
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()

    # split train/val
    x_val = x[-val_days:]
    y_val = y[-val_days:]
    x_train = x[:-val_days]
    y_train = y[:-val_days]

    val_params = fit_model_params(model, x_train, y_train, params0=params0)

    y_pred = model(x, *val_params)

    title = "Validación de modelos"
    if add_title is not None:
        title += f' ({add_title.title()})'
    plt.title(title)

    kwargs_pred_orig = {"alpha": .7, "label": model.__name__.title()}
    if kwargs_pred is not None:
        kwargs_pred_orig.update(kwargs_pred)
    if start_date is not None:
        x_train_date = start_date + x_train * dt.timedelta(days=1)
        x_date = start_date + x * dt.timedelta(days=1)
        x_val_date = start_date + x_val * dt.timedelta(days=1)
        plt.plot(x_train_date, y_train, '.', color='black', alpha=.7)
        plt.plot(x_val_date, y_val, '.', color=color_val, alpha=.7)
        plt.plot(x_date, y_pred, '-', **kwargs_pred_orig)
        format_date_xaxis()
    else:
        plt.plot(x_train, y_train, ".", color='black', alpha=.7)
        plt.plot(x_val, y_val, ".", color=color_val, alpha=.7)
        plt.plot(x, y_pred, "-", **kwargs_pred_orig)

    plt.ylabel("Ingresos hospitalarios acumulados")
    plt.xlabel('Fecha')
    plt.legend()

    return None


def plot_prediction_with_uncertainty(x_pred,
                                     y_pred,
                                     sample_noise,
                                     x_orig,
                                     y_orig,
                                     start_date=None,
                                     add_title=None,
                                     band_past=False):
    x_pred = x_pred.copy()
    # Color
    color = (0, .7, 0, .5)
    y_quantiles, y_center = get_quantiles(y_pred,
                                          sample_noise,
                                          quantiles=(.05, .25, .75, .95),
                                          add_median=True)
    # Quantile column name
    qcols = y_quantiles
    start_date = dt.datetime.fromisoformat(start_date.isoformat())
    if start_date is not None:
        x_date = start_date + x_pred * dt.timedelta(days=1)
        x_plot = x_date
        if x_orig is not None:
            x_orig_plot = start_date + x_orig * dt.timedelta(days=1)
    else:
        x_plot = x_pred
        if x_orig is not None:
            x_orig_plot = x_orig

    # Plot center
    plt.plot(x_plot, y_center, '-', color='green')
    # Add real values
    if (y_orig is not None) and (x_orig is not None):
        plt.plot(x_orig_plot, y_orig, ".", color='black')
    if start_date is not None:
        format_date_xaxis()
    if not band_past:
        mask = x_pred > x_orig.max()
        x_plot = x_plot[mask]
        y_quantiles = y_quantiles[:, mask]
    # Add ribbons
    for i in range(1, len(qcols)):
        q0 = y_quantiles[i - 1]
        q1 = y_quantiles[i]
        plt.fill_between(
            x_plot,
            q0,
            y2=q1,
            color=color,
            alpha=.5,
            #                          edgecolor="b",
            linewidth=0.0)
        # Decrease color intensity as we approach the
        # center, then increase again
        if i < len(qcols) // 2:
            color = [max(0, c - .1) for c in color]
        else:
            color = [min(c + .1, 1) for c in color]

    # Label axis
    plt.xlabel("Fecha")
    plt.ylabel("Ingresos hospitalarios acumulados")
    title = 'Predicción a una semana'
    if add_title is not None:
        title += f' ({str(add_title).title()})'
    plt.title(title)


def plot_cumulative_validation_points(model, params, x, y, val_days=5,
                                      start_date=None, plot_overlap=True):
    """
    Plot cumulative funtions for validation purposes, i.e.,
    it includes the original points and the predicted one
    on the same val_days last days.
    """
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()

    x_val = x[-val_days:]
    y_val_pred = model(x_val, *params)

    if not plot_overlap:
        x = x[:-val_days]
        y = y[:-val_days]

    plt.figure(dpi=100)
    plt.title("Cumulative cases and fit")
    if start_date is not None:
        x_date = start_date + x * dt.timedelta(days=1)
        x_val_date = start_date + x_val * dt.timedelta(days=1)
        plt.plot(x_date, y, '.', color='blue', label='Ground truth', alpha=.7)
        plt.plot(x_val_date, y_val_pred, '.',
                 color='red', alpha=.7, label='Prediction')
        format_date_xaxis()
    else:
        plt.plot(x, y, ".", color='blue', alpha=.7, label='Ground truth')
        plt.plot(x_val, y_val_pred, ".", color='red',
                 alpha=.7, label='Prediction')

    plt.ylabel("Cumulative cases")
    plt.legend()

    return y_val_pred


def plot_cumulative_validation_points(model, params, x, y, val_days=5,
                                      start_date=None, plot_overlap=True):
    """
    Plot cumulative funtions for validation purposes, i.e.,
    it includes the original points and the predicted one
    on the same val_days last days.
    """
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()

    x_val = x[-val_days:]
    y_val_pred = model(x_val, *params)

    if not plot_overlap:
        x = x[:-val_days]
        y = y[:-val_days]

    plt.figure(dpi=100)
    plt.title("Cumulative cases and fit")
    if start_date is not None:
        x_date = start_date + x * dt.timedelta(days=1)
        x_val_date = start_date + x_val * dt.timedelta(days=1)
        plt.plot(x_date, y, '.', color='blue', label='Ground truth', alpha=.7)
        plt.plot(x_val_date, y_val_pred, '.',
                 color='red', alpha=.7, label='Prediction')
        format_date_xaxis()
    else:
        plt.plot(x, y, ".", color='blue', alpha=.7, label='Ground truth')
        plt.plot(x_val, y_val_pred, ".", color='red',
                 alpha=.7, label='Prediction')

    plt.ylabel("Cumulative cases")
    plt.legend()

    return y_val_pred
