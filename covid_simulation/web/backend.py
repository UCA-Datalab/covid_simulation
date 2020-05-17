import os

import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from glob import glob

from covid_simulation.numpy_utils import dt_array_to_str_list
from covid_simulation.config import PARAMS_PREDICT
from covid_simulation.config import PARAMS_PLOTS_PREDICTION
from covid_simulation.config import LIST_ANDALUCIA
from covid_simulation.config import DIC_MAX_BEDS_UCI

from covid_simulation.web.web_utils import read_output_csv

"""
The following functions are used in backend to retrieve and plot the data.
"""

dic_indicators = {"uci_in_diary": "Ingresos UCI diarios",
                  "uci_in": "Ingresos UCI acumulados",
                  "uci_active": "Camas ocupadas en UCI",
                  "hosp_in": "Ingresos no UCI acumulados",
                  "hosp_in_diary": "Ingresos no UCI diarios",
                  "hosp_active": "Camas ocupadas no UCI"
                  }


def _ground_truth(path_output_folder, region, indicator):
    """
    Ground truth data points on UCI admitted
    """
    # Read ground truth data
    path_file = os.path.join(path_output_folder, "ground_truth.csv")
    # Check the data exists
    if not os.path.isfile(path_file):
        raise Exception(f"File not found: {path_file}")

    df = pd.read_csv(path_file)

    assert "region" in df.columns, f"region not in df columns:\n {df.columns}"
    assert "uci" in df.columns, f"uci not in df columns:\n{df.columns}"
    assert "date" in df.columns, f"date not in df columns:\n {df.columns}"
    df['date'] = pd.to_datetime(df['date'])
    # filter by date
    date_min = PARAMS_PLOTS_PREDICTION['date_min']
    df = df.loc[df['date'] >= date_min]

    # Take relevant region and values
    mask_region = df["region"] == region
    df = df.loc[mask_region, :]

    x = df["date"].to_numpy()
    y = df["uci"].to_numpy()

    # If data is diary, we must update Y
    if indicator == "uci_in_diary":
        y = np.diff(y)
        assert (y < 0).sum() == 0, f"{path_file} cumulative values in 'uci' " \
                                   f"column decrease! "
        # one entry less
        x = x[1:]
        assert x.shape == y.shape

    x = dt_array_to_str_list(x)
    y = y.tolist()

    dic = {"y": y, "x": x}
    return dic


def _generate_dic_plot(df, indicator, path_output_folder, region):
    """
    Parameters
    -------------
    df : pandas DataFrame
        Dataframe containing  daily indicators (admitted, occupancy...) and
        metrics for each of them (median, quantiles) Its columns are
        multi-index, in the form (indicator, metric). i.e. (admitted, median),
        (occupancy, q95)...
    indicator : str
        Quantity to plot (uci_in, uci_active, hosp_in...)

    Returns
    ----------
    dic_plot :  dict
        Dictionary containing all the  information to be included in a plot.
        Refer to generate_plot_data to see whole explanation.
    """
    assert type(
        df.columns) == pd.core.indexes.multi.MultiIndex, "df columns must " \
                                                         "be multi-index"
    assert "date" in df.columns, f"Column 'date' not in df columns:\n" \
                                 f" {df.columns}"
    assert is_datetime(df["date"]), "df date column must be datetime"
    assert indicator in df.columns, f"Column {indicator} not in df " \
                                    f"columns:\n{df.columns} "

    # We create a dataframe with the metrics of our target indicator and
    # the date
    data = df[indicator].copy()
    data["date"] = df["date"].copy()
    # Filter by date
    date_min = PARAMS_PLOTS_PREDICTION['date_min']
    data = data.loc[data["date"] >= date_min]

    # X axis will be the same in every plot
    x = data["date"].to_numpy()
    x = dt_array_to_str_list(x)

    # MEDIAN
    assert "median" in data.columns, f"({indicator}, median) not in " \
                                     f"df columns:\n{df.columns} "
    y = data["median"].values.tolist()
    dic_line = {"y": y, "x": x}

    # QUANTILES There are several quantiles. We will store their
    # values on the dictionary dic_quantiles
    quantiles = [q for q in data.columns if q.startswith("q")]
    quantiles = sorted(quantiles)
    dic_quantiles = {}
    for q in quantiles:
        assert q in data.columns, f"({indicator}, {q}) not in df " \
                                  f"columns:\n{df.columns} "
        y = data[q].values.tolist()
        dic_quantiles[q] = {"y": y, "x": x}

    # VLINE
    predicted_days = PARAMS_PREDICT["predicted_days"]
    vline_x = x[-predicted_days]

    # The vertical line starts at the lowest value of the lowest
    # quantile, and then some less
    y0 = min(dic_quantiles[quantiles[0]]["y"])
    y0 = max([0, int(y0 / 1.25)])

    # The vertical line ends at the highest value of the highest
    # quantile, and then some more
    y1 = max(dic_quantiles[quantiles[-1]]["y"])
    y1 = int(y1 * 1.25)

    dic_vline = {"x": vline_x, "y0": y0, "y1": y1}

    # GROUND-TRUTH POINTS
    if indicator in ["uci_in_diary", "uci_in"]:
        dic_points = _ground_truth(
            path_output_folder, region, indicator)

        # If points where added, we may need to change vline
        y1 = dic_vline["y1"]
        max_point = max(dic_points["y"])
        y1 = max([y1, int(max_point * 1.25)])
        dic_vline["y1"] = y1
    else:
        dic_points = None

    # HLINE
    if indicator in ["uci_active"] and region == "andalucia":
        y = int(DIC_MAX_BEDS_UCI[region])
        x0 = dic_line["x"][0]
        x1 = dic_line["x"][-1]
        dic_hline = {"y": y,
                     "x0": x0,
                     "x1": x1}

        # If points where added, we may need to change vline
        y1 = dic_vline["y1"]
        max_beds_uci = DIC_MAX_BEDS_UCI[region]
        y1 = max([y1, int(max_beds_uci * 1.25)])
        dic_vline["y1"] = y1
    else:
        dic_hline = None

    # DICTIONARY
    dic_plot = {"title": region,
                "ylabel": dic_indicators[indicator],
                "xlabel": "Fecha",
                "line": dic_line,
                "vline": dic_vline}

    dic_plot.update(dic_quantiles)

    if dic_points is not None:
        dic_plot["points"] = dic_points

    if dic_hline is not None:
        dic_plot["hline"] = dic_hline

    return dic_plot


def generate_plot_data(path_output_folder, region, indicator):
    """
    Parameters
    --------------
    path_output_folder : string
        Path to the outputs folder, where the web stores
        all the relevant csv
    region : string
        Region of interest (Autonomous Community or province) to plot
    indicator: string
        Quantity to plot (patients admitted in hospital, occupancy...)

    Returns
    ----------
    dic_plot : Dictionary of strings and arrays
        Dictionary containing all the information to be included in a plot.
        It's keys are:
        title: str
            Title of the plot
        xlabel : str
            Label for the x-axis
        ylabel : str
            Label for the y-axis
        line : dict
            Line coordinates {"x": list, "y": list}
            Shows the median of our simulation
        vline : dict
            Vertical line coordinates {"x": int, "y0": int, "y1": int}
            y0 and y1 define the height of the line (bottom and top)
            Marks the first predicted day (first date without real data)
        q* : dict, several of them
            Quantile coordinates {"x": list, "y": list}
        points : dict, optional
            Point coordinates {"x": list, "y": list}
            Ground-truth data, in contrast to our simulation
        hline : dict, optional
            Horizontal line coordinates {"x0": int, "x1": int, "y": int}
            x0 and x1 define the width of the line (left and right)
            Marks the maximum value that can reach the line. Currently it is
            only used to show the maximum bed capacity
    """
    if indicator not in dic_indicators.keys():
        raise ValueError(
            "`indicator` can only be one of these:\n"
            f" {dic_indicators.keys()}")

    # Read the csv containing given region
    path_file = os.path.join(path_output_folder, region + ".csv")
    try:
        df = read_output_csv(path_file)
    except NameError:
        print(f"`region` {region} is not available")
        available_files = os.listdir(path_output_folder)
        available_files = [s.split(".csv")[0]
                           for s in available_files if
                           s.endswith(".csv")]
        print(f"Available regions are:\n{available_files}")

    # Return data according to type
    dic_plot = _generate_dic_plot(df, indicator, path_output_folder,
                                  region)

    return dic_plot


def parse_geo_dict(path_results_folder):
    """
    parse dictionary with predicted values at last day of all quantities
    per region from `region`+'.csv' files produced when running
    `web.background.download_fit_and_simulate_andalucia`.

    keys are regions and also 'date', values are dict with
    {'quantity':value}

    Parameters
    ----------
    path_results_folder : str
        path to folder containing the csv's

    Returns
    -------
    dic_results_total : dict {str:dict}
        dictionary formatted to use in geo plot
    """
    dic_results_total = {}
    list_csvs = glob(os.path.join(path_results_folder, '*.csv'))
    for path_csv in list_csvs:
        print(path_csv)
        fname = os.path.basename(path_csv).split('.csv')[0]
        if fname not in LIST_ANDALUCIA:
            continue
        df_results = read_output_csv(path_csv)
        ser_last_median = df_results.loc[
                          :, (slice(None), 'median')].iloc[-1][:, 'median']
        dic_region = ser_last_median.astype('uint32').to_dict()
        date = df_results.date.iloc[-1]
        dic_results_total[fname] = dic_region
    dic_results_total['date'] = date.date().isoformat()
    return dic_results_total
