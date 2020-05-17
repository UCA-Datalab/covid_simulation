import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd

from covid_simulation.preprocessing.prepro_utils import change_columns_dtype

from covid_simulation.exceptions import DataFormatChangeError
from covid_simulation.exceptions import FileDownloadError
from covid_simulation.preprocessing.prepro_utils import ensure_monotonic_increase
from covid_simulation.preprocessing.prepro_utils import forward_fill_nans
from covid_simulation.preprocessing.prepro_utils import homogenize_string
from urllib.error import HTTPError

URL_ANDALUCIA = "https://www.juntadeandalucia.es" \
                "/institutodeestadisticaycartografia/badea/stpivot/stpivot" \
                "/Print?cube=a2a26120-d774-4fe0-b3f1-202eb541551f&type=3" \
                "&foto=si&ejecutaDesde=&codConsulta=38228&consTipoVisua=JP "

DIC_COL_NAME = {"fecha": "date",
                "territorio": "region",
                "confirmados": "positive",
                "hospitalizados": "admitted",
                "fallecimientos": "death",
                "curados": "recovered",
                "totaluci": "uci",
                "nuevoscasos": "positive_daily"}


def _remove_outliers(df, col_region="region", col_date="date"):
    """
    Remove handpicked outliers.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """
    # UCI
    # Cordoba, 3th april 2020
    df.loc[(df[col_region] == "cordoba") & (df[col_date]
                                            == dt.datetime(2020, 4, 3)), "uci"] = np.nan
    # Huelva, 3th april 2020
    df.loc[(df[col_region] == "huelva") & (df[col_date]
                                           == dt.datetime(2020, 4, 3)), "uci"] = np.nan
    return df


def preprocess_df_andalucia(df, col_region="region",
                            fill_nans=False, ensure_increase=True,
                            remove_outliers=True):
    """
    Preprocess the raw dataframe, downloaded from URL_ANDALUCIA
    The output dataframe will follow our criteria:
        - All column names homogenized
        - Number of cases are cumulative by day

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the raw data from URL_ANDALUCIA
    col_region: string, default region
        Name of the column that identifies regions
    fill_nans : boolean, default=False
        If True, fill NaNs with forward value (only applied in preprocessing)
    ensure_increase : boolean, default=True
        If True, ensure the values are monotonic increasing
    remove_outliers : boolean, default=True
        If True, remove outliers

    Returns
    -------
    df : pandas.DataFrame
        Dataframe preprocessed, formatted following our criteria
    """
    # Homogenize column names
    df.columns = list(map(homogenize_string, df.columns))

    # Check all relevant columns are present
    target_columns = ["valor", "fecha", "territorio", "medida"]
    for column in target_columns:
        if column not in df.columns:
            raise DataFormatChangeError(f"Column {column} missing in URL data:"
                                        f"{df.columns}")

    # The original data has many rows for each combination of date-region
    # Each of those rows correspond to a case (admitted, uci...), which is
    # identified by "medida" column We want a dataframe with one single row
    # for each combination of date-region. The different cases must be
    # accounted in different columns, not in different rows. To change from
    # the original format to our criteria, we pivot the table
    df = df.pivot_table(values=["valor"], index=["fecha", "territorio"],
                        columns=["medida"], aggfunc='first')
    # After pivoting the dataframe, we obtain the desired table, but it has
    # an additional level in the column index We drop it
    df.columns = df.columns.droplevel(0)
    df.reset_index(inplace=True)

    # Homogenize the new column names (after pivot)
    df.columns = list(map(homogenize_string, df.columns))
    # Rename columns
    df.rename(DIC_COL_NAME, axis=1, inplace=True)

    # Drop columns with no date
    mask_nan = df["date"].isna()
    df = df[~mask_nan].copy()

    # Date format
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Keep original province name
    df[col_region + "_full"] = df["region"].copy()
    # Homogenize provinces
    df[col_region] = df["region"].apply(homogenize_string)

    # Sort by region, date
    df.sort_values([col_region, "date"], axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove outliers
    if remove_outliers:
        df = _remove_outliers(df, col_region=col_region)

    # Fill NaNs
    if fill_nans:
        df = forward_fill_nans(df)

    # Convert columns to proper type
    df = change_columns_dtype(df)

    # Ensure values are monotonic increasing
    if ensure_increase:
        df = ensure_monotonic_increase(df,
                                       col_region=col_region, col_date="date",
                                       fill_nans=fill_nans)

    assert 'date' in df.columns
    return df


def download_df_andalucia(url=URL_ANDALUCIA, col_region="region",
                          preprocess=True, fill_nans=False,
                          ensure_increase=True, remove_outliers=True):
    """
    Downloads Andalucia data as a dataframe

    Parameters
    ----------
    url : string
        URL to download the csv
    col_region: string, default region
        Name of the column that identifies regions
    preprocess : bool, default True
        If True, preprocess the dataframe after downloading it
    fill_nans : bool, default=False
        If True, fill NaNs with forward value (only applied in preprocessing)
    ensure_increase : bool, default=True
        If True, ensure the values are monotonic increasing
    remove_outliers : bool, default=True
        If True, remove outliers

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the downloaded csv data

    References
    ----------
    https://www.juntadeandalucia.es/institutodeestadisticaycartografia/salud/index.htm
    """
    # Load from url
    try:
        df = pd.read_csv(url, sep=";")
    except HTTPError:
        raise FileDownloadError(f"URL is not available:\n{url}")

    # Preprocess the dataframe if asked to do so
    if preprocess:
        df = preprocess_df_andalucia(
            df, col_region=col_region, fill_nans=fill_nans,
            ensure_increase=ensure_increase, remove_outliers=remove_outliers)

    return df


if __name__ == "__main__":

    # Locate main path
    path_script = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.dirname(os.path.dirname(path_script))
    path_data = os.path.join(path_main, "data")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output',
                        help='Path to the output folder',
                        type=str,
                        default=path_data)

    args = parser.parse_args()

    andalucia = download_df_andalucia()

    # Get last date
    last_day = andalucia["date"].max()
    last_day = last_day.strftime("%m%d")

    # Store dataframe
    if os.path.isfile(args.output):
        output = args.output
    else:
        output = os.path.join(args.output, last_day +
                              "_COVID_ANDALUCIA_MAT_dataseries.csv")

    andalucia.to_csv(output)
