import argparse
import os
import pandas as pd

from covid_simulation.exceptions import DataFormatChangeError
from covid_simulation.preprocessing.prepro_utils import homogenize_string
from covid_simulation.preprocessing.prepro_utils import forward_fill_nans
from covid_simulation.preprocessing.prepro_utils import change_columns_dtype
from covid_simulation.preprocessing.prepro_utils import ensure_monotonic_increase


URL_MADRID = "https://datos.comunidad.madrid/catalogo/dataset/7da43feb-8d4d" \
             "-47e0-abd5-3d022d29d09e/resource/b2a3a3f9-1f82-42c2-89c7" \
             "-cbd3ef801412/download/covid19_tia_muni_y_distritos.csv "

DIC_COL_NAME = {"municipio_distrito": "region",
                "fecha_informe": "date",
                "casos_confirmados_totales": "positive"}


def preprocess_df_madrid(df, col_region="region", fill_nans=False,
                         ensure_increase=True):
    """
    Preprocess the raw dataframe, downloaded from URL_MADRID
    The output dataframe will follow our criteria:
        - All column names homogenized
        - Number of cases are cumulative by day

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the raw data from URL_MADRID
    col_region: string, default region
        Name of the column that identifies regions
    fill_nans : boolean, default=False
        If True, fill NaNs with 0
    ensure_increase : boolean, default=True
        If True, ensure the values are monotonic increasing

    Returns
    -------
    df : pandas.DataFrame
        Dataframe preprocessed, formatted following our criteria
    """
    # We dont want to work on the original df
    df = df.copy()

    # Homogenize column names
    df.columns = list(map(homogenize_string, df.columns))
    # Replace name
    df.rename(DIC_COL_NAME, axis=1, inplace=True)

    # Check all target columns are present
    # Target columns are the ones listed in DIC_COL_NAME values
    # As the dictionary may have more than one key for the same value,
    # we must apply a "set" to the listed values to ensure they don't repeat
    target_columns = set(DIC_COL_NAME.values())
    for column in target_columns:
        if column not in df.columns:
            raise DataFormatChangeError(f"Column {column} missing in URL data:"
                                        f"{df.columns}")

    # Take relevant columns
    df = df[target_columns]

    # Homogenize region names
    df[col_region] = df["region"].apply(homogenize_string)

    # Date format
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M:%S")

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

    return df


def download_df_madrid(url=URL_MADRID, preprocess=True,
                       col_region="region", fill_nans=False,
                       ensure_increase=True):
    """
    Downloads MADRID data as a dataframe

    Parameters
    ----------
    url : string
        URL to download the csv
    col_region: string, default region
        Name of the column that identifies regions
    preprocess : bool, default True
        If True, preprocess the dataframe after downloading it
    fill_nans : boolean, default=False
        If True, fill NaNs with forward value (only applied in preprocessing)
    ensure_increase : boolean, default=True
        If True, ensure the values are monotonic increasing

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the downloaded csv data

    References
    ----------
    https://datos.comunidad.madrid/catalogo/dataset/covid19_tia_muni_y_distritos
    """
    df = pd.read_csv(url, encoding='latin1', sep=";")

    if preprocess:
        df = preprocess_df_madrid(df, col_region=col_region,
                                  fill_nans=fill_nans,
                                  ensure_increase=ensure_increase)
    return df


if __name__ == "__main__":

    # Locate main path
    path_script = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.dirname(os.path.dirname(path_script))
    path_data = os.path.join(path_main, "data")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output',
                        help='Path to the output folder',
                        type=str,
                        default=path_data)

    args = parser.parse_args()

    madrid = download_df_madrid()

    # Get last date
    last_day = madrid["date"].max()
    last_day = last_day.strftime("%m%d")

    # Store dataframe
    if os.path.isfile(args.output):
        output = args.output
    else:
        output = os.path.join(args.output, last_day +
                              "_COVID_MADRID_MAT_dataseries.csv")

    madrid.to_csv(output)
