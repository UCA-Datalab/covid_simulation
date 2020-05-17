import argparse
import os
import pandas as pd

from covid_simulation.exceptions import DataFormatChangeError
from covid_simulation.preprocessing.prepro_utils import homogenize_string
from covid_simulation.preprocessing.prepro_utils import forward_fill_nans
from covid_simulation.preprocessing.prepro_utils import change_columns_dtype

URL_ISCIII = "https://covid19.isciii.es/resources/serie_historica_acumulados" \
             ".csv "

DIC_COL_NAME = {"ccaacodigoiso": "region",
                "ccaa": "region",
                "fecha": "date",
                "casos": "positive",
                "hospitalizados": "admitted",
                "uci": "uci",
                "fallecidos": "death",
                "recuperados": "recovered"}

dic_region = {"an": "Andalucía",
              "ar": "Aragón",
              "as": "Asturias",
              "ib": "Baleares",
              "cn": "Canarias",
              "cb": "Cantabria",
              "cm": "Castilla La Mancha",
              "cl": "Castilla y León",
              "ct": "Cataluña",
              "ce": "Ceuta",
              "vc": "Comunidad Valenciana",
              "ex": "Extremadura",
              "ga": "Galicia",
              "md": "Madrid",
              "me": "Melilla",
              "mc": "Murcia",
              "nc": "Navarra",
              "pv": "País Vasco",
              "ri": "La Rioja"}

list_region = dic_region.values()


def preprocess_df_isciii(df, col_region="region", fill_nans=False):
    """
    Preprocess the raw dataframe, downloaded from URL_ISCIII
    The output dataframe will follow our criteria:
        - All column names homogenized
        - Number of cases are cumulative by day

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the raw data from URL_ISCIII
    col_region: string, default region
        Name of the column that identifies regions
    fill_nans : boolean, default=False
        If True, fill NaNs with 0

    Returns
    -------
    df : pandas.DataFrame
        Dataframe preprocessed, formatted following our criteria
    """
    # No queremos trabajar en el original
    df = df.copy()

    # Homogenizamos el nombre original de las columnas
    df.columns = list(map(homogenize_string, df.columns))
    # Reemplazamos el nombre
    df.rename(DIC_COL_NAME, axis=1, inplace=True)

    # Check all target columns are present
    # Target columns are the ones listed in DIC_COL_NAME values
    # As the dictionary may have more than one key for the same value,
    # we must apply a "set" to the listed values to ensure they don't repeat
    target_columns = set(DIC_COL_NAME.values())
    for column in target_columns:
        if column not in df.columns:
            raise DataFormatChangeError(
                f"Column {column} missing in URL data: {df.columns}")

    # Take target columns
    df = df[target_columns]

    # Estandarizamos el nombre original de las comunidades
    df["region"] = df["region"].apply(homogenize_string)
    # Obtenemos el nombre completo de cada región a partir del original
    df[col_region + "_full"] = df["region"].replace(dic_region)
    # region_full contiene el nombre completo de las regiones (propiamente
    # escrito). El nombre original son sólo dos letras. Para facilitar su
    # identificación, sustituimos las dos letras por el nombre completo
    # homogeneizado
    df[col_region] = df[col_region + "_full"].apply(homogenize_string)

    # Descartamos las filas que no se corresponden con comunidades
    # (Las dos últimas, que son texto informativo)
    mask_region = df[col_region + "_full"].isin(dic_region.values())
    # Dataframe sin el texto informativo
    df = df[mask_region]

    # Parseamos las fechas
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Fill NaNs
    if fill_nans:
        df = forward_fill_nans(df)

    # Format columns to proper type
    df = change_columns_dtype(df)

    return df


def download_df_isciii(url=URL_ISCIII, col_region="region", preprocess=True,
                       fill_nans=False):
    """
    Downloads ISCIII data as a dataframe

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

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the downloaded csv data

    References
    ----------
    https://covid19.isciii.es/
    """
    df = pd.read_csv(url, encoding='latin1')

    if preprocess:
        df = preprocess_df_isciii(df, col_region=col_region,
                                  fill_nans=fill_nans)

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

    isciii = download_df_isciii()

    # Get last date
    last_day = isciii["date"].max()
    last_day = last_day.strftime("%m%d")

    # Store dataframe
    if os.path.isfile(args.output):
        output = args.output
    else:
        output = os.path.join(args.output, last_day +
                              "_COVID_ISCIII_MAT_dataseries.csv")

    isciii.to_csv(output)
