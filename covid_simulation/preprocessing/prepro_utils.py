import re
from unicodedata import normalize
import numpy as np
import warnings
import datetime as dt
import os

DIC_COL_DTYPE = {"region": "category",
                 "date": np.datetime64,
                 "positive": "int64",
                 "admitted": "int64",
                 "uci": "int64",
                 "death": "int64",
                 "recovered": "int64"}


def homogenize_dic(dic, hom_keys=False, hom_values=False):
    """
    Homogenize keys and/or values of a dictionary.
    All keys/values must be strings.
    It is NOT in-place.
    """
    dic_ = dic.copy()
    if hom_values:
        for k, v in dic.items():
            assert isinstance(v, str), f'{v} is not a str'
            new_v = homogenize_string(v)
            dic_[k] = new_v
    if hom_keys:
        for k, v in dic.items():
            assert isinstance(k, str), f'{k} is not a str'
            new_k = homogenize_string(k)
            dic_[new_k] = dic_.pop(k)
    return dic_


def change_extension(path, new_ext, suffix=None, rm_spaces=True):
    """
    Change the extension of a path, appending suffix to the filename.
    If rm_spaces, substitutes spaces for '_' in filename.
    """
    root, ext = os.path.splitext(path)
    if suffix is not None:
        root += suffix
    new_path = root + '.' + new_ext
    if rm_spaces:
        folder, fname = os.path.split(new_path)
        fname = fname.replace(" ", "_")
        new_path = os.path.join(folder, fname)
    return new_path


def detect_future_dates(df, date_lim, filter_cases=False):
    """
    Detect which entries have a date strictly greater than date_lim.
    If filter_cases, it returns the dataframe with those cases removed.
    """
    assert isinstance(date_lim, (dt.datetime, dt.date)
                      ), 'date_lim must be a dt.datetime or dt.date object'
    mask = (
            (df.select_dtypes(include=[np.datetime64]) > date_lim).sum(
                axis=1) >= 1)
    cases = mask.sum()
    if cases > 0:
        warnings.warn(
            f'There are {cases} entries with date greater than {date_lim}.')
        print(df.select_dtypes(include=[np.datetime64])[mask])

        if filter_cases:
            df_ = df.copy()
            return df_[~mask]


def filepath_to_dt(filepath):
    date_file = filepath.split('/')[-1].split('_')[0]
    date_lim = dt.datetime.strptime('2020' + date_file, "%Y%m%d")
    return date_lim


def deaccent(string, remove_dieresis_u=True):
    """
    Eliminate all the accents from string, keeping the ñ. 
    Optionally removes dieresis in ü.

    Parameters
    ----------
    string : str
    remove_dieresis_u : bool, default=True
        If True, it removes the dieresis on the Ü and ü

    Returns
    -------
    string_deaccent : str
        Deaccent version of string.

    Extra Info
    ----------
    https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
    """

    # -> NFD
    string_decomposed = normalize("NFD", string)
    # delete accents
    if remove_dieresis_u:
        # keep the tilde on the n (ñ -> n)
        string_deaccent = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
            r"\1", string_decomposed, 0, re.I)
    else:
        # keep the tilde on the n (ñ -> n) and dieresis on the u (ü -> u)
        string_deaccent = re.sub(
            r"([^nu\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f]))|u(?!\u0308(?![\u0300-\u036f])))[\u0300-\u036f]+",
            r"\1", string_decomposed, 0, re.I)
    # -> NFC
    string_deaccent = normalize("NFC", string_deaccent)
    assert len(string_deaccent) == len(
        string), "String has different length after applying deaccent."
    return string_deaccent


def homogenize_string(string, remove_dieresis_u=True):
    """
    Lowercases and eliminates all the accents from string, keeping the ñ. 
    Optionally removes dieresis in ü.
    Eliminate spaces.

    Parameters
    ----------
    string : str
    remove_dieresis_u : bool, default=True
        If True, it removes the dieresis on the Ü and ü

    Returns
    -------
    string_deaccent : str
        Lowercase and deaccent version of the input entity.
    """
    if string is np.nan:
        string = "NaN"
    assert isinstance(string, str), f"{string} is not a str object"
    string_low = string.strip().lower().replace(" ", "")
    string_low_deaccent = deaccent(
        string_low, remove_dieresis_u=remove_dieresis_u)
    return string_low_deaccent


def undo_cumulative(df, date_col='date'):
    """
    Calculate the discrete difference for series with daily frequency.
    I.e., the inverse transformation of cumsum.
    """
    # We dont want to work on the original df
    df = df.copy()
    first_day = df["date"].min()

    mask_first_day = df["date"] == first_day

    # Store the total number of cases from original df
    last_day = df[date_col].max()
    mask_last_day = df[date_col] == last_day
    total_count = df[mask_last_day]

    # Create new dataframe to perform the difference
    # Drop date column to avoid issues when differentiating
    df_diff = df.drop("date", axis=1)
    # Get consecutive value differences: value(t) - value(t-1)
    df_diff = df_diff.groupby("region").diff()

    # The difference corresponding to the first day is a NaN
    # as no previous date exists
    # The NaNs of that first day are replaced by the values
    # of the original df
    col_diff = df_diff.columns
    df_diff[mask_first_day] = df.loc[mask_first_day, col_diff]

    # Columns to integer
    df_diff = df_diff.astype(int)
    # Add region
    df_diff["region"] = df["region"]
    # Add date
    df_diff["date"] = df["date"]

    # Reorder once again by date first
    df_diff.sort_values(["region", "date"], axis=0, inplace=True)

    # Assert we have done it right
    # Sum the number of cases per region
    new = df_diff.groupby("region").sum()
    # Make total_count DataFrame mirror the structure of new DataFrame
    total_count_ = total_count.drop("date", axis=1).set_index("region").astype(
        int)

    assert new.equals(
        total_count_), "The total number of cases in the new df isnt equal to the old one"
    return df_diff


def forward_fill_nans(df, col_region="region", col_date="date"):
    """
    Forward fill NaNs iteratively

    Parameters
    ----------
    df : pandas.DataFrame
    col_region : str, default = "region"
    col_date : str, defaukt = "date"

    Returns
    -------
    df : pandas.Dataframe
        Without NaNs!
    """
    # Don't modify the original
    df = df.copy()

    # Sort by region and date
    df.sort_values([col_region, col_date], axis=0, ascending=True,
                   inplace=True)
    # Fill the first date of each region with 0, if needed
    for region in df[col_region].unique():
        # Locate all occurrences of a region
        mask_region = df[col_region] == region
        # Find first occurrence. It will be the first date of that region (as the df is sorted by date)
        first_occurence = mask_region.idxmax()
        # Fill its missing values with 0's
        df.loc[first_occurence, :] = df.loc[first_occurence, :].fillna(0)

    # We will enter a loop, filling each NA with the previous row value containing a non-NA
    while df.isna().sum().sum() > 0:
        df.fillna(method="ffill", inplace=True)

    return df


def ensure_monotonic_increase(df, col_date="date",
                              col_region="region", fill_nans=False):
    """
    Makes sure that numeric columns in the dataframe are monotonic increasing.

    Parameters
    ----------
    df : pandas.DataFrame
    col_date : str, default="date"
    col_region : str, default="region"
    fill_nans : bool, default=False
        If True, the decreased values are replaced with the previous ones

    Returns
    -------
    df : pandas.DataFrame

    """
    # We don't want to work on the original df
    df = df.copy()

    # List columns that should be monotonic increasing
    columns_to_ensure = [k for k, v in DIC_COL_DTYPE.items() if v == "int64"]
    columns_to_ensure = [col for col in columns_to_ensure if col in df.columns]

    # Sort by region, then date
    df.sort_values([col_region, col_date], ascending=True, inplace=True)

    # To ensure each column is monotonic increasing, we will enter do the
    # following loop. For each row:
    # 1. Check the last row belonged to the  same region
    # 2. If so, check if any value has decreased
    # 3. Replace the decreased values with the ones before them OR with NA
    for idx, row in df.iterrows():
        if idx > 0 and row[col_region] == df.loc[idx - 1, col_region]:
            # Check which columns decreased
            previous_values = df.loc[idx - 1, columns_to_ensure]
            decreased = row[columns_to_ensure] < previous_values
            # List them
            decreased = decreased[decreased].index
            # Replace them
            if fill_nans:
                df.loc[idx, decreased] = df.loc[idx - 1, decreased].copy()
            else:
                df.loc[idx, decreased] = np.nan

    return df


def change_columns_dtype(df):
    """
    Changes columns dtype to the one defined in DIC_COL_DTYPE.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
    """
    # We don't want to modify the original df
    df = df.copy()

    # Filter the dictionary
    dic_col_dtype_filtered = {k: v for k, v in DIC_COL_DTYPE.items() if
                              k in df.columns}

    # Some columns are transformed to int type. If those columns have NaNs,
    # it will raise an error. To avoid it, we ensure the columns with NaNs
    # that should be converted to int are turned into float instead
    for col, type_ in dic_col_dtype_filtered.items():
        if type_ == "int64" and df[col].isna().sum() > 0:
            dic_col_dtype_filtered[col] = "float32"

    df = df.astype(dic_col_dtype_filtered)
    return df
