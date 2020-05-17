"""Summary
"""
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime


def assert_filled_in_dates(ser):
    """
    assert that a pandas.Series with datetime index has values for all days
    in between the minimum and maximum one.

    Parameters
    ----------
    ser : pandas.Series
        Series with datetime index
    """
    assert is_datetime(ser.index), 'ser index must be of dtype datetime'
    ndays = (ser.index.max() - ser.index.min()).days + 1
    assert ndays == len(ser), "there are gaps in dates index"


def df_to_datetime_ser(df, col_values, col_date='date', assert_filled=False):
    """
    Obtain the column `col_values` in `df` as a series with
    datetime index from `col_date`.

    Parameters
    ----------
    df : pandas.DataFrame
        Description
    col_values : str
        Column name with values
    col_date : str, default='date'
        Column name with datetime index
    assert_filled : bool, default=False
        Assert that all dates in between have values

    Returns
    -------
    ser : pandas.Series
        Resulting series
    """
    assert col_values in df.columns
    assert col_date in df.columns
    assert is_datetime(df[col_date])

    ser = pd.Series(df[col_values].values, df[col_date])
    # the index must have no duplicate entries
    assert ser.index.duplicated().sum() == 0
    ser.sort_index(inplace=True)
    if assert_filled:
        assert_filled_in_dates(ser)
    return ser
