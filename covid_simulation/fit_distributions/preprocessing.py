import datetime as dt


def los_no_uci(df, last_record=dt.datetime(2020, 3, 29)):
    """
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing individual cases
    last_record : date, default 29th march 2020
        Last date when the data was recored
    
    Returns
    -------
    df : pandas DataFrame
        Datafram containing only the relevant columns,
        with a computed offset
    """
    # We dont want to work on the original df
    df = df.copy()
    
    # Locate the patients with leave date
    df["observed"] = ~df["discharge"].isna()

    # Ignore the patients that were in UCI
    mask_no_uci = df["uci"].fillna(1) == 0

    # Create a new df with relevant information
    mask_admitteds_no_nan = ~df["admitted"].isna()
    mask_admitteds_antes_febrero = df["admitted"] >= dt.datetime(2020,2,1)
    df = df[mask_admitteds_no_nan & mask_admitteds_antes_febrero & mask_no_uci].copy()
    df = df[["discharge", "admitted", "observed"]]

    # Offset between admittance date and leave
    df["offset"] = (df["discharge"] - df["admitted"]).dt.days
    # For those patients without leave date, the offset is
    # between the admittance and the time when data was last recorder
    df.loc[~df["observed"], "offset"] = (last_record - df["admitted"]).dt.days

    df = df[df["offset"] > 0]
    return df
