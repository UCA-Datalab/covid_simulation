import pandas as pd

"""
The following functions and parameters are used by the other web scripts.
"""


def read_output_csv(path_file):
    """
    Load a csv file contained in the outputs folder (that is, one already
    preprocessed) and return it as a dataframe

    Parameters
    ----------
    path_file : string
        Path to the csv file

    Returns
    -------
    outputs : pandas DataFrame
        Dataframe of the csv file

    Raises
    ------
    ValueError
        Description
    """
    # Read file
    # ground_truth.csv is a simple dataframe
    # but the others contain multi-index as columns
    if "ground_truth" in path_file:
        outputs = pd.read_csv(path_file)
    else:
        outputs = pd.read_csv(path_file, header=[0, 1], index_col=0)

        # Get unnamed columns and remove their name
        columns_unnamed = ["Unnamed: " +
                           str(i) + "_level_1" for i in range(150)]
        columns_unnamed = dict(zip(columns_unnamed, [""] * 150))
        outputs = outputs.rename(columns_unnamed, axis=1)

        # Restore date column
        outputs.reset_index(inplace=True)

    if "date" in outputs.columns:
        col = "date"
    else:
        raise ValueError(
            f"No date column in {path_file}:\n{outputs.columns}")
    outputs[col] = pd.to_datetime(outputs[col], format="%Y-%m-%d")
    return outputs
