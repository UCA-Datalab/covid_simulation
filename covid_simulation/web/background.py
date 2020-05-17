"""
Summary

Attributes
----------
dic_p_uci_0 : TYPE
    Description
"""
import os
import pandas as pd
import warnings

from covid_simulation.fit.predict import predict_and_append
from covid_simulation.preprocessing.andalucia import download_df_andalucia
from covid_simulation.simulation.pipeline import fit_p_uci_no_saturation
from covid_simulation.simulation.pipeline import perform_simulations_and_store

from covid_simulation.web.web_utils import read_output_csv

from covid_simulation.pandas_utils import df_to_datetime_ser

from covid_simulation.config import PARAMS_FIT_P_UCI
from covid_simulation.config import PARAMS_PREDICT
from covid_simulation.config import PARAMS_SIMULATION
from covid_simulation.config import DIC_MAX_BEDS_UCI

"""
These functions are used periodically by the web application to check if new 
data is available. If so, this script also downloads and simulates with the 
new data.
"""

dic_p_uci_0 = {
    'almeria': 0.2,
    'andalucia': 0.13555555555555554,
    'cadiz': 0.16075471698113206,
    'cordoba': 0.13777777777777778,
    'españa': 0.10829001367989056,
    'granada': 0.12208695652173912,
    'huelva': 0.1636363636363636,
    'jaen': 0.10478873239436619,
    'malaga': 0.1248,
    'sevilla': 0.1459030837004405
}


def check_new_url(path_output_folder):
    """
    Download the csv from given url and check if our data is up to date
    with the csv

    Parameters
    ----------
    path_output_folder : str
        Path to the outputs folder, where the web stores all the relevant csv

    Returns
    -------
    boolean
        True is new data is available, False otherwise

    Raises
    ------
    ValueError
        Description
    """
    # Download csv
    # TODO: Narrow down the possible Exceptions raised when downloading
    try:
        database = download_df_andalucia(fill_nans=True)
    except Exception:
        database = None

    # Cancel if data is not available
    if database is None:
        print("Something went wrong when downloading the new data")
        print("Cancelling update")
        return False

    assert "date" in database.columns, f"'date' missing in database " \
                                       f"columns:\n{database.columns} "

    # List our current data
    files = os.listdir(path_output_folder)
    # Search for csv
    files = [f for f in files if f.endswith(".csv")]

    # If our folder is empty, we have new info to add
    if len(files) == 0:
        return True

    # Take the first csv we have
    path_file = os.path.join(path_output_folder, files[0])
    # Open it
    outputs = read_output_csv(path_file)

    # Compare dates
    database_date = database["date"].max()
    if "date" in outputs.columns:
        col = "date"
    else:
        raise ValueError(
            f"No date column in outputs: {outputs.columns}")
    outputs_date = outputs[col].max()

    # Remember that outputs dates are `predicted_days`
    # ahead of real ones
    outputs_date -= pd.Timedelta(days=PARAMS_PREDICT["predicted_days"])

    # If there is new dates, we have data to add
    if database_date > outputs_date:
        return True
    else:
        return False


def download_fit_and_simulate_andalucia(path_output_folder,
                                        save_plots=False,
                                        verbose=False):
    """
    Downloads data from Andalucia, fit model to admitted,
    and simulate uci filling.

    **TODO:** only add noise to predicted points
    **TODO:** adapt to work with nans

    Parameters
    ----------
    path_output_folder : str
        Path to the outputs folder, where the web stores all the relevant csv
    save_plots : bool, default=False
        Make and save plots to `path_output_folder`
    verbose : bool, default=False
        Print what the function is doing at every moment
    """
    # We must fill NaN values or else it will raise Errors when
    # simulating
    dataset = download_df_andalucia(fill_nans=True)

    # Store it
    if not os.path.exists(path_output_folder):
        os.mkdir(path_output_folder)
    path_store = os.path.join(path_output_folder, "ground_truth.csv")
    dataset.to_csv(path_store)

    # we don't want to compute for 'españa' in production
    list_region = dataset["region"].unique()
    list_region = [region for region in list_region if
                   region != 'españa']
    for region in list_region:
        print(f'\nCalculando en {region}...')
        # Set output csv path
        path_results_csv = os.path.join(path_output_folder,
                                        region + ".csv")
        if save_plots:
            path_plots = os.path.join(path_output_folder,
                                      region + "_plots")
            if not os.path.exists(path_plots):
                os.mkdir(path_plots)
        else:
            path_plots = None

        df_region = dataset[dataset.region == region]
        ser_admitted = df_to_datetime_ser(
            df_region, 'admitted', assert_filled=True)
        ser_uci = df_to_datetime_ser(df_region, 'uci',
                                     assert_filled=False)

        # PREDICT FUTURE VALUES OF ADMITTED

        ser_admitted_with_pred = predict_and_append(ser_admitted,
                                                    start_date=
                                                    PARAMS_PREDICT[
                                                        'start_date'],
                                                    dic_model=
                                                    PARAMS_PREDICT[
                                                        'dic_model'],
                                                    pred_size=
                                                    PARAMS_PREDICT[
                                                        'predicted_days'],
                                                    fit_model=True)

        # FIT P_UCI

        # set `p_uci_0`
        if region in dic_p_uci_0.keys():
            p_uci_0 = dic_p_uci_0[region]
        else:
            p_uci_0 = PARAMS_FIT_P_UCI['p_uci_0_default']
        p_uci = fit_p_uci_no_saturation(ser_admitted, ser_uci,
                                        size_val=5,
                                        p_uci_0=p_uci_0,
                                        nruns=200,
                                        verbose=False)
        if verbose:
            print(f'Fitted value of `p_uci`:{p_uci:.3f}')

        # RUN SIMULATION

        # set `max_beds_uci`
        if region in DIC_MAX_BEDS_UCI.keys():
            max_beds_uci = DIC_MAX_BEDS_UCI[region]
        else:
            max_beds_uci = PARAMS_SIMULATION['max_beds_uci_default']
            warnings.warn(
                f"{region} missing in `DIC_MAX_BEDS_UCI keys`. use "
                f"`{max_beds_uci}` instead")

        max_beds_hosp = PARAMS_SIMULATION['max_beds_hosp']
        nruns = PARAMS_SIMULATION['nruns']
        # run the simulation and store the outputs in a csv
        perform_simulations_and_store(ser_admitted_with_pred,
                                      nruns=nruns,
                                      max_beds_uci=max_beds_uci,
                                      max_beds_hosp=max_beds_hosp,
                                      path_csv=path_results_csv,
                                      plots_title=region.title(),
                                      p_uci=p_uci,
                                      path_plots=path_plots,
                                      ser_uci=ser_uci,
                                      verbose=verbose)
