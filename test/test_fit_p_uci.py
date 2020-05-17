from covid_simulation.preprocessing.andalucia import download_df_andalucia
from covid_simulation.simulation.pipeline import fit_p_uci_no_saturation


if __name__ == '__main__':

    df = download_df_andalucia(fill_nans=True)
    for region in df.region.unique():
        print(f'\n{region.title()}')
        df_region = df[df.region == region]
        p_uci_0 = .15
        p_uci_best = fit_p_uci_no_saturation(df_region,
                                             col_admitted='admitted',
                                             col_target='uci', size_val=5,
                                             p_uci_0=p_uci_0, nruns=200,
                                             verbose=True,)
        print(f'\nInitial p_uci: {p_uci_0}')
        print(f'\nFitted p_uci: {p_uci_best}')
