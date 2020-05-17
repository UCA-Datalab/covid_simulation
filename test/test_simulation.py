import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path


from covid_simulation.simulation.sampling import noise_on_cases
from covid_simulation.simulation.sampling import noise_on_times
from covid_simulation.simulation.sampling import propagate_times_hospital
from covid_simulation.simulation.sim_filling import simulate_bed_filling
from covid_simulation.simulation.utils import from_counts_to_times

from covid_simulation.noise.noise import get_quantiles
from covid_simulation.plot_utils import plot_prediction_with_uncertainty


if __name__ == "__main__":
    np.random.seed(0)
    x = np.arange(0, 20, 1)
    y = np.random.randint(100, 300, size=x.shape).cumsum()
    x_grid = np.arange(0, x.max() + .5, .5)
    # print(f"Numero total pacientes {y[-1]}")

    MAX_BEDS_HOSP = None
    MAX_BEDS_UCI = 200
    NRUNS = 500
    day = dt.datetime.fromisoformat('2020-03-01')

    print('\nSampling from number of people and random arrival distributions...')
    y_noise = noise_on_cases(y, nruns=NRUNS)
    times = from_counts_to_times(x, y_noise)
    times = noise_on_times(times)
    print('Done!')

    print('\nSampling from length of stay distributions...')
    (times_hosp_in, times_hosp_out, times_uci_in,
     times_uci_out) = propagate_times_hospital(times)
    print('Done!')

    print('\nImposing max beds contraint in UCI 1...')
    uci_in, uci_out, uci_active = simulate_bed_filling(x_grid,
                                                       times_uci_in,
                                                       times_uci_out,
                                                       beds_max=MAX_BEDS_UCI,
                                                       verbose=True,
                                                       try_heuristic=True)
    print('Done!')

    print('\nImposing max beds contraint in UCI 2...')
    uci_in, uci_out, uci_active = simulate_bed_filling(x_grid,
                                                       times_uci_in,
                                                       times_uci_out,
                                                       beds_max=MAX_BEDS_UCI,
                                                       verbose=True,
                                                       try_heuristic=False)
    print('Done!')
    print('\nImposing max beds contraint in hospital 1...')
    hosp_in, hosp_out, hosp_active = simulate_bed_filling(x_grid,
                                                          times_hosp_in,
                                                          times_hosp_out,
                                                          beds_max=2200,
                                                          verbose=True,
                                                          try_heuristic=True)
    print('Done!')

    print('\nImposing max beds contraint in hospital 2...')
    hosp_in, hosp_out, hosp_active = simulate_bed_filling(x_grid,
                                                          times_hosp_in,
                                                          times_hosp_out,
                                                          beds_max=MAX_BEDS_HOSP,
                                                          verbose=True,
                                                          try_heuristic=True)
    print('Done!')

    print('\nImposing max beds contraint in hospital 3...')
    hosp_in, hosp_out, hosp_active = simulate_bed_filling(x_grid,
                                                          times_hosp_in,
                                                          times_hosp_out,
                                                          beds_max=MAX_BEDS_HOSP,
                                                          verbose=True,
                                                          try_heuristic=False)
    print('Done!')

    path_plots = Path(os.path.abspath(__file__)).parent / 'test_plots'
    if not os.path.exists(path_plots):
        os.mkdir(path_plots)

    dic = {"hosp_in": hosp_in, "hosp_out": hosp_out, "hosp_active": hosp_active,
           "uci_in": uci_in, "uci_out": uci_out, "uci_active": uci_active}
    for name, quantity in dic.items():
        quan, median = get_quantiles(quantity, axis=1)
        plt.figure(dpi=100)

        plot_prediction_with_uncertainty(x_grid, median, quan,
                                         start_date=day)
        plt.title(name)
        plt.xlabel('Fecha')
        plt.ylabel(name)
        plt.savefig(path_plots / f'{name}.png')
