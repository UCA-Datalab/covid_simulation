from covid_simulation.web.background import download_fit_and_simulate_andalucia
from covid_simulation.web.background import check_new_url

path_output_folder = './test_background'

check_new_url(path_output_folder)

download_fit_and_simulate_andalucia(path_output_folder, save_plots=True,
                                    verbose=True)
