import os

from covid_simulation.web.backend import generate_plot_data
from covid_simulation.web.backend import dic_indicators
from covid_simulation.web.dist_plots import generate_distribution_data
from covid_simulation.web.dist_plots import dic_distribution_title

from covid_simulation.test_utils import display_dictionary
from covid_simulation.test_utils import plot_dictionary_dist
from covid_simulation.test_utils import plot_dictionary_simul

REGIONS = ["andalucia",
           "almeria",
           "cadiz",
           "cordoba",
           "granada",
           "huelva",
           "jaen",
           "malaga",
           "sevilla"]

"""
Set paths
"""
path_output_folder = "./test_background"
if not os.path.exists(path_output_folder):
    raise FileNotFoundError(
        f"Route not found:\n{path_output_folder}\n Make sure you have "
        f"run test_background.py")

path_plots = "./test_backend"
# Create directory if it doesn't exists
if not os.path.exists(path_plots):
    os.mkdir(path_plots)

"""
Results
"""
print("\n------- RESULTS -------\n")

# We test `generate_plot_data` function for each region listed in REGIONS
# and each available indicator
for region in REGIONS:
    for indicator in dic_indicators.keys():
        print(f"\n{region} - {indicator}\n")
        path_figure = path_plots + "/" + region + "_" + indicator + ".png"

        # Outputs
        print(
            f"Generating plot data for {indicator} in {region}...\n")
        dic_plot = generate_plot_data(path_output_folder, region,
                                      indicator)

        # Visualization
        print("These are the plot parameters:\n")
        display_dictionary(dic_plot)
        print("\nStoring plot...\n")
        plot_dictionary_simul(dic_plot, path_figure)
        print("Done!\n")

"""
Time distributions
"""
print("\n---- TIME DISTRIBUTIONS ----\n")

# We test `generate_distribution_data` for each available
# distribution
for distribution_type in dic_distribution_title.keys():
    path_figure = path_plots + "/" + distribution_type + ".png"
    print(f"Generating distribution data for {distribution_type}...\n")
    dic_plot = generate_distribution_data(distribution_type)

    # Visualization
    print("These are the plot parameters:\n")
    display_dictionary(dic_plot)
    print("\nStoring plot...\n")
    plot_dictionary_dist(dic_plot, path_figure)
    print("\nDone!\n")
