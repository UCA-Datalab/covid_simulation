import matplotlib.pyplot as plt
import pandas as pd


"""
These functions are use by test_backend.py to plot the dictionaries that 
output the backend functions.
"""


def display_dictionary(dic_plot):
    """
    Muestra los elementos del diccionario,
    de manera sencilla.
    """
    # Show the dictionary entries
    for key, value in dic_plot.items():
        if type(value) is dict and "x" in value.keys():
            value = "< Dictionary to plot >"
        print(key, "---", value)


def plot_dictionary_simul(dic_plot, path_figure):
    """
    Grafica el diccionario
    """
    # Retrieve parameters from the dictionary
    x = pd.to_datetime(dic_plot["line"]["x"])
    y = dic_plot["line"]["y"]
    ylabel = dic_plot["ylabel"]
    xlabel = dic_plot["xlabel"]
    title = dic_plot["title"]

    plt.figure(dpi=120)

    plt.plot(x, y, color="orange")
    legend = ["Ajuste"]

    if "points" in dic_plot.keys():
        points_x = pd.to_datetime(dic_plot["points"]["x"])
        points_y = dic_plot["points"]["y"]
        plt.plot(points_x, points_y, "r.")
        legend += ["Datos empíricos"]

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=15)
    plt.title(title)
    plt.legend(legend)

    plt.savefig(path_figure)
    plt.close()


def plot_dictionary_dist(dic_plot, path_figure):
    """
    Grafica el diccionario
    """
    # Retrieve parameters from the dictionary
    x = dic_plot["line"]["x"]
    y = dic_plot["line"]["y"]
    y_label = dic_plot["ylabel"]
    x_label = dic_plot["xlabel"]
    title = dic_plot["title"]
    model = dic_plot['model']
    model_params = dic_plot["model_params"]
    mean = dic_plot['mean']
    variance = dic_plot['variance']

    plt.figure(dpi=120)

    legend = [f"Ajuste por {model}"]

    if "bars" in dic_plot.keys():
        bars_x = dic_plot["bars"]["x"]
        bars_y = dic_plot["bars"]["y"]
        plt.bar(bars_x, bars_y)
        legend += ["Datos empíricos"]

    plt.plot(x, y, color="orange")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend(legend)

    plt.savefig(path_figure)
    plt.close()

    print(f"Parámetros de la distribución de ajuste: {model}")
    for key, value in model_params.items():
        print(f"    {key} = {value}")
    print(f"Media = {mean}")
    print(f"Varianza = {variance}")
