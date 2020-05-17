# COVID Occupancy Simulation

This package performs three main task:

1. [Preprocess](#download-and-preprocessing) hospital data of Covid-19 patients.
See the documentation below to know the available data sources.

2. [Predict](#fit) the number of patients during the next n-days.

3. [Simulate](#simulation) the bed and UCI (intesive care unity) occupancy of those
 patients,
checking if every one of them can be provided a bed.
This simulation is based on the one performed by NHS BNSSG Analytics,
[https://github.com/nhs-bnssg-analytics/covid-simr](https://github.com/nhs-bnssg-analytics/covid-simr).

## Install the package

:warning: TODO - Explain how to install this package


## Download and preprocessing

Here we list all the available data sources for this package and how to
download and preprocess them.

A preprocessed dataframe groups data by region and date. It contains the
cumulative daily values of one or more indicators (positive cases, recovered
patients, etc.). Below we list the possible columns. Apart from **date**
and **region**, all the other columns (which are the indicators) are not
 assured to appear in every data source:

- **date** (date, %Y-%m%-%d).
- **region** (str). Place where the indicators were collected. It may be a
country, a province or even a district.
- **positive** (int). Total number of positive Covid-19 cases recorded in
 the region up to the indicated date.
- **admitted** (int). Total number of patients admitted to regional hospitals
 up to the indicated date.
- **uci** (int). Total number of patients that needed intensive cares in the
region up to the indicated date.
- **death** (int). Total number of deceased patients in the region up to
 the indicated date.
- **recovered** (int). Total number of cases that have tested negative in
Covid-19 tests, having tested positive in the past.
- **discharged** (int). Total number of patients that have left regional
 hospitals up to the indicated date. The may or may not be recovered.


### Andalucia provinces

[The following link](https://www.juntadeandalucia.es/institutodeestadisticaycartografia/badea/operaciones/consulta/anual/38228?CodOper=b3_2314&codConsulta=38228)
contains official hospital data of each Andalucia province. It is updated daily.
To download and preprocess the data, use the following commands:

```
from covid_simulation.preprocessing.andalucia import download_df_andalucia

df = download_df_andalucia()
```

There are some additional parameters that can be added to the
 `download_df_andalucia` function:
 
- `preprocess=True`. By default, the output dataframe is not only downloaded,
but also preprocessed. To get the dataframe in its raw form, change this
 parameter to `preprocess=False`.
- `fill_nans=False`. By default, NA values are not filled during preprocessing.
Feeding NA values to [fit](#fit) and [simulation](#simulation) scripts may
raise errors. To fill NA values, set `fill_nans=True`.

Alternatively, running the script
`python covid_simulation/preprocessing/andalucia.py --output OUTPUT_PATH`
will automatically download and preprocess the data, and then store it in
 given `OUTPUT_PATH` as a csv.

### Spain Autonomous Communities

[The following link](https://covid19.isciii.es/) contains official hospital data
from each Autonomous Community in Spain.

All the rules stated for [Andalucia](#andalucia-provinces) also apply to
 this script. For instance, to download and preprocess ISCIII data, use the
  same commands as Andalucia's but replace "andalucia" with "isciii":
 
 ```
from covid_simulation.preprocessing.isciii import download_df_isciii

df = download_df_isciii()
```

## Fit

This module functions are designed to fit time-series-oriented models
 with ground-truth data, then use those models to predict future occurrences.
 
 Most of these scripts require time series as inputs. We provide the
 `df_to_datetime_ser` function to ease the conversion from pandas DataFrame
 to pandas Series. For instance, if you want to work with the total number
 of admitted patients, you should do:

```
from covid_simulation.pandas_utils import df_to_datetime_ser

col_values = "admitted"
ser_admitted = df_to_datetime_ser(df, col_values)
```

**Note:** The previous command block assumes you are using a `df` preprocessed
following our criteria (see [Download and preprocessing](#download-and-preprocessing))

### Predict future days

Using `predict_and_append` one can estimate the values for the following
 days, given a time series of cumulative values. The following arguments are
 required by the function:
 
 - **ser**, a pandas Series representing one indicator (admitted patients
 , discharges, etc.). It's index are the dates, while it's values are
  cumulative. This is the kind of series outputed by [`df_to_datetime_ser`](#fit).
 - **start_date**, date to consider index 0 (*ser* dates can go further back
  than this value, though).
- **dic_model**, dictionary containing the model name and parameters, in it
's proper order. To know which models are available and which parameters
 they require, check `covid_simulation.fit.models`.
- **pred_size**, the number of days to predict.
- **fit_model**, this parameter is set on False by default, which implies
 that the model uses the parameters provided by *dic_model* to predict. If
  changed to True, then the parameters provided by *dic_model* will be used
   as the starting point, and the model will fit the data in search of better
   parameters.

The following block is an example on how to use the `predict_and_append`
function. Note that the output is the same input time series, but with
 predictions included.

```
import datetime as dt
from covid_simulation.fit.predict import predict_and_append

start_date = dt.datetime(2020, 3, 1)
dic_model = {"model_name": "gompertz",
            "params0": [10000, 1000, 20]}
pred_size = 10  # Number of predicted days

ser_admitted = predict_and_append(ser_admitted, start_date,
                                  dic_model, pred_size)
```

## Simulation

By using the time series of admitted patients (cumulative), one can simulate
 the number of hospital beds and UCI units occupied during each day.

```
from covid_simulation.simulation.pipeline import run_simulation

nruns = 200  # Number of simulations
max_beds_uci = 1200  # Maximum UCI capacity

uci_in, uci_out, uci_active, hosp_in, hosp_out, hosp_active = \
    run_simulation(ser_admitted, nruns, max_beds_uci)
```

## Credits

**Authors:**
- [David Gómez-Ullate](https://github.com/dgullate), [UCADatalab](http://datalab.uca.es/)
- [David Gordo](https://github.com/davidggphy), [ICMAT](https://www.icmat.es/)
- [Leopoldo Gutiérrez](https://github.com/leoguga), [UCADatalab](http://datalab.uca.es/)
- [Daniel Precioso](https://github.com/daniprec), [UCADatalab](http://datalab.uca.es/)
             
**Last update**: 23th april 2020

Many thanks to Fermín Mallor for guidelines and feedback.

## License

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-nd-nc/2.0/jp/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Covid Occupancy Simulation</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/UCA-Datalab" property="cc:attributionName" rel="cc:attributionURL">UCA Datalab</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/UCA-Datalab/covid_simulation" rel="dct:source">https://github.com/UCA-Datalab/covid_simulation</a>.



## Citation

```
@misc{CovidUCA2020,
  author = {Gómez-Ullate Oteiza, David and Gordo Gómez, David and Gutiérrez Galeano, Leopoldo and Precioso Garcelán, Daniel},
  title = {Covid Occupancy Simulation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/UCA-Datalab/covid_simulation}}
}
```
