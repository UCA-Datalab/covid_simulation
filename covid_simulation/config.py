import datetime as dt

#################
# distributions are parameters for each quantity
#################

PARAMS_DAILY_CASES = {'std': .45}
PARAMS_PROB_UCI = {"p_uci": .13, "p_uci_now_given_uci": .25}
PARAMS_HOSP_TO_ALTA = {'distribution': 'weibull', 'scale': 16., 'shape': 1.25}
PARAMS_HOSP_TO_UCI = {'distribution': 'weibull', 'scale': 4.5, 'shape': 1.7}
PARAMS_UCI_TO_ALTA = {'distribution': 'weibull', 'scale': 30., 'shape': 1.5}

LIST_ANDALUCIA = ['andalucia',
                  'almeria',
                  'cadiz',
                  'cordoba',
                  'granada',
                  'huelva',
                  'jaen',
                  'malaga',
                  'sevilla']

DIC_FIT_MODEL = {"model_name": "gompertz",
                 "params0": [10000, 1000, 20]
                 }

PARAMS_PREDICT = {"start_date": dt.datetime(2020, 3, 1),
                  "predicted_days": 10,
                  "dic_model": DIC_FIT_MODEL
                  }

PARAMS_FIT_P_UCI = {'p_uci_0_default': .15}

PARAMS_SIMULATION = {
    "max_beds_hosp": None,
    "nruns": 5000,
    "max_beds_uci_default": None,
}

PARAMS_PLOTS_PREDICTION = {
    "date_min": dt.datetime(2020, 3, 15),
}

param_data = {"min_cases": 0}

param_predict = {"start_date": dt.datetime(2020, 3, 1),
                 "predicted_days": 10,
                 "model": "gompertz_simple",
                 "params0": {"a": 10000,
                             "u": 1000,
                             "d": 20}
                 }

param_simul = {"path_plots": None,
               "max_beds_uci": 1200,
               "max_beds_hosp": None,
               "prob_uci_inmediate": 0.25,
               "tolerance": 100,
               "nruns": 5000,
               }

DIC_MAX_BEDS_UCI = {
    "españa": None,
    "andalucia": 1200,
    "almeria": 1200,  # 100,
    "cadiz": 1200,  # 180,
    "cordoba": 1200,  # 110,
    "granada": 1200,  # 130,
    "huelva": 1200,  # 70,
    "jaen": 1200,  # 90,
    "malaga": 1200,  # 240,
    "sevilla": 1200
}  # 280}
