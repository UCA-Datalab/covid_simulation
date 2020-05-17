from lifelines import WeibullFitter


def fit_weibull(df, x_grid=None):
    # Initialize the model and fit our data
    wbf = WeibullFitter()
    wbf.fit(df["offset"], df["observed"])
    
    # Get weibull parameters
    params = {"scale": wbf.lambda_,
             "shape": wbf.rho_}
    
    # If x_grid is provided, return y
    if x_grid is not None:
        pdf = wbf.density_at_times(x_grid).to_numpy()
        return params, pdf
    else:
        return params
