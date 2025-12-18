import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import warnings
import pandas as pd

try:
    ro.r('library(MSwM)')
except:
    warnings.warn(
        "The R package 'MSwM' is not installed. Please install it in R before proceeding.",
        UserWarning
    )
    


def fit_markov_switching(vRollY, window, y_lag, p=1, h=1, switch_variance=True):
    """
        Fit a two-regime Markov-Switching autoregressive model in R using the
        `MSwM` package and produce an h-step-ahead forecast.

        This function serves as a Python wrapper around the R function `msmFit()`
        (from the `MSwM` package) through the `rpy2` bridge. It constructs an
        AR(p) regression model in R, estimates the Markov-switching parameters,
        extracts filtered regime probabilities, and then computes an h-step
        ahead forecast using the regime-dependent coefficients and the
        filtered probability vector through the transition matrix.

        Parameters
        ----------
        vRollY : array
            Time series used to construct the rolling-window regression sample.
            Must contain at least `window` observations.

        window : int
            Size of the estimation window. Only observations within this window
            are used to construct the R dataframe for AR(p) estimation.

        y_lag : array
            Vector of lagged values of `y` used as the initial state for multi-step
            forecasting. Must have p columns

        p : int, default=1
            Autoregressive order of the AR(p) process.

        h : int, default=1
            Forecast horizon. The function performs iterative forecasting:
            forecasts for horizons > 1 use previously predicted values.

        switch_variance : bool, default=True
            Determines whether the conditional variance is allowed to switch
            across regimes:
            - True: both mean and variance parameters are regime-specific.
            - False: only the mean equation switches; variance is constant.

        Returns
        -------
        y_hat : float
            The h-step-ahead forecast based on the Markov-switching model.

        """
    cols = {}
    cols['y'] = vRollY[p:window - (h - 1)].copy()
    for j in range(1, p + 1):
        cols[f'x{j}'] = vRollY[p - j:window - (h - 1) - j]
    df_MS = pd.DataFrame(cols)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['r_df'] = ro.conversion.py2rpy(df_MS)
        ro.globalenv['p'] = int(p)
        ro.globalenv["switch_variance"] = switch_variance

    # Load MSwM package and fit AR(1) with 2 regimes
    ro.r('''
            library(MSwM)

            # Construct AR(p) formula dynamically
            rhs_vars <- paste0("x", 1:p, collapse = " + ")
            fmla <- as.formula(paste("y ~", rhs_vars))

            # Fit AR(p)
            ar_model <- lm(fmla, data = na.omit(r_df))

            # sw vector: TRUE for intercept + p lag coefficients, FALSE for sigma (if variance doesn't switch)
            sw_vec <- c(TRUE, rep(FALSE, p), switch_variance)

            control <- list(maxiter = 150000, maxiterInner=50, maxiterOuter=30, tol = 1e-5, parallel = TRUE, trace = FALSE)
            ms_model <- suppressWarnings(msmFit(ar_model, k = 2, sw = sw_vec, control=control))
            filtered_probs <- ms_model@Fit@filtProb
            P <- ms_model@transMat
            params <- ms_model@Coef
            ''')


    # Get results back to Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        filtered_probs = ro.conversion.rpy2py(ro.r('filtered_probs'))
        params = ro.conversion.rpy2py(ro.r('params'))

    mu = np.array(params['(Intercept)'])
    phi = params.filter(like='x').to_numpy()

    with localconverter(ro.default_converter + pandas2ri.converter):
        P = ro.conversion.rpy2py(ro.r('P'))
    xi_hat = P @ filtered_probs[-1, :]

    y_lag_t = y_lag.copy()
    for hh in range(h):
        y_hat_regime = np.array([
            mu[i] + np.dot(phi[i], y_lag_t) for i in range(len(mu))
        ])
        y_hat = np.dot(xi_hat, y_hat_regime)
        y_lag_t = np.insert(y_lag_t[:-1], 0, y_hat)
        xi_hat = P @ xi_hat

    return y_hat