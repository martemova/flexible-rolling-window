import numpy as np
import statsmodels.api as sm


def inv_logit(x):
    """
            Transform an unconstrained scalar into a value in (0.5, 1).

            This function is used to map an unconstrained optimization parameter
            into a valid range for the binary-weight hyperparameter `rho`.

            Parameters
            ----------
            x : float
                Unconstrained parameter.

            Returns
            -------
            float
                Value in the interval (0.5, 1).
            """
    return 1- 0.5 * np.exp(x) / (1 + np.exp(x))  


def DieboldMarianoTest(vE1, vE2, K):
    """
        Perform the Diebold–Mariano test for equal predictive accuracy.

        The test is implemented using squared forecast errors and
        heteroskedasticity- and autocorrelation-consistent (HAC)
        standard errors with lag truncation parameter `h`.

        Parameters
        ----------
        vE1 : array
            Forecast errors from model 1.
        vE2 : array
            Forecast errors from model 2.
        K : int
            HAC lag length

        Returns
        -------
        float
            p-value of the Diebold–Mariano test.
        """
    d = vE1 ** 2 - vE2 ** 2
    X = np.ones(len(d))
    model = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags': K})

    return model.pvalues[0]


def compute_forecast_comparisons(y, X_hat_FRW, X_hat_ML, X_hat_ORW, X_hat_MS,
                                 vE_FRW, vE_ML, vE_ORW, vE_MS, mask, rho, ORW, fperiods, h):
    """
    Compute forecast comparisons across competing models and methods.

    This function evaluates relative forecast performance of the
    Flexible Rolling-Window (FRW) model against several benchmarks:
    Maximum Likelihood (ML), Optimal Rolling Window (ORW), and a
    Markov-switching model.

    Evaluation is performed using:
        - RMSE ratios
        - Diebold–Mariano tests 

    The evaluation can be restricted to specific subsamples
    (e.g. expansions or recessions) via a Boolean mask.

    Parameters
    ----------
    y : array
        Observed target series.
    X_hat_FRW : array
        Forecasts from the FRW model.
    X_hat_ML : array
        Forecasts from the ML benchmark.
    X_hat_ORW : array
        Forecasts from the ORW benchmark.
    X_hat_MS : array
        Forecasts from the Markov-switching benchmark.
    vE_FRW : array
        Forecast errors from FRW.
    vE_ML : array
        Forecast errors from ML.
    vE_ORW : array
        Forecast errors from ORW.
    vE_MS : array
        Forecast errors from the Markov-switching model.
    mask : array (bool)
        Boolean mask indicating which observations are used
        in the evaluation.
    rho : array
        Optimized (untransformed) FRW hyperparameter.
    ORW : int
        Optimal rolling-window length.
    fperiods : int
        Length of the out-of-sample evaluation period.
    h : int
        Forecast horizon.

    Returns
    -------
    np.ndarray
        A (6 × 2) array containing:
            - Rows 0,2,4: RMSE ratios (FRW vs ML, ORW, MS)
            - Rows 1,3,5: corresponding DM test p-values
            - Column 1: selected parameter values (rho, ORW)
    """
    par_optim = lambda x: [inv_logit(x[0])]
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    # Apply mask
    if sum(mask) == fperiods or sum(mask) > 50:
        H = h
    else:
        H = h - 1
    y_ = y[-fperiods:][mask]
    f_FRW = X_hat_FRW[mask]
    f_ML = X_hat_ML[mask]
    f_ORW = X_hat_ORW[mask]
    f_MS2_sv = X_hat_MS[mask]

    v_FRW = vE_FRW[mask]
    v_ML = vE_ML[mask]
    v_ORW = vE_ORW[mask]
    v_MS2_sv = vE_MS[mask]

    out = np.full((6, 2), np.nan)

    # --- RMSE Ratios ---
    out[0, 0] = rmse(y_, f_FRW) / rmse(y_, f_ML)
    out[2, 0] = rmse(y_, f_FRW) / rmse(y_, f_ORW)
    out[4, 0] = rmse(y_, f_FRW) / rmse(y_, f_MS2_sv)

    # --- DM Tests ---
    out[1, 0] = DieboldMarianoTest(v_FRW, v_ML, H)
    out[3, 0] = DieboldMarianoTest(v_FRW, v_ORW, H)
    out[5, 0] = DieboldMarianoTest(v_FRW, v_MS2_sv, H)
    
    # --- Parameters ---
    out[0, 1] = par_optim(rho)[0]  # optimized rho
    out[2, 1] = ORW  # other param (e.g. ORW weight)

    return out
