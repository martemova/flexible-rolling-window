import numpy as np


def weighted_ar1(y, w):
    """
    Closed-form weighted least squares estimator for an AR(1) model.

    The model is:
        y_t = alpha + beta * y_{t-1} + eps_t,

    where observations are weighted according to a user-specified weight
    vector.

    Parameters
    ----------
    y : array
        Time series of length T.
    w : array
        Weight vector of length T, where larger values imply greater
        importance of the corresponding observations.

    Returns
    -------
    beta_hat : array
        Estimated coefficient vector [alpha, beta].
    sigma_hat : float
        Estimated standard deviation of the residuals.
    """
    T = len(y)
    X = np.column_stack([np.ones(T - 1), y[:-1]])  
    Y = y[1:]  
    w = w[1:]  

    # Apply weights
    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    Yw = Y * Wsqrt

    # WLS solution
    beta_hat = np.linalg.lstsq(Xw, Yw, rcond=None)[0]

    # Residual variance
    resid = Y - X @ beta_hat
    sigma2 = np.sum(w * (resid ** 2)) / np.sum(w)

    return beta_hat, np.sqrt(sigma2)


# ------------------ Loss function (RMSE/MAE/etc) for given gamma ------------------ #
def RMSEFRW(dGamma0, y, iRW, iCVbegin, iT, iH=1):
    """
    Compute cross-validated forecasting loss for FRW with exponential weights.

    Parameters
    ----------
    dGamma0 : float
        Untransformed optimization parameter for gamma.
    y : array
        Simulated time series.
    iRW : int
        Rolling-window length.
    iCVbegin : int
        Starting index of the cross-validation period.
    iT : int
        Length of the in-sample estimation period.
    iH : int, default=1
        Forecast horizon.

    Returns
    -------
    float
        Cross-validated forecasting loss.
    """
    dGamma = np.exp(dGamma0) / (1 + np.exp(dGamma0))

    exponents = iRW - np.arange(1, iRW + 1)
    vW = dGamma ** exponents

    verror = []
    y_begin = iCVbegin - iRW
    
    for it in range(y_begin, iT - iRW - iH + 1):
        start = it + iH - 1
        end = it + iRW + iH - 1
        g_vRollY = y[start:end]  

        Y = g_vRollY

        # compute weighted estimates
        try:
            beta_hat, sigma_hat = weighted_ar1(Y, vW)
        except Exception:
            # fallback: simple OLS
            beta_hat, sigma_hat = weighted_ar1(Y, np.ones_like(vW))

        # 1-step ahead forecast: use last available lag value (the last element of lagged vector)
        dyhat = beta_hat[0] + beta_hat[1] * y[it + iRW - 1]

        # actual y at forecast point
        actual_index = it + iRW + iH - 1
        if actual_index >= len(y):
            continue
        actual = y[actual_index]

        verror.append(actual - dyhat)

    verror = np.asarray(verror, dtype=float)
    # if no forecasts computed, return large penalty
    if verror.size == 0:
        return 1e6
   
    return np.sqrt(np.mean(verror ** 2))


def RMSEFRW_binary(dGamma0, y, z, iRW, iCVbegin, iT, iH=1):
    """
    Compute cross-validated forecasting loss for FRW with binary similarity weights.

    Parameters
    ----------
    dGamma0 : float
        Untransformed parameter for gamma.
    y : array
        Simulated time series.
    z : array
        Binary state indicator.
    iRW : int
        Rolling-window length.
    iCVbegin : int
        Starting index of the cross-validation period.
    iT : int
        Length of the in-sample estimation period.
    iH : int, default=1
        Forecast horizon.

    Returns
    -------
    float
        Cross-validated forecasting loss.
    """
    dGamma = 1 - 0.5*np.exp(dGamma0) / (1 + np.exp(dGamma0))

    verror = []
    y_begin = iCVbegin - iRW
    # iterate rolling windows
    for it in range(y_begin, iT - iRW - iH + 1):
        start = it + iH - 1
        end = it + iRW + iH - 1
        g_vRollY = y[start:end]  # length RW 
        g_vRollZ = z[start - 1:end - 1]

        actual_index = it + iRW + iH - 1

        Z_target = z[it + iRW - 1]
        diff2 = (g_vRollZ - Z_target) ** 2
        vW = dGamma * (1 - diff2) + (1 - dGamma) * diff2

        Y = g_vRollY

        # compute weighted estimates
        try:
            beta_hat, sigma_hat = weighted_ar1(Y, vW)
        except Exception:
            # fallback: simple OLS
            beta_hat, sigma_hat = weighted_ar1(Y, np.ones_like(vW))

        # 1-step ahead forecast: use last available lag value (the last element of lagged vector)
        dyhat = beta_hat[0] + beta_hat[1] * y[it + iRW - 1]

        # actual y at forecast point
        if actual_index >= len(y):
            continue
        actual = y[actual_index]

        verror.append(actual - dyhat)

    verror = np.asarray(verror, dtype=float)
    # if no forecasts computed, return large penalty
    if verror.size == 0:
        return 1e6

    return np.sqrt(np.mean(verror ** 2))


def weighted_ar1_X(y, x, w):
    """
        Closed-form weighted least squares estimator for an AR(1) model
        with exogenous regressors.

        Parameters
        ----------
        y : array
            Dependent variable.
        x : array
            Matrix of exogenous regressors.
        w : array
            Weight vector.

        Returns
        -------
        beta_hat : array
            Estimated coefficient vector.
        sigma_hat : float
            Estimated standard deviation of the residuals.
        """
    T = len(y)
    X = np.column_stack([np.ones(T - 1), y[:-1], x[:-1, :]])  
    Y = y[1:]  
    w = w[1:]  

    # Apply weights
    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    Yw = Y * Wsqrt

    # WLS solution
    beta_hat = np.linalg.lstsq(Xw, Yw, rcond=None)[0]

    # Residual variance
    resid = Y - X @ beta_hat
    sigma2 = np.sum(w * (resid ** 2)) / np.sum(w)

    return beta_hat, np.sqrt(sigma2)


def RMSEFRW_X(dGamma0, y, x, z, iRW, iCVbegin, iT, iH=1):
    """
        Compute cross-validated forecasting loss for FRW with binary weights
        and exogenous regressors.

        This function extends RMSEFRW_binary to a setting where the predictive
        regression includes additional covariates.

        Parameters
        ----------
        dGamma0 : float
            Untransformed optimization parameter for gamma.
        y : array
            Simulated time series.
        x : array
            Matrix of exogenous regressors.
        z : array
            Binary state indicator.
        iRW : int
            Rolling-window length.
        iCVbegin : int
            Starting index of the cross-validation period.
        iT : int
            Length of the in-sample estimation period.
        iH : int, default=1
            Forecast horizon.

        Returns
        -------
        float
            Cross-validated forecasting loss.
        """
    dGamma = 1 - 0.5*np.exp(dGamma0) / (1 + np.exp(dGamma0))

    verror = []
    y_begin = iCVbegin - iRW
    # iterate rolling windows
    for it in range(y_begin, iT - iRW - iH + 1):
        start = it + iH - 1
        end = it + iRW + iH - 1
        g_vRollY = y[start:end]     # length RW 
        g_vRollX = x[start:end, :]  # length RW 
        g_vRollZ = z[start - 1:end - 1]

        actual_index = it + iRW + iH - 1

        Z_target = z[it + iRW - 1]
        diff2 = (g_vRollZ - Z_target) ** 2
        vW = dGamma * (1 - diff2) + (1 - dGamma) * diff2

        Y = g_vRollY
        X = g_vRollX

        # compute weighted estimates
        try:
            beta_hat, sigma_hat = weighted_ar1_X(Y, X, vW)
        except Exception:
            # fallback: simple OLS
            beta_hat, sigma_hat = weighted_ar1_X(Y, X, np.ones_like(vW))

        # 1-step ahead forecast: use last available lag value (the last element of lagged vector)
        dyhat = beta_hat[0] + beta_hat[1] * y[it + iRW - 1] + beta_hat[2:] * x[it + iRW - 1, :]

        # actual y at forecast point
        if actual_index >= len(y):
            continue
        actual = y[actual_index]

        verror.append(actual - dyhat)

    verror = np.asarray(verror, dtype=float)
    # if no forecasts computed, return large penalty
    if verror.size == 0:
        return 1e6
    
    return np.sqrt(np.mean(verror ** 2))
