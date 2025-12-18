import numpy as np
import scipy.optimize as opt

def logit(x):
    """
        Transform a scalar from the interval (0.5, 1) to the real line.

        This transformation is used to map the bounded hyperparameter of the
        binary-weighting scheme into an unconstrained space for numerical
        optimization.

        Parameters
        ----------
        x : float
            Scalar value in the open interval (0.5, 1).

        Returns
        -------
        float
            Unconstrained real-valued transformation of `x`.
        """
    if not (0.5 < x < 1):
        raise ValueError("x must be in (0.5, 1)")
    return np.log((x - 1) / (0.5 - x))

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


def OptimRho(y, x, z, Tprime, T, rho_ini, ML, h):
    """
    Optimize the hyperparameter `rho` for a given weighting scheme.

    The objective minimized is the out-of-sample forecast error computed
    by `CriterionF`.

    Parameters
    ----------
    y : array (T, )
        Time series of interest.
    x : array (T x p)
        Matrix of lagged regressors.
    z : array (T x 1)
        State variable used for constructing binary similarity weights.
    Tprime : int
        Size of the estimation window.
    T : int
        Size of the cross-validation window.
    rho_ini : list 
        Starting value for the hyperparameters.
    ML : string, options= {"ML", "RW", "binary"} 
        Type of weighting scheme.
    h : int
        Forecast horizon.

    Returns
    -------
    array
        Optimal hyperparameter values (untransformed).
    """
    fun_optim = lambda XX: CriterionF(XX, y, x, z, Tprime, T, ML, horizon=h)[0]
    res = opt.minimize(fun_optim, rho_ini, method='BFGS')

    return res.x


def MLE_AR(y, X, w):
    """
        Compute weighted OLS estimates for an AR(p) model.

        Parameters
        ----------
        y : array
            Dependent variable (T, ).
        X : array
            Regressor matrix (T x (p+1)) including a constant.
        w : array
            Weight vector of length T.

        Returns
        -------
        array
            Estimated coefficient vector (p+1 x 1).
        """
    W = np.diag(w[:, 0])
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y.reshape(-1, 1))
    return beta


def make_weights(z, Tprime, T_CV, ML, rho=None):
    """
        Construct weights for FRW/ORW/Binary schemes.

        Parameters
        ----------
        z : array
            State variable used in binary similarity weighting.
        Tprime : int
            Estimation window size.
        T_CV : int
            Cross-validation window size.
        ML : string, options={"ML", "RW", "binary"}
            Weighting scheme.
        rho : float, optional
            Hyperparameter controlling the weighting scheme.
            - ML: ignored
            - RW: cutoff length
            - binary: similarity weighting strength

        Returns
        -------
        W : array (T_CV x Tprime)
            Weight matrix for each CV window.
        """
    if ML == 'ML':
        return np.ones((T_CV, Tprime))
    elif ML == 'binary':
        I = 1 * (z[:, 0] == 0.0)
        Z = np.lib.stride_tricks.sliding_window_view(I, window_shape=Tprime)[:T_CV]
        Z_target = I[Tprime:Tprime + T_CV]  
        diff2 = (Z - Z_target[:, None]) ** 2  
        return (rho * (1 - diff2) + (1 - rho) * diff2)
    elif ML == 'RW':
        cut = int(rho)
        cols = np.arange(Tprime) 
        end = Tprime - (T_CV - 1) + np.arange(T_CV)  
        W = 1 * ((cols < end[:, None]) & (cols >= (end - cut)[:, None]))
        return W
    else:
        raise NotImplementedError(f"Unknown weighting scheme: {ML}")


def CriterionF(rho0, y, x, z, Tprime, T_CV, ML, horizon=1):  # TODO
    """
        Cross-validation criterion for tuning weighting parameters.

        Computes the out-of-sample prediction error for a given hyperparameter
        value rho.

        Parameters
        ----------
        rho0 : array
            Unconstrained hyperparameter (transformed inside if needed).
        y : array (T, )
            Dependent variable .
        x : array (T, p)
            Regressor matrix (lags).
        z : array (T, 1)
            Binary state series.
        Tprime : int
            Length of estimation window.
        T_CV : int
            Length of cross-validation window.
        ML : string, options={"ML", "RW", "binary"}
            Weighting scheme.
        horizon : int, default=1
            Forecast horizon.

        Returns
        -------
        list
            [Q, vE, X_hat, theta] where:
            - Q : RMSE of CV errors
            - vE : prediction errors for each point
            - X_hat : forecasts
            - theta : final estimated coefficient vector
        """
    
    if ML == 'binary':
        rho = inv_logit(rho0[0])
    elif ML == 'RW':
        rho = rho0
    elif ML == 'ML':
        rho = 1

    Tprime_h = Tprime - (horizon - 1)           # finish sample earlier if horizon>1
    vE, X_hat = np.zeros(T_CV), np.zeros(T_CV)  # store errors and predictions over validation sample
    w_mat = make_weights(z, Tprime_h, T_CV, ML, rho=rho)

    for t in range(T_CV):
        w = w_mat[t, :].reshape(-1, 1)

        p = np.shape(x)[1]  # lags in AR(p)
        X_new = [1]         # add constant 
        X0 = np.zeros((Tprime_h, p + 1))
        X0[:, 0] = np.ones(Tprime_h)
        for j in range(p):
            X0[:, j + 1] = x[t:Tprime_h + t, j]  # regressors in sample
            X_new += [x[Tprime_h + t, j]]        # regressors out of sample

        X_new = np.array(X_new).reshape(p + 1, 1)

        theta = MLE_AR(y[t:Tprime_h + t], X0, w)  # Weighted OLS

        # make prediction
        for _ in range(horizon):
            X_new_h = sum(theta * X_new)
            if horizon > 1:
                X_new[2:] = X_new[1:-1].copy()
                X_new[1] = X_new_h.copy()
        X_hat[t] = X_new_h.item()
        vE[t] = y[Tprime + t] - X_hat[t]  # prediction error

    Q = np.sqrt(np.mean(vE ** 2))

    if T_CV > 1:
        return [Q, vE, X_hat, theta]
    else:
        return [Q, vE[0], X_hat[0], theta]