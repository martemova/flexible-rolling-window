import numpy as np

def CreatDGP1(alpha, sigma, T, beta):
    """
    Generate a stationary AR(1) process with constant parameters.

    The process is given by:
        y_t = alpha + beta * y_{t-1} + sigma * eps_t,
    where eps_t ~ N(0, 1).

    This DGP serves as a baseline with no structural change.

    Parameters
    ----------
    alpha : float
        Intercept term.
    sigma : float
        Innovation standard deviation.
    T : int
        Length of the time series.
    beta : float
        AR(1) coefficient.

    Returns
    -------
    y : array
        Simulated time series of length T.
    """
    y = np.zeros(T)
    y0 = np.random.randn()
    for it in range(T):
        y[it] = alpha + beta * y0 + sigma * np.random.randn()
        y0 = y[it]
    return y


def CreatDGP2(alpha, sigma, T, iBreak):
    """
        Generate an AR(1) process with a structural break in persistence.

        The AR coefficient changes deterministically at time iBreak:
            beta_t = 0.2       for t < iBreak,
            beta_t = 0.9       for t >= iBreak.

        This DGP captures an abrupt structural change in dynamics.

        Parameters
        ----------
        alpha : float
            Intercept term.
        sigma : float
            Innovation standard deviation.
        T : int
            Length of the time series.
        iBreak : int
            Break point location.

        Returns
        -------
        y : array
            Simulated time series of length T.
        """
    y = np.zeros(T)
    y0 = np.random.randn()
    vInd = np.ones(T)
    vInd[:iBreak] = 0
    for it in range(T):
        beta_t = 0.2 + 0.7 * vInd[it]  
        y[it] = alpha + beta_t * y0 + sigma * np.random.randn()
        y0 = y[it]
    return y

def CreatDGP3(alpha, sigma, T, K):
    """
    Generate an AR(1) process with smoothly time-varying persistence.

    The AR coefficient evolves deterministically as:
        beta_t = 0.5 + 0.5 * sin(2πt / K),

    and the process follows:
        y_t = alpha + beta_t * y_{t-1} + sigma * eps_t.

    A binary state variable z_t is constructed to indicate
    high-persistence regimes.

    Parameters
    ----------
    alpha : float
        Intercept term.
    sigma : float
        Innovation standard deviation.
    T : int
        Length of the time series.
    K : int
        Period length of the sinusoidal cycle.

    Returns
    -------
    y : array
        Simulated time series of length T.
    z : array
        Binary state indicator capturing high-persistence phases.
    """
    y = np.zeros(T)
    z = np.zeros(T)
    y0 = np.random.randn()

    for it in range(T):
        beta_t = 0.5 + 0.5 * np.sin(2 * np.pi * it / K)
        y[it] = alpha + beta_t * y0 + sigma * np.random.randn()
        y0 = y[it]
        z[it] = ((0.5 + 0.5*np.sin(2 * np.pi * it / K)) >= 0.5).astype(int)
    return y, z

def CreatDGP4(alpha, sigma, T, K, p=2):
    """
    Generate an AR(1) process with exogenous regressors and
    smoothly time-varying persistence.

    The model is:
        y_t = alpha + beta_t * y_{t-1} + kappa' x_{t-1} + sigma * eps_t,

    where:
        beta_t = 0.5 + 0.5 * sin(2πt / K),
        x_t follows a persistent AR-type process,
        z_t indicates high-persistence regimes.

    This DGP is used to assess FRW performance in multivariate
    predictive regressions.

    Parameters
    ----------
    alpha : float
        Intercept term.
    sigma : float
        Innovation standard deviation.
    T : int
        Length of the time series.
    K : int
        Period length of the persistence cycle.
    p : int, default=2
        Number of exogenous regressors.

    Returns
    -------
    y : array
        Simulated dependent variable.
    x : array
        Matrix of exogenous regressors.
    z : array
        Binary state indicator.
    """
    y = np.zeros(T)
    x = np.zeros((T, p))
    y0 = np.random.randn()
    x0 = np.random.randn(p)
    z = np.zeros(T)
    
    gamma = [0.95 - 0.05*j for j in range(p)]
    kappa = [0.1 + 0.1 * j for j in range(p)]
    
    for it in range(12,T):
        beta_t = 0.5 + 0.5 * np.sin(2 * np.pi * it / K)
        x[it, :] = gamma*x0 + 0.2 * np.random.randn(p)
        y[it] = alpha + beta_t * y0 + x0@kappa + sigma * np.random.randn()
        y0 = y[it]
        x0 = x[it, :]
        z[it] = ((0.5 + 0.5 * np.sin(2 * np.pi * it / K)) >= 0.5).astype(int)
    return y, x, z