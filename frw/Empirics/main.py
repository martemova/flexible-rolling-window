"""
Empirical replication of Tables 3–4 and Figure 4 in
“On the Use of Flexible Rolling-Window Estimation for Macroeconomic Forecasting”.

This script implements the empirical forecasting exercise using U.S.
macroeconomic time series. It compares the Flexible Rolling-Window (FRW)
estimator against several benchmark models:
    - Maximum Likelihood / equal weights (ML)
    - Ordinary Rolling Window (ORW)
    - Markov-Switching AR model (MS)

The script:
    1. Loads and preprocesses macroeconomic data.
    2. Constructs growth rates / differences.
    3. Selects AR lag length using BIC.
    4. Optimizes FRW hyperparameters via cross-validation.
    5. Generates recursive out-of-sample forecasts.
    6. Evaluates forecast performance over:
        - full sample
        - expansion periods
        - recession periods
    7. Produces LaTeX tables and figures replicating the paper.

Dependencies
------------
- Python: numpy, pandas, statsmodels, scipy, pickle
- R (via rpy2): MSwM package (required for Markov-switching benchmark)

Outputs
-------
- results/ForecastEval.pkl
- results/tables/Table3.tex
- results/tables/Table4.tex
- results/Figures/alpha_FRW.pdf

author: Mariia Artemova
"""

# =========================
# Imports and paths
# =========================
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
import pickle
from pathlib import Path

from frw.Empirics.models.MarkovSwitching import fit_markov_switching
from frw.Empirics.models.WML import OptimRho, CriterionF
from frw.Empirics.fun.build_table import build_table
from frw.Empirics.fun.create_figure import figure
from frw.Empirics.fun.evaluate_forecast import compute_forecast_comparisons
from frw.Empirics.models.WML import logit, inv_logit

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# =========================
# Load data
# =========================

# Macroeconomic series (monthly)
data_full = pd.read_csv(DATA_DIR / "macro_data.csv", index_col="Date")  
data_full.index = pd.date_range(start='1948-01-01', periods=len(data_full), freq='MS')

# Series used in the empirical exercise
series = ["UNRATE"]#["INDPRO", "PAYEMS", "UNRATE", "CUMFNS"] 

# NBER recession indicator, used only for plots
NBER = pd.read_csv(DATA_DIR / 'USREC.csv', index_col='Date')
NBER.index = pd.date_range(start='1949-01-01', periods=len(NBER), freq='MS')
NBER = NBER[(NBER.index.year > 1949)]

# =========================
# State variable (Sahm rule)
# =========================

# Sahm indicator used to define the binary state variable Z_t
df_s = data_full['SAHMCURRENT']

# State variable:
# Z_t = 1 if Sahm indicator >= 0.5 AND increasing
state = (df_s.iloc[1:] >= 0.5) * (df_s.diff(1).iloc[1:] > 0) 
state = state[(state.index.year > 1949)]

# =========================
# Forecasting setup
# =========================
horizons = [1, 3]       # forecast horizons
fperiods = 144          # out-of-sample evaluation length (12 years)

# Containers for results
res = {}                # coefficient paths for plotting
tables = {}             # forecast evaluation tables

# Weighting scheme for FRW


# Transformations for optimization
start_val = lambda x: [logit(x[0])]
par_optim = lambda x: [inv_logit(x[0])]


# =========================
# Main loop over series
# =========================
for s in series: 
    print(f"Processing series: {s}")

    # Tables: rows = models/statistics, columns = horizons
    mTable = np.zeros((6, 4))   # full sample results
    mTable2 = np.zeros((6, 4))  # recession results
    mTable3 = np.zeros((6, 4))  # expansion results

    # -------------------------
    # Data transformation
    # -------------------------
    df = data_full[s] 

    # Annual growth rates or differences
    if s != 'UNRATE' and s != 'CUMFNS':
        data = 100 * np.log(df).diff(12)
    else:
        data = 10 * (df).diff(12)

    data = data[(data.index.year > 1949)]  # start sample in 1950

    # -------------------------
    # Lag selection
    # -------------------------
    mod = ar_select_order(data.iloc[:len(data) - fperiods], maxlag=12, ic='bic')  
    p = np.max(mod.ar_lags)

    # -------------------------
    # Sample construction
    # -------------------------
    y = np.array(data)[p:]
    T_full = np.size(y)
    
    window = int(0.7 * (T_full - fperiods))  # estimation window 
    CV_period = T_full - window - fperiods   # cross-validation window
    rw_sizes = np.arange(48, window, 3)      # ORW grid 

    # Construct lag matrix
    N = len(y)
    x = np.zeros((N, p))
    for i in range(1, p + 1):
        x[:, i - 1] = data[p - i: - i]  

    # Lagged state variable
    z = np.array(state)[p - 1:-1, None]

    # Expansion/recession mask for OOS period
    mask = np.array(NBER[-fperiods:]) == 0 #expansion periods

    # Storage for forecasts and errors
    f_ORW, f_FRW, f_ML, f_MS = (np.full((fperiods, len(horizons)), np.nan) for _ in range(4))
    e_ORW, e_FRW, e_ML, e_MS = (np.full((fperiods, len(horizons)), np.nan) for _ in range(4))

    # Storage for coefficient paths
    theta_ML = np.full((fperiods, len(horizons), p + 1), np.nan)
    theta_FRW = np.full((fperiods, len(horizons), p + 1), np.nan)

    # =========================
    # Loop over horizons
    # =========================
    jj = 0
    for ind_H, h in enumerate(horizons):
        print(f"  Forecast horizon h = {h}")

        # -------------------------
        # Determine FRW parameter (Algorithm 1)
        # -------------------------
        rho0 = OptimRho(
            y[:window + CV_period],
            x[:window + CV_period],
            z[:window + CV_period],
            window,
            CV_period,
            start_val([0.6]),
            'binary',
            h
        )

        # -------------------------
        # ORW window selection
        # -------------------------
        cv_loss_rw = np.zeros(len(rw_sizes))
        for kk, RW_size in enumerate(rw_sizes):
            cv_loss_rw[kk] = CriterionF(
                RW_size,
                y[:window + CV_period],
                x[:window + CV_period],
                z[:window + CV_period],
                window,
                CV_period,
                "RW",
                horizon=h
            )[0]

        ORW = rw_sizes[np.argmin(cv_loss_rw)]

        # -------------------------
        # Recursive forecasting
        # -------------------------
        for l in range(fperiods):

            vRollY = y[CV_period + l:]
            vRollX = x[CV_period + l:]
            vRollZ = z[CV_period + l:]

            if l != fperiods - 1:
                vRollY = vRollY[:-fperiods + l + 1]
                vRollX = vRollX[:-fperiods + l + 1]
                vRollZ = vRollZ[:-fperiods + l + 1]

            # ML
            _, e_ML[l, ind_H], f_ML[l, ind_H], theta_hat_ML = CriterionF(
                None, vRollY, vRollX, vRollZ, window, 1, "ML", horizon=h
            )

            # ORW
            _, e_ORW[l, ind_H], f_ORW[l, ind_H], _ = CriterionF(
                ORW, vRollY, vRollX, None, window, 1, "RW", horizon=h
            )

            # FRW
            _, e_FRW[l, ind_H], f_FRW[l, ind_H], theta_hat_FRW = CriterionF(
                rho0, vRollY, vRollX, vRollZ, window, 1, 'binary', horizon=h
            )

            # Store *all* coefficients
            theta_ML[l, ind_H, :] = theta_hat_ML[:, 0]
            theta_FRW[l, ind_H, :] = theta_hat_FRW[:, 0]

            # Markov-switching benchmark (R-based)
            y_lag = vRollY[-p - h:-h][::-1].copy()
            y_hat_ms = fit_markov_switching(vRollY, window, y_lag, p=p, h=h)
            e_MS[l, ind_H] = vRollY[-1] - y_hat_ms
            f_MS[l, ind_H] = y_hat_ms

        # -------------------------
        # Forecast evaluation
        # -------------------------
        for name, current_mask in zip(
                ["full", "expansion", "recession"],
                [np.ones(fperiods, bool), mask[:, 0], ~mask[:, 0]]
        ):
            results = compute_forecast_comparisons(
                y, f_FRW[:, ind_H], f_ML[:, ind_H], f_ORW[:, ind_H], f_MS[:, ind_H],
                e_FRW[:, ind_H], e_ML[:, ind_H], e_ORW[:, ind_H], e_MS[:, ind_H],
                current_mask, rho0, ORW, fperiods, h
            )

            if name == "full":
                mTable[:, jj:jj + 2] = results
            elif name == "recession":
                mTable2[:, jj:jj + 2] = results
            else:
                mTable3[:, jj:jj + 2] = results

        jj += 2

        res[f"{s}_AR{p}"] = {
            "theta_FRW": theta_FRW,  # (fperiods, horizons, p+1)
            "theta_ML": theta_ML,  # (fperiods, horizons, p+1)
            "p": p,
            "horizons": horizons
        }

        # Store tables
        tables[s] = {
            "res": mTable,
            "res_recess": mTable2,
            "res_expansion": mTable3,
        }

# =========================
# Save and output results
# =========================

with open(RESULTS_DIR / "ForecastEval.pkl", "wb") as f:
    pickle.dump(tables, f)

for k in range(7):
    figure(res, np.asarray(state)[-fperiods:], NBER[-fperiods:].index, s="UNRATE", p=6, k=k)

latex_h1 = build_table(tables, series, 1, horizon_label="n=1")
latex_h3 = build_table(tables, series, 3, horizon_label="n=3")

with open(RESULTS_DIR / "tables" / "Table3.tex", "w") as f:
    f.write(latex_h1)

with open(RESULTS_DIR / "tables" / "Table4.tex", "w") as f:
    f.write(latex_h3)