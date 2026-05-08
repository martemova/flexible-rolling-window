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


ALternative option is to replicate Tables 5 and 6 from Appendix 

author: Mariia Artemova
"""

# =========================
# Imports and paths
# =========================

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order

from frw.Empirics.fun.build_table import build_table
from frw.Empirics.fun.evaluate_forecast import compute_forecast_comparisons
from frw.Empirics.models.WML import CriterionF, OptimRho, logit


SERIES = ["INDPRO", "PAYEMS", "UNRATE", "CUMFNS"]
HORIZONS = [1, 3]


@dataclass(frozen=True)
class RunConfig:
    name: str
    macro_file: str
    nber_file: str
    fperiods: int
    table_h1: str
    table_h3: str
    include_ms: bool
    include_recession_table: bool
    sample_end: str | None = None
    exclusion_start: str | None = None
    exclusion_end: str | None = None
    save_pickle: bool = False
    save_figure: bool = False
    pickle_name: str = "ForecastEval.pkl"
    figure_series: str = "UNRATE"
    figure_p: int = 6
    figure_k_values: tuple[int, ...] = (1,)


CONFIGS = {
    "main": RunConfig(
        name="main",
        macro_file="macro_data.csv",
        nber_file="USREC.csv",
        fperiods=144,
        table_h1="Table3.tex",
        table_h3="Table4.tex",
        include_ms=True,
        include_recession_table=True,
        sample_end="2019-12-01",
        save_pickle=True,
        save_figure=True,
    ),
    "appendix": RunConfig(
        name="appendix",
        macro_file="macro_data.csv",
        nber_file="USREC.csv",
        fperiods=213,
        table_h1="Table5.tex",
        table_h3="Table6.tex",
        include_ms=False,
        include_recession_table=False,
        exclusion_start="2020-03-01",
        exclusion_end="2021-09-01",
    ),
}


def _repo_paths():
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir, base_dir / "data", base_dir / "results"


def _date_mask(index, start, end):
    if start is None or end is None:
        return pd.Series(False, index=index)
    return (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))


def run(config: RunConfig):
    _, data_dir, results_dir = _repo_paths()

    data_full = pd.read_csv(data_dir / config.macro_file, index_col="Date")
    data_full.index = pd.date_range(start="1948-01-01", periods=len(data_full), freq="MS")
    if config.sample_end is not None:
        data_full = data_full[data_full.index <= pd.Timestamp(config.sample_end)]

    nber = pd.read_csv(data_dir / config.nber_file, index_col="Date")
    nber.index = pd.date_range(start="1949-01-01", periods=len(nber), freq="MS")
    if config.sample_end is not None:
        nber = nber[nber.index <= pd.Timestamp(config.sample_end)]
    nber = nber[(nber.index.year > 1949)]
    nber_values = nber.to_numpy()
    evaluation_exclusion = _date_mask(nber.index, config.exclusion_start, config.exclusion_end)

    df_s = data_full["SAHMCURRENT"]
    state = (df_s.iloc[1:] >= 0.5) * (df_s.diff(1).iloc[1:] > 0)
    state = state[(state.index.year > 1949)]
    if config.exclusion_start is not None and config.exclusion_end is not None:
        state.loc[_date_mask(state.index, config.exclusion_start, config.exclusion_end)] = 0
    state_values = state.to_numpy().reshape(-1, 1)

    tables = {}
    res = {} if config.save_figure else None
    start_val = lambda x: [logit(x[0])]

    for s in SERIES:
        print(f"Processing series: {s}")

        n_rows = 6 if config.include_ms else 4
        m_table = np.zeros((n_rows, 4))
        m_table_expansion = np.zeros((n_rows, 4))
        m_table_recession = np.zeros((n_rows, 4)) if config.include_recession_table else None

        df = data_full[s]
        if s not in {"UNRATE", "CUMFNS"}:
            data = 100 * np.log(df).diff(12)
        else:
            data = 10 * df.diff(12)
        data = data[(data.index.year > 1949)]

        estimation_keep = ~_date_mask(data.index, config.exclusion_start, config.exclusion_end)

        mod = ar_select_order(data.iloc[:len(data) - config.fperiods], maxlag=12, ic="bic")
        p = np.max(mod.ar_lags)

        data_values = data.to_numpy()
        y = data_values[p:]
        keep_mask = estimation_keep[p:]
        t_full = np.size(y)

        window = int(0.7 * (t_full - config.fperiods))
        cv_period = t_full - window - config.fperiods
        rw_sizes = np.arange(48, window, 3)

        x = np.column_stack([data_values[p - i:-i] for i in range(1, p + 1)])
        z = state_values[p - 1:-1]

        expansion_mask = nber_values[-config.fperiods:] == 0
        evaluation_mask = (~evaluation_exclusion[-config.fperiods:])

        f_orw, f_frw, f_ml = (np.full((config.fperiods, len(HORIZONS)), np.nan) for _ in range(3))
        e_orw, e_frw, e_ml = (np.full((config.fperiods, len(HORIZONS)), np.nan) for _ in range(3))
        if config.include_ms:
            f_ms = np.full((config.fperiods, len(HORIZONS)), np.nan)
            e_ms = np.full((config.fperiods, len(HORIZONS)), np.nan)

        theta_ml = np.full((config.fperiods, len(HORIZONS), p + 1), np.nan)
        theta_frw = np.full((config.fperiods, len(HORIZONS), p + 1), np.nan)

        jj = 0
        for ind_h, h in enumerate(HORIZONS):
            print(f"  Forecast horizon h = {h}")

            rho0 = OptimRho(
                y[:window + cv_period],
                x[:window + cv_period],
                z[:window + cv_period],
                window,
                cv_period,
                start_val([0.6]),
                "binary",
                h,
                keep_mask=keep_mask[:window + cv_period],
            )

            cv_loss_rw = np.zeros(len(rw_sizes))
            for kk, rw_size in enumerate(rw_sizes):
                cv_loss_rw[kk] = CriterionF(
                    rw_size,
                    y[:window + cv_period],
                    x[:window + cv_period],
                    z[:window + cv_period],
                    window,
                    cv_period,
                    "RW",
                    horizon=h,
                    keep_mask=keep_mask[:window + cv_period],
                )[0]

            orw = rw_sizes[np.argmin(cv_loss_rw)]

            for l in range(config.fperiods):
                start = cv_period + l
                stop = window + start + 1
                v_roll_y = y[start:stop]
                v_roll_x = x[start:stop]
                v_roll_z = z[start:stop]
                v_roll_keep = keep_mask[start:stop]

                _, e_ml[l, ind_h], f_ml[l, ind_h], theta_hat_ml = CriterionF(
                    None, v_roll_y, v_roll_x, v_roll_z, window, 1, "ML", horizon=h, keep_mask=v_roll_keep
                )
                _, e_orw[l, ind_h], f_orw[l, ind_h], _ = CriterionF(
                    orw, v_roll_y, v_roll_x, None, window, 1, "RW", horizon=h, keep_mask=v_roll_keep
                )
                _, e_frw[l, ind_h], f_frw[l, ind_h], theta_hat_frw = CriterionF(
                    rho0, v_roll_y, v_roll_x, v_roll_z, window, 1, "binary", horizon=h, keep_mask=v_roll_keep
                )

                theta_ml[l, ind_h, :] = theta_hat_ml[:, 0]
                theta_frw[l, ind_h, :] = theta_hat_frw[:, 0]

                if config.include_ms:
                    from frw.Empirics.models.MarkovSwitching import fit_markov_switching

                    y_lag = v_roll_y[-p - h:-h][::-1].copy()
                    y_hat_ms = fit_markov_switching(v_roll_y, window, y_lag, p=p, h=h)
                    e_ms[l, ind_h] = v_roll_y[-1] - y_hat_ms
                    f_ms[l, ind_h] = y_hat_ms

            sections = [
                ("full", np.ones(config.fperiods, dtype=bool) & evaluation_mask),
                ("expansion", expansion_mask[:, 0] & evaluation_mask),
            ]
            if config.include_recession_table:
                sections.append(("recession", (~expansion_mask[:, 0]) & evaluation_mask))

            for name, current_mask in sections:
                kwargs = {}
                if config.include_ms:
                    kwargs["X_hat_MS"] = f_ms[:, ind_h]
                    kwargs["vE_MS"] = e_ms[:, ind_h]

                results = compute_forecast_comparisons(
                    y,
                    f_frw[:, ind_h],
                    f_ml[:, ind_h],
                    f_orw[:, ind_h],
                    e_frw[:, ind_h],
                    e_ml[:, ind_h],
                    e_orw[:, ind_h],
                    current_mask,
                    rho0,
                    orw,
                    config.fperiods,
                    h,
                    **kwargs,
                )

                if name == "full":
                    m_table[:, jj:jj + 2] = results
                elif name == "expansion":
                    m_table_expansion[:, jj:jj + 2] = results
                else:
                    m_table_recession[:, jj:jj + 2] = results

            jj += 2

        tables[s] = {
            "res": m_table,
            "res_expansion": m_table_expansion,
        }
        if config.include_recession_table:
            tables[s]["res_recess"] = m_table_recession

        if res is not None:
            res[f"{s}_AR{p}"] = {
                "theta_FRW": theta_frw,
                "theta_ML": theta_ml,
                "p": p,
                "horizons": HORIZONS,
            }

    latex_h1 = build_table(tables, SERIES, 1, horizon_label="n=1")
    latex_h3 = build_table(tables, SERIES, 3, horizon_label="n=3")

    with open(results_dir / "tables" / config.table_h1, "w") as f:
        f.write(latex_h1)
    with open(results_dir / "tables" / config.table_h3, "w") as f:
        f.write(latex_h3)

    if config.save_pickle:
        with open(results_dir / config.pickle_name, "wb") as f:
            pickle.dump(tables, f)

    if config.save_figure and res is not None:
        from frw.Empirics.fun.create_figure import figure

        for k in config.figure_k_values:
            figure(
                res,
                np.asarray(state)[-config.fperiods:],
                nber[-config.fperiods:].index,
                s=config.figure_series,
                p=config.figure_p,
                k=k,
            )

    return tables


# run_setup = "main" 
run_setup = "appendix"

parser = argparse.ArgumentParser(description="Run one of the empirical forecasting exercises.")
parser.add_argument(
    "--variant",
    choices=sorted(CONFIGS),
    default=run_setup,
    help="Which existing script behavior to run.",
)
args = parser.parse_args()
run(CONFIGS[args.variant])
