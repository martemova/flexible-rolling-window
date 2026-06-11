# FRW

This repository provides the code and replication materials for the paper  
*“On the Use of Flexible Rolling-Window Estimation for Macroeconomic Forecasting”*  
by **Mariia Artemova, Francisco Blasques, Siem Jan Koopman, and Zhaokun Zhang**.

Author: Mariia Artemova (artemova@ese.eur.nl). First version date: 18 December 2025

The code is written primarily in Python and provides full replication materials for the simulation experiments and empirical results presented in the paper. 

# Repository Structure

```text
flexible-rolling-window/
├── frw/                 # source code 
│   ├── MonteCarlo/      # Monte Carlo simulation study
│   │   ├── run_MC.py    # Main script for the simulation study
│   │   └── fun/         # Helper functions for DGPs, estimation, tables, and plots
│   └── Empirics/        # Empirical application
│       ├── main.py      # Main script for the empirical analysis
│       ├── fun/         # Helper functions for forecast evaluation, figures, and tables
│       └── models/      # Forecasting model implementations
├── data/                # Input data used by the empirical application
│   ├── macro_data.csv   # U.S. macroeconomic time series
│   └── USREC.csv        # U.S. recession indicator data
└── results/             # Output files produced by the replication scripts
    ├── Figures/         # Generated figures
    └── tables/          # Generated LaTeX tables
```

The main executable scripts are `frw/MonteCarlo/run_MC.py` for the simulation study and `frw/Empirics/main.py` for the empirical application. The `data/` directory contains the input datasets required for the empirical analysis, while `results/` contains generated tables and figures.

## Monte Carlo Simulation Study (Section 5 of the paper)

**Location:** `frw/MonteCarlo/`  

**File:** `run_MC.py`  
Replicates **Table 2** and **Figure 3** from Section 5 of the paper.  
It implements the Flexible Rolling-Window (FRW) estimator and evaluates its performance relative to MLE across multiple simulation scenarios.

Run from the repository root:

```bash
python -m frw.MonteCarlo.run_MC
```

The script writes `results/tables/Table2.tex`, `results/Figures/E1.pdf`--`E4.pdf`, `results/res.npy`, and `results/gamma.npy`.

## Empirical Application (Section 6 of the paper)

**Location:** `frw/Empirics/`  

**File:** `main.py`  
Replicates **Tables 3-6** and **Figure 4** from Section 6.  
It estimates FRW, ORW, MLE, and Markov-Switching benchmark models using U.S. macroeconomic time series stored in `data/macro_data.csv` and generates the forecast evaluation results.

Run from the repository root to reproduce the main empirical results in **Tables 3-4** and **Figure 4**:

```bash
python -m frw.Empirics.main
```

This writes `results/tables/Table3.tex`, `results/tables/Table4.tex`, `results/Figures/alpha_FRW.pdf`, and `results/ForecastEval.pkl`.

To reproduce the appendix empirical results in **Table 5**, run:

```bash
python -m frw.Empirics.main --variant appendix
```

This writes `results/tables/Table5.tex`.

## Data

The data used in the paper are obtained from FRED, Federal Reserve Bank of St. Louis. The data is contained in two CSV files stored in `data/`:

- `macro_data.csv`: monthly U.S. macroeconomic series: `PAYEMS`, `UNRATE`, `CUMFNS`, `INDPRO`, and `SAHMCURRENT`.
- `USREC.csv`: monthly U.S. recession indicator.


# Software

The code is written primarily in Python. The empirical Markov-switching benchmark uses R through `rpy2` and requires the R package `MSwM`. When this benchmark is enabled, `MSwM::msmFit()` is run with parallelization enabled. 


The code was tested with the following versions:

```text
Python 3.12.4
R 4.4.1
```

Required Python packages:

```text
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
statsmodels==0.14.2
matplotlib==3.8.4
seaborn==0.13.2
tqdm==4.66.4
rpy2==3.6.4
```
Required R packages for the Markov-switching benchmark:

```text
MSwM==1.5  # external CRAN package
nlme       # MSwM dependency
parallel   # base R package 
```


# Hardware and running time

On a laptop with Apple Silicon M3, the Monte Carlo code takes around 2 hours to run. 

In the empirical application, the most computationally intensive part is the estimation of the Markov-switching benchmark. For this benchmark, parallel computation is enabled through the R package `MSwM`, which uses R's `parallel` package. On our test machine, this corresponds to 8 CPU cores. The empirical application in the main text then takes around 2 hours to run.
