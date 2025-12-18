"""
Monte Carlo simulation study for Flexible Rolling-Window (FRW) estimation.

This script conducts a Monte Carlo experiment to evaluate the
forecasting performance of the Flexible Rolling-Window (FRW) estimator
relative to a standard maximum-likelihood (ML) benchmark under a variety
of data-generating processes (DGPs).

The simulation design considers multiple scenarios featuring:
    - different forms of parameter instability,
    - exogenous regressors,
    - alternative weighting schemes (exponential and binary).

For each design, the script:
    1. Simulates artificial time series from a specified DGP.
    2. Selects the FRW hyperparameter by cross-validation.
    3. Computes recursive one-step-ahead forecasts using FRW and ML.
    4. Evaluates forecast accuracy using RMSE ratios.
    5. Stores estimated weighting parameters and performance statistics.
    6. Produces summary tables and plots.

The results replicate the Monte Carlo findings reported in the paper
“On the Use of Flexible Rolling-Window Estimation for Macroeconomic Forecasting”.

Dependencies
------------
- numpy
- scipy.optimize
- frw.MonteCarlo.fun (custom simulation, estimation, and plotting utilities)

Outputs
-------
- results/res.npy      : Average RMSE ratios across replications
- results/gamma.npy   : Estimated FRW hyperparameters across replications
- Printed LaTeX-style summary table
- Diagnostic plots of estimated hyperparameters

Author
------
Mariia Artemova
"""

import numpy as np
from scipy.optimize import minimize

from frw.MonteCarlo.fun.fun import weighted_ar1, weighted_ar1_X
from frw.MonteCarlo.fun.create_table import create_experiment_table
from frw.MonteCarlo.fun.dgps import CreatDGP1, CreatDGP2, CreatDGP3, CreatDGP4
from frw.MonteCarlo.fun.plots import create_plots

from pathlib import Path

# Project root (adjust parents[...] if needed)
BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "Figures"

# Create directories if they do not exist
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ------------------ Main simulation loop ------------------ #
iD = 1000 # number of MC replications
verbose = False
np.random.seed(0)


vRMSE = np.zeros((iD, 2))
vRatio = np.zeros(iD)

# ------------------ Settings ------------------ #
lRW = [60, 120, 240, 480]

iT = 600 + 200
iCVbegin = 600
iH = 1      # forecast horizon
iLag = 1    # AR(1)
    
alpha, beta, sigma = 0.13, 0.9, 0.5
iTest = 200

lSetups = [1, 2, 3, 4]

vGamma = np.zeros(iD)
res_Ratio = np.zeros((2 + 2 + 4*3, len(lRW)))
res_Ratio_std = np.zeros((2 + 2 + 4*3, len(lRW)))
res_Gamma = np.zeros((2 + 2 + 4*3, len(lRW), iD))
idx = 0
weight_type = ['exp', 'binary']
for setup in lSetups:
    print('Setup ', setup)
    if setup == 1:
        lPar = [0.5, 0.9]
    elif setup == 2:
        lPar = [776, 728]
    elif setup == 3 or setup == 4:
        lPar = [72, 144, 288]
   
    for wt in weight_type:
        if (setup == 1 or setup == 2) and wt == 'binary':
            continue
        if setup == 4 and wt == 'exp':
            p=2
            wt = 'binary'
        else:
            p=10
            
        for k, K in enumerate(lPar):
            print('Cycle length = ', K)
            for tau, iTau in enumerate(lRW):
                print('RW length = ', iTau)
                for rep in range(iD):
                    if setup == 1:
                        vY = CreatDGP1(alpha, sigma, iT + iTest, K)
                        from frw.MonteCarlo.fun.fun import RMSEFRW
    
                    elif setup == 2:
                        vY = CreatDGP2(alpha, sigma, iT + iTest, K)
                        from frw.MonteCarlo.fun.fun import RMSEFRW
                        
                    elif setup == 3:
                        vY, z = CreatDGP3(alpha, sigma, iT + iTest, K)
                        if wt == 'binary':
                            from frw.MonteCarlo.fun.fun import RMSEFRW_binary as RMSEFRW
                        elif wt == 'exp':
                            from frw.MonteCarlo.fun.fun import RMSEFRW
                    elif setup == 4:
                        vY, vX, z = CreatDGP4(alpha, sigma, iT + iTest, K, p=p)
                        from frw.MonteCarlo.fun.fun import RMSEFRW_X as RMSEFRW

                    if wt == 'binary':
                        if setup == 4 :
                            fun = lambda g: RMSEFRW(g[0], vY[:iT], vX[:iT], z[:iT], iTau, iCVbegin, iT)
                        else:    
                            fun = lambda g: RMSEFRW(g[0], vY[:iT], z[:iT], iTau, iCVbegin, iT)
                    elif wt == 'exp':
                        fun = lambda g: RMSEFRW(g[0], vY[:iT], iTau, iCVbegin, iT)
        
                    
     
                    if wt == 'exp':
                        res = minimize(fun, x0=[np.log((0.9) / (1 - 0.9))], method="BFGS")
                        gamma_hat = float(np.exp(res.x[0]) / (1 + np.exp(res.x[0]))) if res.success else 1.0
                        w = gamma_hat ** (iTau - np.arange(1, iTau + 1))
                    else:
                        res = minimize(fun, x0=[np.log((0.8 - 1) / (0.5 - 0.8))], method="BFGS")
                        gamma_hat = float(1 - 0.8 * np.exp(res.x[0]) / (1 + np.exp(res.x[0]))) if res.success else 0.5
                    verror_ML, verror_FRW = [], []
                    y_begin = iT - iTau
                    for it in range(y_begin, iT + iTest - iTau - iH + 1):
                        start = it + iH - 1
                        end = it + iTau + iH - 1
                        g_vRollY = vY[start:end]  # length RW 
        
                        Y = g_vRollY
                        
                        if setup == 4:
                            g_vRollX = vX[start:end, :]  # length RW 
                            X = g_vRollX
                            
                        if wt == 'binary':
                            g_vRollZ = z[start - 1:end - 1]
                            Z_target = z[it + iTau - 1]
                            diff2 = (g_vRollZ - Z_target) ** 2

                            w = gamma_hat * (1 - diff2) + (1 - gamma_hat) * diff2
                        
                        if setup == 4:
                            # compute coefficient from FRW and ML
                            beta_FRW = weighted_ar1_X(Y, X, w)[0]
                            beta_ML = weighted_ar1_X(Y, X, np.ones(iTau))[0]

                            # 1-step ahead forecast: use last available lag value 
                            dyhat_FRW = beta_FRW[0] + beta_FRW[1] * vY[it + iTau - 1] + beta_FRW[2:] * vX[it + iTau - 1, :]
                            dyhat_ML = beta_ML[0] + beta_ML[1] * vY[it + iTau - 1] + beta_ML[2:] * vX[it + iTau - 1, :]
                        else:
                            # compute coefficient from FRW and ML
                            beta_FRW = weighted_ar1(Y, w)[0]
                            beta_ML = weighted_ar1(Y, np.ones(iTau))[0]
            
                            # 1-step ahead forecast: use last available lag value 
                            dyhat_FRW = beta_FRW[0] + beta_FRW[1] * vY[it + iTau - 1]
                            dyhat_ML = beta_ML[0] + beta_ML[1] * vY[it + iTau - 1]
            
                        # actual y at forecast point
                        actual_index = it + iTau + iH - 1
                        actual = vY[actual_index]
        
                        verror_ML.append(actual - dyhat_ML)
                        verror_FRW.append(actual - dyhat_FRW)
        
                    verror_FRW = np.asarray(verror_FRW, dtype=float)
                    verror_ML = np.asarray(verror_ML, dtype=float)
        
                    vRMSE[rep, 0] = np.sqrt(np.mean(verror_ML ** 2))
                    vRMSE[rep, 1] = np.sqrt(np.mean(verror_FRW ** 2))
                    vGamma[rep] = gamma_hat
                    vRatio[rep] = vRMSE[rep, 1] / vRMSE[rep, 0]
                  
                    if verbose and (rep % 50 == 0):
                        print(f"iter {rep:4d}: gamma={gamma_hat:.5g}, loss={vRatio[rep]:.5g}, success={res.success}")
                    
                    
                res_Ratio[idx, tau] = np.mean(vRatio)
                res_Gamma[idx, tau, :] = vGamma
            
            idx += 1
            

np.save(RESULTS_DIR / "res.npy", res_Ratio)
np.save(RESULTS_DIR / "gamma.npy", res_Gamma)

tab = create_experiment_table(res_Ratio, TABLES_DIR)

print(tab)

Idx = [0, 3, 7, 13]
for i, idx in enumerate(Idx):
    create_plots(res_Gamma[idx, -1, :], FIGURES_DIR, E=i+1)

    
    

