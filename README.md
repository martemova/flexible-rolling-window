# FRW

This repository provides the code and replication materials for the paper  
*“On the Use of Flexible Rolling-Window Estimation for Macroeconomic Forecasting”*  
by **Mariia Artemova, Francisco Blasques, Siem Jan Koopman, and Zhaokun Zhang**.

The code is written in Python and provides full replication materials for the simulation experiments and empirical results presented in the paper.

# Repository Structure

The repository is organized into two main components:

## 1. Monte Carlo Simulation Study

**Location:** `frw/MonteCarlo/`  

**File:** `run_MC.py`  
Replicates **Table 2** and **Figure 3** from Section 5 of the paper.  
It implements the Flexible Rolling-Window (FRW) estimator and evaluates its performance relative to MLE across multiple simulation scenarios.

## 2. Empirical Application

**Location:** `frw/Empirics/`  

**File:** `main.py`  
Replicates **Tables 3 and 4** and **Figure 4** from Section 6.  
It estimates FRW, ORW, MLE, and Markov-Switching benchmark models using U.S. macroeconomic time series stored in `data/macro_data.csv` and generates the forecast evaluation results.
