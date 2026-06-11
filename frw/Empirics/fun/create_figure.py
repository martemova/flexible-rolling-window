import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def figure(res, Z, x, s='UNRATE', p=4, k=1):
    """
    Plot the state variable and time-varying AR coefficients.
    Parameters
    ----------
    res : dict
        Dictionary containing estimation results. Expected to include
        entries of the form ``f"{s}_AR{p}"`` with keys:
        ``'theta_FRW'`` and ``'theta_ML'``, each of shape
        (T, n_horizons, p+1).
    Z : array
        Estimated state variable \\( \\hat{Z}_t \\) over the out-of-sample
        period.
    x : array
        Time index
    s : str, default="UNRATE"
        Name of the macroeconomic series to plot.
    p : int, default=4
        Autoregressive order of the model.
    k : int, default=1
        Index of the coefficient to plot:
        ``k = 0`` corresponds to the intercept,
        ``k >= 1`` corresponds to the AR(k) coefficient.

    """
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "results" / "Figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_eps = output_dir / "alpha_FRW.eps"
    output_path_jpg = output_dir / "alpha_FRW.jpg"
    theta_FRW = res[f'{s}_AR{p}']['theta_FRW'][:, 0, k]
    theta_ML = res[f'{s}_AR{p}']['theta_ML'][:, 0, k]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    # --- Primary axis: Z_t ---
    l1 = ax1.plot(x, Z, label='$\\hat{Z}_{t}$', linewidth=2, linestyle='-.')
    ax1.set_ylabel('$\\hat{Z}_{t}$', fontsize=18, rotation=0, labelpad=15)
    ax1.tick_params(axis='y', labelsize=18)

    # --- Secondary axis: coefficients ---
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, theta_FRW, label='$\\hat{\\alpha}_{1}$, FRW', linewidth=2, color='green')
    l3 = ax2.plot(x, theta_ML, label='$\\hat{\\alpha}_{1}$, ML', linewidth=2, color='orange')
    ax2.set_ylabel('$\\hat{\\alpha}_{1}$', fontsize=18, rotation=0,  labelpad=15)
    ax2.tick_params(axis='y', labelsize=18)

    # --- Recession shading ---
    recessions = [("2007-12-01", "2009-06-30")]
    for start, end in recessions:
        ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lavender')

    # --- Combine legends ---
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=16, loc='upper right')
    for label in ax1.get_yticklabels():
        label.set_rotation(0)
        label.set_ha('right')

    ax1.set_xlabel("Date", fontsize=18)
    ax1.tick_params(axis='x', labelsize=16)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path_eps, format="eps")
    fig.savefig(output_path_jpg, format="jpg", dpi=300)
    plt.show()
