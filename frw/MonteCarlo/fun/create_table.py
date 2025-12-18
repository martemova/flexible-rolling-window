import numpy as np


def create_experiment_table(data, RESULTS_DIR):
    """
        Construct a LaTeX table summarizing Monte Carlo RMSE ratios.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (16, 4) containing average RMSE ratios for
            four rolling-window lengths across all simulation experiments.

        Returns
        -------
        str
            LaTeX table code as a single string.
    """
    latex_lines = []
    latex_lines.append(r"\begin{table}[ht!]\centering")
    latex_lines.append(r"\begin{tabular}{l c c c c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"  &  $\tau=60$     &    $\tau=120$     &   $\tau=240$ &   $\tau=480$  \\")
    latex_lines.append(r"\toprule")

    # Experiment 1
    latex_lines.append(r"\textit{Experiment 1}\\")
    latex_lines.append("        & " + " & ".join(f"{x:.4f}" for x in data[0]) + r"\\")
    latex_lines.append("        & " + " & ".join(f"{x:.4f}" for x in data[1]) + r"\\[1ex]")

    # Experiment 2
    latex_lines.append(r"\textit{Experiment 2}\\")
    latex_lines.append("JAB  & " + " & ".join(f"{x:.4f}" for x in data[2]) + r"\\")
    latex_lines.append("LAB  & " + " & ".join(f"{x:.4f}" for x in data[3]) + r"\\[1ex]")

    # Experiment 3
    latex_lines.append(r"\textit{Experiment 3}\\")
    latex_lines.append(r"Exponential decay weights\\")
    latex_lines.append("$B=72$  & " + " & ".join(f"{x:.4f}" for x in data[4]) + r"\\")
    latex_lines.append("$B=144$  & " + " & ".join(f"{x:.4f}" for x in data[5]) + r"\\")
    latex_lines.append("$B=288$  & " + " & ".join(f"{x:.4f}" for x in data[6]) + r"\\[1ex]")

    latex_lines.append(r"Binary weights\\")
    latex_lines.append("$B=72$  & " + " & ".join(f"{x:.4f}" for x in data[7]) + r"\\")
    latex_lines.append("$B=144$  & " + " & ".join(f"{x:.4f}" for x in data[8]) + r"\\")
    latex_lines.append("$B=288$  & " + " & ".join(f"{x:.4f}" for x in data[9]) + r"\\[1ex]")

    # Experiment 4
    latex_lines.append(r"\textit{Experiment 4 }\\")
    latex_lines.append("Binary weights\\    [1ex]")
    latex_lines.append("$R=2$\\")
    latex_lines.append("$B=72$  & " + " & ".join(f"{x:.4f}" for x in data[10]) + r"\\")
    latex_lines.append("$B=144$  & " + " & ".join(f"{x:.4f}" for x in data[11]) + r"\\")
    latex_lines.append("$B=288$  & " + " & ".join(f"{x:.4f}" for x in data[12]) + r"\\[2ex]")

    latex_lines.append("$R=10$\\")
    latex_lines.append("$B=72$  & " + " & ".join(f"{x:.4f}" for x in data[13]) + r"\\")
    latex_lines.append("$B=144$  & " + " & ".join(f"{x:.4f}" for x in data[14]) + r"\\")
    latex_lines.append("$B=288$  & " + " & ".join(f"{x:.4f}" for x in data[15]) + r"\\")

    latex_lines.append(r"\toprule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(
        r"\caption{Average RMSE ratio: FRW relative to MLE for different experiments. \small{This table presents the sample average RMSE ratio of AR(1)/ADL(1,1) model with FRW method relative to a simple AR(1)/ADL(1,1) model estimated using MLE. Forecasts are calculated for $T=812:1000$. } }\label{tab:E5}")
    latex_lines.append(r"\end{table}")

    # Save to file
    with open(RESULTS_DIR/"Table2.tex", "w") as f:
        f.write("\n".join(latex_lines))
    return "\n".join(latex_lines)