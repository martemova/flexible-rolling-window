import pandas as pd
import numpy as np


def format_row(ratio_row, dm_row, is_orw=False):
    """
        Format a row of RMSE ratios and associated parameters for LaTeX output.

        This function formats numerical values to three decimals and appends
        significance stars based on Diebold–Mariano (DM) test p-values.
        Parameter columns are formatted differently for ORW and FRW models.

        Parameters
        ----------
        ratio_row : array
            Array containing RMSE ratios and parameter values in alternating columns.
        dm_row : array or None
            Array of DM test p-values corresponding to the ratio columns.
            If None, no significance stars are added.
        is_orw : bool, default=False
            If True, parameter values are formatted as integers (used for ORW
            window length). Otherwise, parameters are formatted as floats.

        Returns
        -------
        list of str
            Formatted strings for a single table row, ready for LaTeX output.
        """
    out = []
    for j, x in enumerate(ratio_row):
        if pd.isna(x):
            out.append("")
            continue
        if j % 2 == 0:  # Ratio column
            p = dm_row[j] if dm_row is not None else None
            stars = ""
            if p is not None:
                if p <= 0.01:
                    stars = "$^{***}$"
                elif p <= 0.05:
                    stars = "$^{**}$"
                elif p <= 0.10:
                    stars = "$^{*}$"
            out.append(f"{float(x):.3f}{stars}")
        else:  
            if is_orw:
                out.append(f"{int(round(float(x)))}")
            else:
                out.append(f"{float(x):.3f}")
    return out


def fmt_dm(x):
    """
       Format a Diebold–Mariano test p-value for LaTeX output.

       Parameters
       ----------
       x : float or NaN
           DM test p-value.

       Returns
       -------
       str
           Formatted p-value enclosed in parentheses, or an empty string
           if the value is missing.
       """
    if pd.isna(x):
        return ""
    return f"({x:.3f})"

def build_table(tables, series_list, horizon, horizon_label):
    """
        Construct a LaTeX table summarizing forecast comparisons.

        This function builds a LaTeX table reporting:
            - RMSE ratios of FRW relative to competing models
            - Diebold–Mariano test p-values
            - Selected tuning parameters (FRW and ORW)

        Results are reported for three subsamples:
            1. Mixed economic conditions
            2. Expansion periods
            3. Recession periods

        Parameters
        ----------
        tables : dict
            Dictionary containing forecast evaluation results for each series.
            For each series ``s``, the dictionary must include:
                - ``tables[s]['res']``: full-sample results
                - ``tables[s]['res_expansion']``: expansion results
                - ``tables[s]['res_recess']``: recession results
        series_list : list of str
            List of series names to include in the table.
        horizon : int
            Forecast horizon (e.g., 1 or 3).
        horizon_label : str
            Label used in the table caption (e.g., ``"n=1"`` or ``"n=3"``).

        Returns
        -------
        str
            A LaTeX-formatted table as a single string.
        """
    n_rows = 6  # FRW/MLE, DM, FRW/ORW, DM, FRW/MS, DM
    cols = []
    for s in series_list:
        cols += [f"{s}_Ratio", f"{s}_rho2"]
    df = pd.DataFrame(np.nan, index=range(n_rows * 3), columns=cols, dtype=object)  # 3 sections

    # Fill the DataFrame
    for s in series_list:
        table_full = tables[s]['res']
        table_exp = tables[s]['res_expansion']
        table_recess = tables[s]['res_recess']
        for i in range(0, n_rows, 2):
            # Full sample
            ratio_row = table_full[i, :2] if horizon == 1 else table_full[i, 2:]
            dm_row = table_full[i+1, :2] if horizon == 1 else table_full[i+1, 2:]
            formatted = format_row(ratio_row, dm_row)
            df.loc[i, f"{s}_Ratio"] = formatted[0]
            df.loc[i, f"{s}_rho2"] = formatted[1]
            df.loc[i+1, f"{s}_Ratio"] = fmt_dm(dm_row[0])
            df.loc[i+1, f"{s}_rho2"] = fmt_dm(dm_row[1])

            # Expansion sample
            ratio_row = table_exp[i, :2] if horizon == 1 else table_exp[i, 2:]
            dm_row = table_exp[i+1, :2] if horizon == 1 else table_exp[i+1, 2:]
            formatted = format_row(ratio_row, dm_row)
            df.loc[i+6, f"{s}_Ratio"] = formatted[0]
            df.loc[i+6, f"{s}_rho2"] = formatted[1]
            df.loc[i+7, f"{s}_Ratio"] = fmt_dm(dm_row[0])
            df.loc[i+7, f"{s}_rho2"] = fmt_dm(dm_row[1])

            # Recession sample
            ratio_row = table_recess[i, :2] if horizon == 1 else table_recess[i, 2:]
            dm_row = table_recess[i+1, :2] if horizon == 1 else table_recess[i+1, 2:]
            formatted = format_row(ratio_row, dm_row)
            df.loc[i+12, f"{s}_Ratio"] = formatted[0]
            df.loc[i+12, f"{s}_rho2"] = formatted[1]
            df.loc[i+13, f"{s}_Ratio"] = fmt_dm(dm_row[0])
            df.loc[i+13, f"{s}_rho2"] = fmt_dm(dm_row[1])

    index_labels = ["FRW/MLE","", "FRW/ORW","", "FRW/MS",""]*3
    df.index = index_labels

    # --- Build LaTeX manually with raw strings ---
    latex_lines = []
    latex_lines.append(r"\begin{table}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{%s}" % ("l" + "ll"*len(series_list)))
    latex_lines.append(r"\toprule")
    latex_lines.append(" & ".join([r""] + [rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in series_list]) + r" \\")
    latex_lines.append(" & ".join([r"", "Ratio", r"$\rho_2$"] * len(series_list)) + r" \\")

    # Section headers and rows
    section_rows = [0, 6, 12]  # starting row of each section
    section_names = [
        "Forecasting over mixed economic conditions",
        "Forecasting during expansion",
        "Forecasting during Great recession"
    ]

    for sec_idx, start_row in enumerate(section_rows):
        latex_lines.append(r"\midrule")
        latex_lines.append(rf"\multicolumn{{{1 + len(series_list)*2}}}{{l}}{{\textit{{{section_names[sec_idx]}}}}}\\\midrule")

        for row_offset in range(0, 6, 2):
            row_main = start_row + row_offset
            row_dm = row_main + 1

            # Main ratio row
            row_values = [str(df.iloc[row_main, c]) for c in range(len(df.columns))]
            latex_lines.append(f"{df.index[row_main]} & " + " & ".join(row_values) + r" \\")

            # DM row with [1ex] spacing
            row_values_dm = [str(df.iloc[row_dm, c]) for c in range(len(df.columns))]
            latex_lines.append(f"{df.index[row_dm]} & " + " & ".join(row_values_dm) + r" \\[1ex]")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(rf"\caption{{Out-of-sample forecasts, horizon {horizon_label}.}}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)
