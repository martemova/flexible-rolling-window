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

        Results are reported for the subsamples available in ``tables``.

        Parameters
        ----------
        tables : dict
            Dictionary containing forecast evaluation results for each series.
            For each series ``s``, the dictionary must include
            ``tables[s]['res']`` and may optionally include
            ``tables[s]['res_expansion']`` and ``tables[s]['res_recess']``.
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
    n_rows = tables[series_list[0]]['res'].shape[0]
    sections = [("res", "Forecasting over mixed economic conditions")]
    sample_table = tables[series_list[0]]
    if "res_expansion" in sample_table:
        sections.append(("res_expansion", "Forecasting during expansion"))
    if "res_recess" in sample_table:
        sections.append(("res_recess", "Forecasting during Great recession"))

    cols = []
    for s in series_list:
        cols += [f"{s}_Ratio", f"{s}_rho2"]
    df = pd.DataFrame(np.nan, index=range(n_rows * len(sections)), columns=cols, dtype=object)

    # Fill the DataFrame
    for s in series_list:
        for section_idx, (section_key, _) in enumerate(sections):
            table_section = tables[s][section_key]
            base_row = section_idx * n_rows
            for i in range(0, n_rows, 2):
                ratio_row = table_section[i, :2] if horizon == 1 else table_section[i, 2:]
                dm_row = table_section[i+1, :2] if horizon == 1 else table_section[i+1, 2:]
                formatted = format_row(ratio_row, dm_row)
                df.loc[base_row + i, f"{s}_Ratio"] = formatted[0]
                df.loc[base_row + i, f"{s}_rho2"] = formatted[1]
                df.loc[base_row + i + 1, f"{s}_Ratio"] = fmt_dm(dm_row[0])
                df.loc[base_row + i + 1, f"{s}_rho2"] = fmt_dm(dm_row[1])

    row_labels = ["FRW/MLE", "", "FRW/ORW", ""]
    if n_rows == 6:
        row_labels += ["FRW/MS", ""]
    index_labels = row_labels * len(sections)
    df.index = index_labels

    # --- Build LaTeX manually with raw strings ---
    latex_lines = []
    latex_lines.append(r"\begin{table}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{%s}" % ("l" + "ll"*len(series_list)))
    latex_lines.append(r"\toprule")
    latex_lines.append(" & ".join([r""] + [rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in series_list]) + r" \\")
    latex_lines.append(" & ".join([r""] + ["Ratio", r"$\rho_2$"] * len(series_list)) + r" \\")

    # Section headers and rows
    for sec_idx, (_, section_name) in enumerate(sections):
        start_row = sec_idx * n_rows
        latex_lines.append(r"\midrule")
        latex_lines.append(rf"\multicolumn{{{1 + len(series_list)*2}}}{{l}}{{\textit{{{section_name}}}}}\\\midrule")

        for row_offset in range(0, n_rows, 2):
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
