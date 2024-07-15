import json

import pandas as pd


def create_nonarametric_table(mw_results: dict[str, dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(mw_results).T
    table = table.rename(
        columns={
            "statistic": "U statistic",
            "pvalue": "p-value",
        }
    )
    table = table.rename(
        index=lambda s: "{" + " < ".join(f"[Y = {t}]" for t in s.split("-")) + "}"
    )
    return table


if __name__ == "__main__":
    with open(snakemake.input.mann_whitney, "r") as f:  # noqa F821 # type: ignore
        mw_results = json.load(f)

    table = create_nonarametric_table(mw_results)

    latex_table = table.to_latex(float_format="%.3f")
    with open(snakemake.output.table, "w") as f:  # noqa F821 # type: ignore
        f.write(latex_table)
