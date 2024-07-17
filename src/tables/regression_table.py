import pickle
import re

from statsmodels.iolib.summary2 import Summary, summary_col
from statsmodels.regression.linear_model import RegressionResultsWrapper


def create_regression_table(models: list[RegressionResultsWrapper]) -> Summary:
    return summary_col(
        models,
        stars=True,
        float_format="%.2f",
        model_names=["(1)", "(2)"],
        include_r2=False,
        info_dict={"N": lambda reg: f"{int(reg.nobs):d}"},
    )


if __name__ == "__main__":
    model_paths = [path for path in snakemake.input]  # noqa F821 # type: ignore
    models = []
    for path in model_paths:
        with open(path, "rb") as f:
            reg = pickle.load(f)
            if not isinstance(reg, RegressionResultsWrapper):
                raise ValueError("Model must be a statsmodels regression result")
            models.append(reg)

    table = create_regression_table([model for model in models])
    latex_table = table.as_latex()

    pattern = re.compile(r"\\begin{tabular}.*\\end{tabular}", re.DOTALL)
    latex_table_wo_fluff = pattern.search(latex_table).group()  # type: ignore

    table_head, *table_mid, table_foot = latex_table_wo_fluff.split("\\hline")
    latex_table_nicer = (
        table_head
        + "\\toprule"
        + "\\midrule".join(table_mid)
        + "\\bottomrule"
        + table_foot
    )

    with open(snakemake.output.table, "w") as f:  # noqa F821 # type: ignore
        f.write(latex_table_nicer)
