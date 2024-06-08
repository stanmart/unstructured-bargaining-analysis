import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def prepare_dataset(outcomes: pl.DataFrame) -> pl.DataFrame:
    player_names = pl.Enum(["P1", "P2", "P3"])
    agreement_types = pl.Enum(["Breakdown", "Partial agreement", "Full agreement"])

    roles = {
        1: "P1",
        2: "P2",
        3: "P3",
    }

    values = pl.from_dicts(
        [
            {
                "treatment_name": "treatment_dummy_player",
                "role": "P1",
                "shapley_value": 50.0,
                "nucleolus": 50.0,
            },
            {
                "treatment_name": "treatment_dummy_player",
                "role": "P2",
                "shapley_value": 50.0,
                "nucleolus": 50.0,
            },
            {
                "treatment_name": "treatment_dummy_player",
                "role": "P3",
                "shapley_value": 0.0,
                "nucleolus": 0.0,
            },
            {
                "treatment_name": "treatment_y_10",
                "role": "P1",
                "shapley_value": 110 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_10",
                "role": "P2",
                "shapley_value": 95 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_10",
                "role": "P3",
                "shapley_value": 95 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_30",
                "role": "P1",
                "shapley_value": 130 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_30",
                "role": "P2",
                "shapley_value": 85 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_30",
                "role": "P3",
                "shapley_value": 85 / 3,
                "nucleolus": 100 / 3,
            },
            {
                "treatment_name": "treatment_y_90",
                "role": "P1",
                "shapley_value": 190 / 3,
                "nucleolus": 90,
            },
            {
                "treatment_name": "treatment_y_90",
                "role": "P2",
                "shapley_value": 55 / 3,
                "nucleolus": 5,
            },
            {
                "treatment_name": "treatment_y_90",
                "role": "P3",
                "shapley_value": 55 / 3,
                "nucleolus": 5,
            },
        ]
    ).with_columns(
        role=pl.col("role").cast(player_names),
    )

    df = (
        outcomes.filter(pl.col("round_number") > 1)
        .with_columns(
            role=pl.col("id_in_group").replace(roles).cast(player_names),
            agreement=(
                pl.when(
                    pl.col("payoff_this_round")
                    .sum()
                    .over(["session_code", "round_number", "group_id"])
                    == 0
                )
                .then(pl.lit("Breakdown"))
                .otherwise(
                    pl.when(
                        (pl.col("payoff_this_round") > 0)
                        .all()
                        .over(["session_code", "round_number", "group_id"])
                    )
                    .then(pl.lit("Full agreement"))
                    .otherwise(pl.lit("Partial agreement"))
                )
            ).cast(agreement_types),
            treatment_name_nice=pl.col("treatment_name").replace(
                {
                    "treatment_dummy_player": "Dummy player",
                    "treatment_y_10": "Y = 10",
                    "treatment_y_30": "Y = 30",
                    "treatment_y_90": "Y = 90",
                }
            ),
            matching_group=pl.when(pl.col("treatment_name") == "treatment_y_10")
            .then(((pl.col("participant_id") - 1) // 6))
            .when(pl.col("treatment_name") == "treatment_y_30")
            .then(((pl.col("participant_id") - 1) // 6) + 1 * 6)
            .when(pl.col("treatment_name") == "treatment_y_90")
            .then(((pl.col("participant_id") - 1) // 6) + 2 * 6)
            .otherwise(((pl.col("participant_id") - 1) // 6) + 3 * 6),
        )
        .join(
            values,
            on=["treatment_name", "role"],
            how="left",
        )
    )

    return df


def prepare_for_mw(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.filter((pl.col("agreement") != "Breakdown") & (pl.col("role") == "P1"))
        .group_by(["treatment_name_nice", "matching_group"])
        .mean()
        .select(pl.col(["treatment_name_nice", "matching_group", "payoff_this_round"]))
    )
    return df


def prepare_for_reg(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.with_columns(
            Y=df.to_series(df.get_column_index("treatment_name_nice"))
            .str.strip_chars("Y = ")
            .str.to_integer(strict=False)
        )
        .filter(
            (pl.col("treatment_name_nice") != "Dummy player")
            & (pl.col("agreement") != "Breakdown")
            & (pl.col("role") == "P1")
        )
        .select(
            [
                "treatment_name_nice",
                "payoff_this_round",
                "Y",
                "group_id",
                "matching_group",
            ]
        )
        .to_pandas()
    )

    return df


def compute_one_sided_mw(
    df: pl.DataFrame, Y1: int, Y2: int
) -> scipy.stats._mannwhitneyu.MannwhitneyuResult:
    res = scipy.stats.mannwhitneyu(
        df.filter((pl.col("treatment_name_nice") == f"Y = {Y1}")).select(
            pl.col("payoff_this_round")
        ),
        df.filter((pl.col("treatment_name_nice") == f"Y = {Y2}")).select(
            pl.col("payoff_this_round")
        ),
        alternative="less",
        method="auto",
    )
    return res


def run_reg(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResults:
    res = sm.regression.linear_model.OLS(
        df.payoff_this_round, sm.add_constant(df.Y)
    ).fit(cov_type="cluster", cov_kwds={"groups": df.matching_group})
    return res


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)
    df_mw = prepare_for_mw(df)
    df_reg = prepare_for_reg(df)

    f = open("out/analysis/analysis_results.txt", "w")
    f.write("Analysis results \n")
    f = open("out/analysis/analysis_results.txt", "a")

    # Mann-Whitney U: Y=10 vs Y=90 (main analysis), Y=30 vs Y=90 (robustness check), Y=10 vs Y=30 (exploratory analysis)
    for Y1, Y2 in [(10, 90), (30, 90), (10, 30)]:
        res = compute_one_sided_mw(df_mw, Y1, Y2)
        f.write(
            f"The p-value of the one-sided Mann-Whitney U test comparing Y = {Y1} and Y = {Y2} on the matching-group average of player A is {res.pvalue}."
        )

    # regression (robustness check)
    reg_res = run_reg(df_reg)
    f.write("\n\nRegression results: \n")
    for table in reg_res.summary().tables[:2]:
        f.write(table.as_latex_tabular())

    # mean squared error (exploratory analysis)
    mse_nuc = mean_squared_error(
        df.select(pl.col("payoff_this_round")), df.select(pl.col("nucleolus"))
    )
    mse_shap = mean_squared_error(
        df.select(pl.col("payoff_this_round")), df.select(pl.col("shapley_value"))
    )
    mse_es = mean_squared_error(
        df.select(pl.col("payoff_this_round")),
        np.ones(len(df.select(pl.col("payoff_this_round")))) * 33,
    )
    f.write(
        f"\n\n Mean squared error of the nucleolus, Shapley value, and equal split: {mse_nuc, mse_shap, mse_es}. \n"
    )

    f.close()
