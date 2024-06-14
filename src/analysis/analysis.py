import json
import pickle

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


def prepare_for_reg(df: pl.DataFrame) -> pd.DataFrame:
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
    )

    return df.to_pandas()


def prepare_for_axiom_tests(df: pl.DataFrame) -> pl.DataFrame:
    df = df.pivot(
        index=["treatment_name_nice", "round_number", "group_id"],
        columns="id_in_group",
        values="payoff_this_round",
    )

    df = df.with_columns(
        Y=df.to_series(0).str.strip_chars("Y = ").str.to_integer(strict=False)
    ).with_columns(
        coal12=pl.col("1") + pl.col("2"),
        coal13=pl.col("1") + pl.col("3"),
        grand_coal=pl.col("1") + pl.col("2") + pl.col("3"),
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


def run_reg(
    df: pd.DataFrame, dummies: bool = False
) -> sm.regression.linear_model.RegressionResultsWrapper:
    if dummies:
        df = pd.get_dummies(
            df,
            columns=["treatment_name_nice"],
            prefix="",
            prefix_sep="",
            drop_first=True,
            dtype=np.float64,
        )
        model = sm.regression.linear_model.OLS(
            df.payoff_this_round,
            sm.add_constant(df[["Y = 30", "Y = 90"]]),
        )
    else:
        model = sm.regression.linear_model.OLS(
            df.payoff_this_round, sm.add_constant(df.Y)
        )
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df.matching_group})
    return res


def test_efficiency(df: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_tmp = df_test.clone()
    for treatment in df["treatment_name_nice"].unique():
        vals = df.filter(pl.col("treatment_name_nice") == treatment)["grand_coal"]
        res = scipy.stats.mannwhitneyu(
            vals, np.ones(len(vals)) * 100, alternative="two-sided", method="auto"
        )
        df_res = pl.DataFrame(
            {
                "axiom": "efficiency",
                "treatment_name_nice": f"{treatment}",
                "details": "",
                "statistic": res.statistic.item(),
                "pvalue": res.pvalue.item(),
            }
        )
        df_tmp = pl.concat([df_tmp, df_res])
    return df_tmp


def test_symmetry(df: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_tmp = df_test.clone()
    for treatment in df["treatment_name_nice"].unique():
        if treatment == "Dummy player":
            res = scipy.stats.mannwhitneyu(
                df.filter((df["treatment_name_nice"] == treatment))["1"],
                df.filter((df["treatment_name_nice"] == treatment))["2"],
                alternative="two-sided",
                method="auto",
            )
        else:
            res = scipy.stats.mannwhitneyu(
                df.filter((df["treatment_name_nice"] == treatment))["2"],
                df.filter((df["treatment_name_nice"] == treatment))["3"],
                alternative="two-sided",
                method="auto",
            )

        df_res = pl.DataFrame(
            {
                "axiom": "symmetry",
                "treatment_name_nice": f"{treatment}",
                "details": "",
                "statistic": res.statistic.item(),
                "pvalue": res.pvalue.item(),
            }
        )
        df_tmp = pl.concat([df_tmp, df_res])
    return df_tmp


def test_stability(df: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_tmp = df_test.clone()
    for treatment in df["treatment_name_nice"].unique():
        if treatment == "Dummy player":
            res = scipy.stats.mannwhitneyu(
                df.filter(df["treatment_name_nice"] == treatment)["coal12"],
                np.ones(60) * 100,
                alternative="two-sided",
                method="auto",
            )

            df_res = pl.DataFrame(
                {
                    "axiom": "stability",
                    "treatment_name_nice": f"{treatment}",
                    "details": "",
                    "statistic": res.statistic.item(),
                    "pvalue": res.pvalue.item(),
                }
            )
            df_tmp = pl.concat([df_tmp, df_res])
        else:
            # sum of payoffs of small coalition of P1 and P2 >= Y
            res_12 = scipy.stats.mannwhitneyu(
                df.filter(pl.col("treatment_name_nice") == treatment)["coal12"],
                df.filter(pl.col("treatment_name_nice") == treatment)["Y"],
                alternative="greater",
                method="auto",
            )

            # sum of payoffs of small coalition of P1 and P3 >= Y
            res_13 = scipy.stats.mannwhitneyu(
                df.filter(pl.col("treatment_name_nice") == treatment)["coal13"],
                df.filter(pl.col("treatment_name_nice") == treatment)["Y"],
                alternative="greater",
                method="auto",
            )

            # grand coalition payoffs == 100
            # same as efficiency above
            res_grand = scipy.stats.mannwhitneyu(
                df.filter(pl.col("treatment_name_nice") == treatment)["grand_coal"],
                np.ones(60) * 100,
                alternative="two-sided",
                method="auto",
            )

            coal_names = {res_12: "coal_12", res_13: "coal_13", res_grand: "grand_coal"}
            for res in [res_12, res_13, res_grand]:
                df_res = pl.DataFrame(
                    {
                        "axiom": "stability",
                        "treatment_name_nice": f"{treatment}",
                        "details": coal_names[res],
                        "statistic": res.statistic.item(),
                        "pvalue": res.pvalue.item(),
                    }
                )
                df_tmp = pl.concat([df_tmp, df_res])
    return df_tmp


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)
    df_mw = prepare_for_mw(df)
    df_reg = prepare_for_reg(df)
    df_ax = prepare_for_axiom_tests(df)
    df_res_ax = pl.DataFrame(
        schema={
            "axiom": str,
            "treatment_name_nice": str,
            "details": str,
            "statistic": float,
            "pvalue": float,
        }
    )

    with open(snakemake.output.summary, "w") as f:  # noqa F821 # type: ignore
        f.write("Analysis results \n")

    # Mann-Whitney U: Y=10 vs Y=90 (main analysis), Y=30 vs Y=90 (robustness check), Y=10 vs Y=30 (exploratory analysis)
    mann_whitney_results = {}
    with open(snakemake.output.summary, "a") as f:  # noqa F821 # type: ignore
        for Y1, Y2 in [(10, 90), (30, 90), (10, 30)]:
            res = compute_one_sided_mw(df_mw, Y1, Y2)
            mann_whitney_results[f"{Y1}-{Y2}"] = {
                "statistic": res.statistic.item(),
                "pvalue": res.pvalue.item(),
            }
            f.write(
                f"The p-value of the one-sided Mann-Whitney U test comparing Y = {Y1} and Y = {Y2} on the matching-group average of player A is {res.pvalue}."
            )
    with open(snakemake.output.mann_whitney, "w") as f:  # noqa F821 # type: ignore
        json.dump(mann_whitney_results, f)

    # regression (robustness check)
    reg_res = run_reg(df_reg)
    with open(snakemake.output.summary, "a") as f:  # noqa F821 # type: ignore
        f.write("\n\nRegression results: \n")
        for table in reg_res.summary().tables[:2]:
            f.write(table.as_latex_tabular())
    with open(snakemake.output.regression, "wb") as f:  # noqa F821 # type: ignore
        pickle.dump(reg_res, f)

    # regression with dummies (robustness check)
    reg_res_dummies = run_reg(df_reg, dummies=True)
    with open(snakemake.output.summary, "a") as f:  # noqa F821 # type: ignore
        f.write("\n\nRegression (dummies) results: \n")
        for table in reg_res_dummies.summary().tables[:2]:
            f.write(table.as_latex_tabular())
    with open(snakemake.output.regression_dummies, "wb") as f:  # noqa F821 # type: ignore
        pickle.dump(reg_res_dummies, f)

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

    with open(snakemake.output.summary, "a") as f:  # noqa F821 # type: ignore
        f.write(
            f"\n\n Mean squared error of the nucleolus, Shapley value, and equal split: {mse_nuc, mse_shap, mse_es}. \n"
        )

    mse_dict = {
        "nucleolus": mse_nuc,
        "shapley_value": mse_shap,
        "equal_split": mse_es,
    }
    with open(snakemake.output.mse, "w") as f:  # noqa F821 # type: ignore
        json.dump(mse_dict, f)

    # test axioms on bargaining outcomes (exploratory analysis)
    df_ax_eff = test_efficiency(df_ax, df_res_ax)
    df_ax_symm = test_symmetry(df_ax, df_res_ax)
    df_ax_stab = test_stability(df_ax, df_res_ax)
    df_res_ax = pl.concat([df_res_ax, df_ax_eff, df_ax_symm, df_ax_stab])

    with open(snakemake.output.axiom_results, "wb") as f:  # noqa F821 # type: ignore
        pickle.dump(df_res_ax.to_pandas(), f)
