import polars as pl
import seaborn.objects as so


def prepare_dataset(outcomes: pl.DataFrame) -> pl.DataFrame:
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
    )

    df = (
        outcomes.filter(
            pl.col("round_number") > 1,
        )
        .with_columns(
            role=pl.col("id_in_group").replace(roles),
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
            ),
            treatment_name_nice=pl.col("treatment_name").replace(
                {
                    "treatment_dummy_player": "Dummy player",
                    "treatment_y_10": "Y = 10",
                    "treatment_y_30": "Y = 30",
                    "treatment_y_90": "Y = 90",
                }
            ),
        )
        .join(
            values,
            on=["treatment_name", "role"],
            how="left",
        )
    )

    return df


def add_nucleolus_and_shapley(plot: so.Plot) -> so.Plot:
    return plot.add(
        so.Dot(marker="^", pointsize=6, stroke=2, color="black"),
        so.Dodge(),
        y="nucleolus",
        legend=False,
    ).add(
        so.Dot(marker="_", pointsize=12, stroke=2, color="black"),
        so.Dodge(),
        y="shapley_value",
        legend=False,
    )


def payoff_average(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(df, x="treatment_name_nice", color="role")
        .add(
            so.Bar(),
            so.Agg(),
            so.Dodge(),
            y="payoff_this_round",
            label="Payoff",
        )
        .label(x="Treatment", y="Average payoff", color="Player")
    )
    return add_nucleolus_and_shapley(plot)


def payoff_scatterplot(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(df, x="treatment_name_nice", color="role")
        .add(
            so.Dot(alpha=0.5),
            so.Jitter(),
            so.Dodge(),
            y="payoff_this_round",
        )
        .scale(color=so.Nominal())
        .label(x="Treatment", y="Payoff", color="Player")
    )
    return add_nucleolus_and_shapley(plot)


def payoff_by_agreement_type(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(
            df.filter(pl.col("role") == "P1"),
            x="treatment_name_nice",
        )
        .add(so.Dot(alpha=0.5), so.Jitter(), y="payoff_this_round", color="agreement")
        .label(x="Treatment", y="P1's payoff", color="Coordination outcome")
    )
    return add_nucleolus_and_shapley(plot)


def payoff_share_of_agreement_types(df: pl.DataFrame) -> so.Plot:
    return (
        so.Plot(
            df.filter(pl.col("role") == "P1"),
            x="treatment_name_nice",
            color="agreement",
        )
        .add(so.Bar(), so.Count(), so.Stack())
        .label(x="Treatment", y="Count", color="Coordination outcome")
    )


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)
    width = float(snakemake.wildcards.width)  # noqa F821 # type: ignore
    height = float(snakemake.wildcards.height)  # noqa F821 # type: ignore

    if snakemake.wildcards.plot == "scatterplot":  # noqa F821 # type: ignore
        plot = payoff_scatterplot(df)
    elif snakemake.wildcards.plot == "average":  # noqa F821 # type: ignore
        plot = payoff_average(df)
    elif snakemake.wildcards.plot == "by_agreement_type":  # noqa F821 # type: ignore
        plot = payoff_by_agreement_type(df)
    elif snakemake.wildcards.plot == "share_of_agreement_types":  # noqa F821 # type: ignore
        plot = payoff_share_of_agreement_types(df)

    plot.layout(size=(width, height)).save(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
