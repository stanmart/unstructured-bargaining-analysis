import polars as pl
import seaborn.objects as so


def prepare_dataset(actions: pl.DataFrame) -> pl.DataFrame:
    treatment_names = pl.Enum(["Dummy player", "Y = 10", "Y = 30", "Y = 90"])

    proposals = actions.filter(
        pl.col("round_number") > 1, pl.col("action") == "proposal"
    ).with_columns(
        treatment_name_nice=pl.col("treatment_name")
        .replace(
            {
                "treatment_dummy_player": "Dummy player",
                "treatment_y_10": "Y = 10",
                "treatment_y_30": "Y = 30",
                "treatment_y_90": "Y = 90",
            }
        )
        .cast(treatment_names),
        round_number_nice=pl.format("Round {}", pl.col("round_number") - 1),
        round_number_corrected=pl.col("round_number") - 1,
    )

    return proposals


def proposal_number_per_round(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(
            df.group_by(
                ["treatment_name_nice", "round_number_corrected", "group_id"]
            ).agg(pl.len().alias("n"))
        )
        .add(
            so.Bar(),
            so.Agg("mean"),
            y="n",
            x="round_number_corrected",
        )
        .label(x="Round", y="Average number of proposals")
        .facet(col="treatment_name_nice")
        .scale(x=so.Nominal(order=[i + 1 for i in range(5)]))
    )
    return plot


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    df = prepare_dataset(actions)
    width = float(snakemake.wildcards.width)  # noqa F821 # type: ignore
    height = float(snakemake.wildcards.height)  # noqa F821 # type: ignore

    try:
        funcname = "proposal_" + snakemake.wildcards.plot  # noqa F821 # type: ignore
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore

    plot.layout(size=(width, height)).save(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
