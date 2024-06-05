import polars as pl
import seaborn as sns


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


def proposal_number_per_round(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.FacetGrid(
        df.group_by(["treatment_name_nice", "round_number_corrected", "group_id"]).agg(
            pl.len().alias("n")
        ),
        col="treatment_name_nice",
    )
    g.map_dataframe(
        sns.barplot,
        x="round_number_corrected",
        y="n",
        estimator="mean",
        errorbar=None,
        alpha=0.9,
    )
    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("Round", "Average number of proposals")

    return g


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    df = prepare_dataset(actions)
    sns.set_style(
        "whitegrid",
        {
            "grid.color": "grey",
            "grid.linestyle": "-",
            "grid.linewidth": 0.25,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
        },
    )

    try:
        funcname = "proposal_" + snakemake.wildcards.plot  # noqa F821 # type: ignore
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
