import matplotlib.pyplot as plt
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


def proposal_gini(df: pl.DataFrame) -> sns.FacetGrid:
    df_gini = (
        df.with_columns(
            gini=(
                (pl.col("allocation_1") - pl.col("allocation_2")).abs()
                + (pl.col("allocation_1") - pl.col("allocation_3")).abs()
                + (pl.col("allocation_2") - pl.col("allocation_3")).abs()
            )
            / (
                3
                * (
                    pl.col("allocation_1")
                    + pl.col("allocation_2")
                    + pl.col("allocation_3")
                )
            ),
            is_A=pl.when(pl.col("treatment_name_nice") != "Dummy player")
            .then(pl.col("id_in_group") == 1)
            .otherwise(pl.col("id_in_group") != 3),
        )
        .group_by(["treatment_name_nice", "participant_code", "is_A"])
        .agg(avg_gini=pl.col("gini").mean())
        .pivot(
            columns="is_A",
            index=["treatment_name_nice", "participant_code"],
            values="avg_gini",
        )
        .fill_null(-0.1)
        .fill_nan(-0.1)
    )

    g = sns.FacetGrid(
        data=df_gini.rename({"true": "gini_avg_A", "false": "gini_avg_B"}).with_columns(
            incomplete=(pl.col("gini_avg_A") == -0.1) | (pl.col("gini_avg_B") == -0.1),
        ),
        col="treatment_name_nice",
        col_wrap=2,
    )
    g.map_dataframe(
        sns.scatterplot,
        x="gini_avg_B",
        y="gini_avg_A",
        style="incomplete",
        alpha=0.5,
    )

    def const_line(*args, **kwargs):
        x = [0, 0.43]
        y = [0, 0.43]
        plt.plot(y, x, color="Red")

    g.map(const_line)
    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels(
        "Average gini coefficient\nof proposals as B",
        "Average gini coefficient\nof proposals as A",
        # fontsize=8,
    )

    g.set(
        xticks=[-0.1, 0.0, 0.2, 0.4],
        xticklabels=["N/A", "0.0", "0.2", "0.4"],
        yticks=[-0.1, 0.0, 0.2, 0.4],
        yticklabels=["N/A", "0.0", "0.2", "0.4"],
    )

    return g


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    df = prepare_dataset(actions)
    sns.set_style(
        "whitegrid",
        {
            "axes.edgecolor": "black",
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
