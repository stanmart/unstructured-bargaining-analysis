import numpy as np
import polars as pl
import seaborn as sns


def prepare_dataset(outcomes: pl.DataFrame) -> pl.DataFrame:
    df = outcomes.filter(pl.col("round_number") > 1).with_columns(
        treatment_name_nice=pl.col("treatment_name").replace(
            {
                "treatment_dummy_player": "Dummy player",
                "treatment_y_10": "Y = 10",
                "treatment_y_30": "Y = 30",
                "treatment_y_90": "Y = 90",
            }
        )
    )
    return df


def plot_difficulty_rating(df: pl.DataFrame) -> sns.axisgrid.FacetGrid:
    order = [
        "Easy",
        "Very easy",
        "Medium difficulty",
        "Difficult",
        "Very difficult",
        "No opinion",
    ]
    order_diff_dtype = pl.Enum(order)

    df_diff = df.filter(pl.col("round_number") == 6)

    g = sns.FacetGrid(
        df_diff.with_columns(
            difficulty_ordered=pl.Series(
                df_diff.select(pl.col("difficulty")), dtype=order_diff_dtype
            )
        ),
        col="treatment_name_nice",
    )
    g.map_dataframe(
        sns.histplot,
        x="difficulty_ordered",
        stat="count",
        hue="difficulty_ordered",
    )

    line_positions = np.arange(0, 25, 5)
    for ax in g.axes.flat:
        for pos in line_positions:
            ax.axhline(y=pos, color="grey", linestyle="-", linewidth=0.25)
        ax.tick_params(axis="x", labelrotation=90)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("", "Count")
    g.figure.suptitle(
        '"How would you rate the difficulty level of the game?"',
        y=1.1,
        verticalalignment="top",
    )
    return g


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)

    try:
        funcname = "plot_" + snakemake.wildcards.plot  # noqa F821 # type: ignore
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
