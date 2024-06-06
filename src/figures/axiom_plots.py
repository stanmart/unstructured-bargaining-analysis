import numpy as np
import polars as pl
import seaborn as sns


def prepare_dataset(outcomes: pl.DataFrame) -> pl.DataFrame:
    df = outcomes.filter(pl.col("round_number") == 6).with_columns(
        treatment_name_nice=pl.col("treatment_name").replace(
            {
                "treatment_dummy_player": "Dummy player",
                "treatment_y_10": "Y = 10",
                "treatment_y_30": "Y = 30",
                "treatment_y_90": "Y = 90",
            }
        ),
    )
    return df


def plot_axiom_survey(
    df: pl.DataFrame, axiom: str, axiom_nice: str
) -> sns.axisgrid.FacetGrid:
    order = [
        "Strongly Disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly Agree",
        "No opinion",
    ]
    order_axiom_dtype = pl.Enum(order)

    palette = sns.color_palette("Spectral", n_colors=5) + [(0.5, 0.5, 0.5)]
    mapping = {cat: color for cat, color in zip(order, palette)}

    g = sns.FacetGrid(
        df.with_columns(axiom_ordered=pl.col(f"{axiom}_axiom").cast(order_axiom_dtype)),
        col="treatment_name_nice",
    )
    g.map_dataframe(
        sns.histplot,
        x="axiom_ordered",
        stat="count",
        hue="axiom_ordered",
        alpha=0.9,
        palette=mapping,
    )

    line_positions = np.arange(0, 25, 5)
    for ax in g.axes.flat:
        ax.set_yticks(line_positions)
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", labelrotation=90)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("", "Count")
    g.figure.suptitle(
        f"Survey: Agreement with the {axiom_nice} axiom", y=1.1, verticalalignment="top"
    )

    return g


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)
    axioms_renamed = {
        "efficiency": "Efficiency",
        "symmetry": "Symmetry",
        "dummy_player": "Dummy Player",
        "linearity_HD1": "Linearity (HD1)",
        "linearity_additivity": "Linearity (Additivity)",
        "stability": "Stability",
    }

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
        plot = plot_axiom_survey(
            df=df,
            axiom=snakemake.wildcards.axiom,  # noqa F821 # type: ignore
            axiom_nice=axioms_renamed[snakemake.wildcards.axiom],  # noqa F821 # type: ignore
        )
        plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore
