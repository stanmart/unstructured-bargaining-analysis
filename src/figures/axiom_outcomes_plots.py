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

    df = df.pivot(
        index=["treatment_name_nice", "round_number", "group_id"],
        columns="id_in_group",
        values="payoff_this_round",
    )

    return df


def plot_outcomes_dummy_player(df: pl.DataFrame) -> sns.FacetGrid:
    df_plot = df.filter(pl.col("treatment_name_nice") == "Dummy player").with_columns(
        satisfies_dummy_player_axiom=(pl.col("3") == 0)
    )

    g = sns.displot(
        df_plot,
        x="treatment_name_nice",
        multiple="stack",
        hue="satisfies_dummy_player_axiom",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("", "Count")
    g.legend.set_title("Satisfies Dummy player axiom")  # type: ignore

    return g


def plot_outcomes_efficiency(df: pl.DataFrame) -> sns.FacetGrid:
    df_plot = df.with_columns(
        satisfies_efficiency=(pl.col("1") + pl.col("2") + pl.col("3") == 100)
    )

    g = sns.displot(
        df_plot,
        x="treatment_name_nice",
        multiple="stack",
        hue="satisfies_efficiency",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Count")
    g.legend.set_title("Satisfies efficiency axiom")  # type: ignore

    return g


def plot_outcomes_symmetry(df: pl.DataFrame) -> sns.FacetGrid:
    df_plot = df.with_columns(
        satisfies_symmetry=pl.when(pl.col("treatment_name_nice") == "Dummy player")
        .then((pl.col("1") == pl.col("2")))
        .otherwise((pl.col("2") == pl.col("3")))
    )

    g = sns.displot(
        df_plot,
        x="treatment_name_nice",
        multiple="stack",
        hue="satisfies_symmetry",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Count")
    g.legend.set_title("Satisfies symmetry axiom")  # type: ignore

    return g


def plot_outcomes_stability(df: pl.DataFrame) -> sns.FacetGrid:
    df_plot = df.with_columns(
        Y=df.to_series(0).str.strip_chars("Y = ").str.to_integer(strict=False)
    ).with_columns(
        satisfies_stability=pl.when(pl.col("treatment_name_nice") == "Dummy player")
        .then(pl.col("1") + pl.col("2") == 100)
        .otherwise(
            (pl.col("1") + pl.col("2") >= pl.col("Y"))
            & (pl.col("1") + pl.col("3") >= pl.col("Y"))
            & (pl.col("1") + pl.col("2") + pl.col("3") == 100)
        )
    )

    g = sns.displot(
        df_plot,
        x="treatment_name_nice",
        multiple="stack",
        hue="satisfies_stability",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Count")
    g.legend.set_title("Satisfies stability axiom")  # type: ignore

    return g


def plot_outcomes_linearity_additivity(df: pl.DataFrame) -> sns.axisgrid.FacetGrid:
    df_results = (
        df.group_by("treatment_name_nice")
        .agg(pl.col("1").mean(), pl.col("2").mean(), pl.col("3").mean())
        .filter((pl.col("treatment_name_nice") != "Dummy player"))
        .with_columns(observed=True)
    )

    theoretical = pl.DataFrame(
        {
            "treatment_name_nice": "Y = 30",
            "1": 0.75
            * df_results.filter(pl.col("treatment_name_nice") == "Y = 10")["1"]
            + 0.25 * df_results.filter(pl.col("treatment_name_nice") == "Y = 90")["1"],
            "2": 0.75
            * df_results.filter(pl.col("treatment_name_nice") == "Y = 10")["2"]
            + 0.25 * df_results.filter(pl.col("treatment_name_nice") == "Y = 90")["2"],
            "3": 0.75
            * df_results.filter(pl.col("treatment_name_nice") == "Y = 10")["3"]
            + 0.25 * df_results.filter(pl.col("treatment_name_nice") == "Y = 90")["3"],
            "observed": False,
        }
    )

    df_results = pl.concat([df_results, theoretical]).melt(
        id_vars=["treatment_name_nice", "observed"],
        variable_name="player_role",
        value_name="mean_payoff",
    )

    g = sns.FacetGrid(
        df_results.with_columns(
            treatments_int=pl.Series(
                df_results.to_series(0).str.strip_chars("Y = ").str.to_integer()
            ),
            linear=pl.when(
                (pl.col("treatment_name_nice") == "Y = 30") & (pl.col("observed"))
            )
            .then(False)
            .otherwise(True),
        ),
        col="player_role",
    )

    g.map_dataframe(
        sns.scatterplot,
        x="treatments_int",
        y="mean_payoff",
        hue="observed",
    ).add_legend(title="Observed data")

    g.map_dataframe(
        sns.lineplot,
        x="treatments_int",
        y="mean_payoff",
        hue="linear",
        palette="flare",
    )

    for ax in g.axes.flat:
        ax.set_xticks([10, 30, 90])

    g.set_titles(col_template="Player {col_name}")
    g.set_axis_labels("Treatment value Y", "Average payoff")
    g.figure.suptitle(
        "Bargaining outcomes: Agreement with the linearity axiom",
        y=1.1,
        verticalalignment="top",
    )
    return g


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    df = prepare_dataset(outcomes)

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
        funcname = "plot_outcomes_" + snakemake.wildcards.axiom  # noqa F821 # type: ignore
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.axiom}")  # noqa F821 # type: ignore

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
