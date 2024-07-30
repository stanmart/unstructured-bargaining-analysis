import numpy as np
import polars as pl
import seaborn as sns


def prepare_outcomes(outcomes: pl.DataFrame) -> pl.DataFrame:
    df = outcomes.filter(pl.col("round_number") == 6).with_columns(
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


def prepare_personal(personal: pl.DataFrame) -> pl.DataFrame:
    df = personal.with_columns(
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


def plot_gender(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.displot(
        df,
        x="treatment_name_nice",
        multiple="stack",
        hue="gender",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Count")
    g.legend.set_title("Gender")  # type: ignore

    return g


def plot_age(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.catplot(
        data=df,
        kind="bar",
        x="treatment_name_nice",
        y="age",
        alpha=0.9,
        errorbar=None,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Average age")
    return g


def plot_study_field(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.FacetGrid(
        df,
        height=6,
        aspect=1,
        col="treatment_name_nice",
    )

    study_fields = list(df["study_field"].unique())
    palette = sns.color_palette("hls", len(study_fields))
    mapping = {cat: color for cat, color in zip(study_fields, palette)}

    g.map_dataframe(
        sns.histplot,
        x="study_field",
        stat="count",
        alpha=1,
        hue="study_field",
        palette=mapping,
    )

    line_positions = np.arange(0, 12)
    for ax in g.axes.flat:
        ax.set_yticks(line_positions)
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", labelrotation=90, labelsize=13)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("", "Count")

    return g


def plot_nationality(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.FacetGrid(
        df,
        height=5,
        aspect=1,
        col="treatment_name_nice",
        col_wrap=2,
    )

    nationalities = list(df["nationality"].unique())
    palette = sns.color_palette("hls", len(nationalities))
    mapping = {cat: color for cat, color in zip(nationalities, palette)}

    g.map_dataframe(
        sns.histplot,
        x="nationality",
        stat="count",
        alpha=1,
        hue="nationality",
        palette=mapping,
    )

    line_positions = np.arange(0, 18)
    for ax in g.axes.flat:
        ax.set_yticks(line_positions)
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", labelrotation=90, labelsize=9)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("", "Count")
    g.tick_params(axis="x", labelbottom=True)
    g.figure.tight_layout()

    return g


def plot_degree(df: pl.DataFrame) -> sns.FacetGrid:
    g = sns.displot(
        df,
        x="treatment_name_nice",
        multiple="stack",
        hue="degree",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Count")
    g.legend.set_title("Degree")  # type: ignore

    return g


def plot_difficulty_rating(df: pl.DataFrame) -> sns.axisgrid.FacetGrid:
    order = [
        "Very difficult",
        "Difficult",
        "Medium difficulty",
        "Easy",
        "Very easy",
        "No opinion",
    ]
    order_diff_dtype = pl.Enum(order)

    palette = sns.color_palette("Spectral", n_colors=5) + [(0.5, 0.5, 0.5)]
    mapping = {cat: color for cat, color in zip(order, palette)}

    g = sns.FacetGrid(
        df.with_columns(
            difficulty_ordered=pl.Series(
                df.select(pl.col("difficulty")), dtype=order_diff_dtype
            )
        ),
        col="treatment_name_nice",
    )
    g.map_dataframe(
        sns.histplot,
        x="difficulty_ordered",
        stat="count",
        hue="difficulty_ordered",
        alpha=0.9,
        palette=mapping,
    )

    line_positions = np.arange(0, 25, 5)
    for ax in g.axes.flat:
        for pos in line_positions:
            ax.set_yticks(line_positions)
            ax.xaxis.grid(False)
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
    personal_variables = [
        "age",
        "gender",
        "gender_other",
        "degree",
        "degree_other",
        "study_field",
        "study_field_other",
        "nationality",
        "has_second_natinality",
        "second_nationality",
    ]
    variable = snakemake.wildcards.plot  # noqa F821 # type: ignore
    if variable in personal_variables:
        personal = pl.read_csv(snakemake.input.personal)  # noqa F821 # type: ignore
        df = prepare_personal(personal)
    else:
        personal = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
        df = prepare_outcomes(personal)

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
        funcname = "plot_" + variable
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {variable}")

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
