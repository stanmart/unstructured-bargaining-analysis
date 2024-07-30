import ast
import warnings

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def prepare_dataset(outcomes: pl.DataFrame) -> pl.DataFrame:
    treatment_names = pl.Enum(["Dummy player", "Y = 10", "Y = 30", "Y = 90"])
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
        outcomes.filter(
            pl.col("round_number") > 1,
        )
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
            round_number_corrected=pl.col("round_number") - 1,
            matching_group=(pl.col("participant_id") - 1) // 6,
        )
        .join(
            values,
            on=["treatment_name", "role"],
            how="left",
        )
    )

    return df


def add_nucleolus_and_shapley(
    df: pl.DataFrame, ax: Axes, hue: str = "role", dodge: bool = True
) -> Axes:
    values = df.select(
        "treatment_name_nice",
        "nucleolus",
        "shapley_value",
        hue,
    ).unique()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sns.stripplot(
            values,
            x="treatment_name_nice",
            y="nucleolus",
            hue=hue,
            palette=["black"],
            linewidth=2,
            marker="1",
            s=10,
            ax=ax,
            dodge=dodge,
            jitter=False,
            legend=False,
        )
        sns.stripplot(
            values,
            x="treatment_name_nice",
            y="shapley_value",
            hue=hue,
            palette=["black"],
            linewidth=2,
            marker="2",
            s=10,
            ax=ax,
            dodge=dodge,
            jitter=False,
            legend=False,
        )

    nucleolus_marker = Line2D(
        [0],
        [0],
        color="black",
        markersize=10,
        markeredgewidth=2,
        marker="1",
        linestyle="None",
        label="Nucleolus",
    )
    shapley_marker = Line2D(
        [0],
        [0],
        color="black",
        markersize=10,
        markeredgewidth=2,
        marker="2",
        linestyle="None",
        label="Shapley value",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, nucleolus_marker, shapley_marker])

    return ax


def payoff_average(
    df: pl.DataFrame, round_numbers: list[int] = [2, 3, 4, 5, 6]
) -> Figure:
    fig, ax = plt.subplots()
    sns.barplot(
        df.filter(pl.col("round_number").is_in(round_numbers)).with_columns(
            role=pl.col("role").replace(
                {
                    "P1": "$A_1$ / $A$",
                    "P2": "$A_2$ / $B_1$",
                    "P3": "$B$ / $B_2$",
                }
            )
        ),
        x="treatment_name_nice",
        y="payoff_this_round",
        hue="role",
        alpha=0.9,
        ax=ax,
        errorbar="ci",
        dodge=True,
    )

    for line in ax.get_lines():
        line.set_xdata(line.get_xdata() + 0.07)

    add_nucleolus_and_shapley(df, ax)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Average payoff")

    return fig


def payoff_scatterplot(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    sns.stripplot(
        df.with_columns(
            role=pl.col("role").replace(
                {
                    "P1": "$A_1$ / $A$",
                    "P2": "$A_2$ / $B_1$",
                    "P3": "$B$ / $B_2$",
                }
            )
        ),
        x="treatment_name_nice",
        y="payoff_this_round",
        hue="role",
        alpha=0.5,
        ax=ax,
        dodge=True,
        jitter=0.25,
    )

    add_nucleolus_and_shapley(df, ax)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Payoff")

    return fig


def payoff_by_agreement_type(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    df_p1 = df.filter(pl.col("role") == "P1")
    sns.stripplot(
        df_p1,
        x="treatment_name_nice",
        y="payoff_this_round",
        hue="agreement",
        alpha=0.5,
        ax=ax,
        dodge=False,
        jitter=0.1,
    )

    add_nucleolus_and_shapley(df_p1, ax, hue="agreement", dodge=False)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("P1's Payoff")
    ax.legend(title="Coordination outcome")

    return fig


def payoff_share_of_agreement_types(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    sns.countplot(
        df.filter(pl.col("role") == "P1"),
        x="treatment_name_nice",
        hue="agreement",
        ax=ax,
        alpha=0.9,
    )

    ax.set_xlabel("Treatment")
    ax.set_ylabel("Count")
    ax.legend(title="Coordination outcome")

    return fig


def payoff_share_of_agreement_types_by_round(df: pl.DataFrame) -> sns.FacetGrid:
    round_numbers = pl.Enum(["1", "2", "3", "4", "5"])

    g = sns.displot(
        df.filter(pl.col("role") == "P1").with_columns(
            round_number_corrected=pl.col("round_number_corrected")
            .cast(pl.String)
            .cast(round_numbers)
        ),
        x="round_number_corrected",
        hue="agreement",
        multiple="stack",
        col="treatment_name_nice",
        alpha=0.9,
    )

    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("Round", "Count")
    g.legend.set_title("Coordination outcome")  # type: ignore

    return g


def payoff_equal_splits_by_round(df: pl.DataFrame) -> sns.FacetGrid:
    round_numbers = pl.Enum(["1", "2", "3", "4", "5"])
    df_plot = (
        df.group_by(["treatment_name_nice", "round_number_corrected", "group_id"])
        .agg(
            min_payoff=pl.col("payoff_this_round").min(),
            max_payoff=pl.col("payoff_this_round").max(),
        )
        .with_columns(
            equal_split=pl.col("max_payoff") - pl.col("min_payoff") <= 1,
            round_number_corrected=pl.col("round_number_corrected")
            .cast(pl.String)
            .cast(round_numbers),
        )
        .with_columns(
            split_type=pl.when(pl.col("equal_split"))
            .then(pl.lit("Equal split"))
            .otherwise(pl.lit("Unequal split"))
        )
    )

    g = sns.displot(
        df_plot,
        x="round_number_corrected",
        hue="split_type",
        multiple="stack",
        col="treatment_name_nice",
        alpha=0.9,
    )
    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("Round", "Count")
    g.legend.set_title("Coordination outcome")  # type: ignore

    return g


def payoff_matching_group_average(df: pl.DataFrame) -> sns.axisgrid.FacetGrid:
    role_order = ["$A_1$ / $A$", "$A_2$ / $B_1$", "$B$ / $B_2$"]
    role_dtype = pl.Enum(role_order)

    df_plot = (
        df.group_by(["treatment_name_nice", "matching_group", "role"])
        .mean()
        .with_columns(
            role=pl.col("role")
            .replace(
                {
                    "P1": "$A_1$ / $A$",
                    "P2": "$A_2$ / $B_1$",
                    "P3": "$B$ / $B_2$",
                }
            )
            .cast(role_dtype)
        )
    )

    g = sns.FacetGrid(
        df_plot,
        col="treatment_name_nice",
        height=3,
        aspect=1,
        col_wrap=2,
    )

    g.map_dataframe(
        sns.scatterplot,
        x="role",
        y="payoff_this_round",
        hue="matching_group",
        alpha=1,
        palette="deep",
    ).add_legend(title="Matching group")

    g.map_dataframe(
        sns.lineplot,
        x="role",
        y="payoff_this_round",
        hue="matching_group",
        alpha=1,
        palette="deep",
        linestyle="dashed",
    )

    g.map_dataframe(
        sns.scatterplot, x="role", y="nucleolus", color="black", linewidth=2, marker="1"
    )

    g.map_dataframe(
        sns.scatterplot,
        x="role",
        y="shapley_value",
        color="black",
        linewidth=2,
        marker="2",
    )

    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_titles(col_template="{col_name} treatment")
    g.set_axis_labels("Player role", "Average payoff")
    return g


def payoff_groups(df: pl.DataFrame) -> sns.FacetGrid:
    outcome_groups = df.pivot(
        index=["treatment_name_nice", "round_number", "group_id", "agreement"],
        columns="id_in_group",
        values="payoff_this_round",
    ).with_columns(
        group=pl.when(pl.col("agreement") == "Breakdown")
        .then(pl.lit("Coordination failure"))
        .when(pl.col("agreement") == "Partial agreement")
        .then(pl.lit("Partial agreement"))
        .when((pl.col("1") == 33) & (pl.col("2") == 33) & (pl.col("3") == 33))
        .then(pl.lit("Equal split"))
        .when(
            (pl.col("1").is_in([33, 34]))
            & (pl.col("2").is_in([33, 34]))
            & (pl.col("3").is_in([33, 34]))
        )
        .then(pl.lit("Almost-equal split"))
        .when((pl.col("1") == pl.col("2")) & (pl.col("1") > pl.col("3")))
        .then(pl.lit("Grand coalition: 1 = 2 > 3"))
        .when((pl.col("1") > pl.col("2")) & (pl.col("2") == pl.col("3")))
        .then(
            pl.lit(
                "Grand coalition: 1 > 2 = 3"
            )  # note that this is a slight misnomer as this excludes almost-equal splits
        )
        .otherwise(pl.lit("Grand coalition: other"))
    )

    g = sns.displot(
        outcome_groups,
        x="treatment_name_nice",
        hue="group",
        multiple="stack",
        alpha=0.9,
    )

    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_axis_labels("Treatment", "Cluster size")
    g.legend.set_title("Outcome clusters")  # type: ignore

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
        funcname = "payoff_" + snakemake.wildcards.plot  # noqa F821 # type: ignore
        if snakemake.wildcards.plot == "average":  # noqa F821 # type: ignore
            round_numbers = (
                ast.literal_eval(snakemake.wildcards.rounds)  # noqa F821 # type: ignore
                if snakemake.wildcards.rounds != "all"  # noqa F821 # type: ignore
                else [2, 3, 4, 5, 6]
            )
            plot = payoff_average(df, round_numbers)  # noqa F821 # type: ignore
        else:
            plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
