from math import ceil
from typing import Literal

import matplotlib.pyplot as plt
import mpltern
import numpy as np
import polars as pl
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def prepare_dataset(
    input_df: pl.DataFrame, type: Literal["actions"] | Literal["outcomes"]
) -> pl.DataFrame:
    treatments = pl.Enum(
        [
            "Dummy player",
            "Y = 10",
            "Y = 30",
            "Y = 90",
        ]
    )

    values = pl.from_dicts(
        [
            {
                "treatment_name": "treatment_dummy_player",
                "shap_1": 50.0 / 100,
                "shap_2": 50.0 / 100,
                "shap_3": 0.0 / 100,
                "nuc_1": 50.0 / 100,
                "nuc_2": 50.0 / 100,
                "nuc_3": 0.0 / 100,
            },
            {
                "treatment_name": "treatment_y_10",
                "role": "P1",
                "shap_1": 110.0 / 300,
                "shap_2": 95.0 / 300,
                "shap_3": 95.0 / 300,
                "nuc_1": 100.0 / 300,
                "nuc_2": 100.0 / 300,
                "nuc_3": 100.0 / 300,
            },
            {
                "treatment_name": "treatment_y_30",
                "role": "P1",
                "shap_1": 130.0 / 300,
                "shap_2": 85.0 / 300,
                "shap_3": 85.0 / 300,
                "nuc_1": 100.0 / 300,
                "nuc_2": 100.0 / 300,
                "nuc_3": 100.0 / 300,
            },
            {
                "treatment_name": "treatment_y_90",
                "shap_1": 190.0 / 300,
                "shap_2": 55.0 / 300,
                "shap_3": 55.0 / 300,
                "nuc_1": 270.0 / 300,
                "nuc_2": 15.0 / 300,
                "nuc_3": 15.0 / 300,
            },
        ]
    )

    if type == "outcomes":
        df = (
            input_df.select(
                [
                    "session_code",
                    "treatment_name",
                    "round_number",
                    "group_id",
                    "id_in_group",
                    "payoff_this_round",
                ]
            )
            .pivot(
                index=["session_code", "treatment_name", "round_number", "group_id"],
                values=["payoff_this_round"],
                columns="id_in_group",
                aggregate_function=None,
            )
            .rename(
                {
                    "1": "P1",
                    "2": "P2",
                    "3": "P3",
                }
            )
        )

    elif type == "actions":
        df = (
            input_df.filter(pl.col("action") == "proposal")
            .select(
                [
                    "session_code",
                    "treatment_name",
                    "round_number",
                    "group_id",
                    "id_in_group",
                    "participant_code",
                    "allocation_1",
                    "allocation_2",
                    "allocation_3",
                ]
            )
            .rename(
                {
                    "allocation_1": "P1",
                    "allocation_2": "P2",
                    "allocation_3": "P3",
                }
            )
        )

    else:
        raise ValueError(f"Unknown type: {type}")

    df = (
        df.with_columns(total=pl.col("P1") + pl.col("P2") + pl.col("P3"))
        .with_columns(
            prop_1=pl.col("P1") / pl.col("total"),
            prop_2=pl.col("P2") / pl.col("total"),
            prop_3=pl.col("P3") / pl.col("total"),
        )
        .with_columns(
            treatment_name_nice=pl.col("treatment_name")
            .replace(
                {
                    "treatment_dummy_player": "Dummy player",
                    "treatment_y_10": "Y = 10",
                    "treatment_y_30": "Y = 30",
                    "treatment_y_90": "Y = 90",
                }
            )
            .cast(treatments)
        )
        .join(
            values,
            on="treatment_name",
            how="left",
        )
    )

    return df.filter(pl.col("round_number") > 1)


def plot_allocations(df: pl.DataFrame, colors_var: str = "total") -> Figure:
    treatments_in_data: list[str] = df["treatment_name_nice"].unique().sort().to_list()
    fig, axes = plt.subplots(
        ceil(len(treatments_in_data) / 2), 2, subplot_kw={"projection": "ternary"}
    )
    fig.subplots_adjust(wspace=0.3, hspace=0.6)

    viridis = colormaps["viridis"]

    if colors_var == "total":
        color_bin_limits = {
            "[0, 70)": 0,
            "[70, 80)": 70,
            "[80, 90)": 80,
            "[90, 99)": 90,
            "99": 99,
            "100": 100,
        }
        color_bin_rename = {
            "[99, 100)": "99",
            "[100, inf)": "100",
        }
        colormap = {
            category: to_hex(viridis(i / len(color_bin_limits)))
            for i, category in enumerate(color_bin_limits.keys())
        }
    elif colors_var.endswith("axiom"):
        categories = [
            "Strongly Disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly Agree",
            "No opinion",
        ]
        palette = colormaps["Spectral"](np.linspace(0, 1, len(categories)))
        colormap = {
            category: to_hex(color) for category, color in zip(categories, palette)
        }
        color_bin_limits = None
        color_bin_rename = {}
    else:
        raise ValueError(f"No colormap for {colors_var}")

    for ax, treatment in zip(axes.flatten(), treatments_in_data):
        plot_treatment(
            ax=ax,
            df=df,
            treatment=treatment,
            color_var=colors_var,
            color_bin_limits=color_bin_limits,
            colormap=colormap,
            color_bin_rename=color_bin_rename,
        )

    fig.set_size_inches(8, 8)

    point_handles = [
        Line2D(
            [0],
            [0],
            markeredgecolor=color,
            marker="o",
            linestyle="None",
            markerfacecolor="None",
            label=label,
        )
        for label, color in colormap.items()
    ]

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
    legend = fig.legend(handles=point_handles + [nucleolus_marker, shapley_marker])
    legend.set_title("Total points\nallocated")

    return fig


def plot_treatment(
    ax: mpltern.TernaryAxes,
    df: pl.DataFrame,
    treatment: str,
    colormap: dict[str, str],
    color_var: str = "total",
    color_bin_limits: dict[str, int] | None = None,
    color_bin_rename: dict[str, str] = {},
) -> None:
    df_i = df.filter(pl.col("treatment_name_nice") == treatment, pl.col("total") > 0)

    if treatment == "Dummy player":
        ax.set_tlabel("Share of B")
        ax.set_llabel("Share of A1")
        ax.set_rlabel("Share of A2")

        top = df_i["prop_3"]
        left = df_i["prop_1"]
        right = df_i["prop_2"]

        shap_t = df_i["shap_3"][0]
        shap_l = df_i["shap_1"][0]
        shap_r = df_i["shap_2"][0]

        nuc_t = df_i["nuc_3"][0]
        nuc_l = df_i["nuc_1"][0]
        nuc_r = df_i["nuc_2"][0]

    else:
        ax.set_tlabel("Share of A")
        ax.set_llabel("Share of B1")
        ax.set_rlabel("Share of B2")

        top = df_i["prop_1"]
        left = df_i["prop_2"]
        right = df_i["prop_3"]

        shap_t = df_i["shap_1"][0]
        shap_l = df_i["shap_2"][0]
        shap_r = df_i["shap_3"][0]

        nuc_t = df_i["nuc_1"][0]
        nuc_l = df_i["nuc_2"][0]
        nuc_r = df_i["nuc_3"][0]

    top_jitter = jitter_series(top, type="normal", spread=0.02)
    left_jitter = jitter_series(left, type="normal", spread=0.02)
    right_jitter = jitter_series(right, type="normal", spread=0.02)

    if color_bin_limits:
        color_bins = (
            df_i[color_var]
            .cut(list(color_bin_limits.values()), left_closed=True)
            .cast(pl.String)
            .replace(color_bin_rename)
        )
        colors = color_bins.replace(colormap)
    else:
        colors = df[color_var].replace(colormap)

    ax.grid(alpha=0.5)

    ax.scatter(
        top_jitter,
        left_jitter,
        right_jitter,
        c="None",
        edgecolors=colors,
        s=10,  # type: ignore
        alpha=0.8,
    )

    ax.scatter(
        shap_t,
        shap_l,
        shap_r,
        linewidth=2,
        marker="1",
        s=50,  # type: ignore
        color="black",
    )

    ax.scatter(
        nuc_t,
        nuc_l,
        nuc_r,
        linewidth=2,
        marker="2",
        s=50,  # type: ignore
        color="black",
    )

    num_ticks = 7
    ax.taxis.set_ticks([i / 6 for i in range(num_ticks + 1)])
    ax.laxis.set_ticks([i / 6 for i in range(num_ticks + 1)])
    ax.raxis.set_ticks([i / 6 for i in range(num_ticks + 1)])
    ax.taxis.set_ticklabels(
        ["0"] + [f"{i} / {num_ticks - 1}" for i in range(1, num_ticks)] + ["1"]
    )
    ax.laxis.set_ticklabels(
        ["0"] + [f"{i} / {num_ticks - 1}" for i in range(1, num_ticks)] + ["1"]
    )
    ax.raxis.set_ticklabels(
        ["0"] + [f"{i} / {num_ticks - 1}" for i in range(1, num_ticks)] + ["1"]
    )
    ax.taxis.set_label_position("tick1")  # type: ignore
    ax.laxis.set_label_position("tick1")  # type: ignore
    ax.raxis.set_label_position("tick1")  # type: ignore

    ax.set_title(treatment)


def jitter_series(
    series: pl.Series, type: Literal["normal"] | Literal["uniform"], spread: float
) -> pl.Series:
    if type == "normal":
        noise = pl.Series(np.random.normal(0, spread, len(series)))
    elif type == "uniform":
        noise = series + pl.Series(np.random.uniform(-spread, spread, len(series)))
    else:
        raise ValueError(f"Unknown type: {type}")

    return (series + noise).clip(0.0, 1.0)


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

    if snakemake.wildcards.type == "outcomes":  # noqa F821 # type: ignore
        df = prepare_dataset(outcomes, type="outcomes")
        colors_var = "total"
    elif snakemake.wildcards.type.startswith("proposals"):  # noqa F821 # type: ignore
        df = prepare_dataset(actions, type="actions")
        if snakemake.wildcards.type.endswith("axiom"):  # noqa F821 # type: ignore
            colors_var = snakemake.wildcards.type.split("_by_")[-1]  # noqa F821 # type: ignore
            survey = outcomes.select("participant_code", colors_var).unique()
            df = df.join(survey, on="participant_code", how="left")
        else:
            colors_var = "total"
    else:
        raise ValueError(f"Unknown type: {snakemake.wildcards.type}")  # noqa F821 # type: ignore

    plot = plot_allocations(df, colors_var=colors_var)
    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
