import itertools
import textwrap

import matplotlib.pyplot as plt
import polars as pl


def prepare_dataset(actions: pl.DataFrame) -> pl.DataFrame:
    actions_clean = actions.with_columns(
        player_name=pl.when(
            pl.col("treatment_name") == "treatment_dummy_player",
        )
        .then(
            pl.col("id_in_group").replace(
                {
                    1: "A1",
                    2: "A2",
                    3: "B",
                }
            )
        )
        .otherwise(
            pl.col("id_in_group").replace(
                {
                    1: "A",
                    2: "B1",
                    3: "B2",
                }
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

    return actions_clean


def plot_chat(actions: pl.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 4.8))
    ax.set_xlim(0, 1)

    yloc = 0

    alignment = {
        1: "left",
        2: "center",
        3: "right",
    }
    horizontal_location = {
        1: 0.0,
        2: 0.5,
        3: 1.0,
    }
    colors = {
        1: "tab:blue",
        2: "tab:red",
        3: "tab:green",
    }

    for row in reversed(actions.rows(named=True)):
        id = row["id_in_group"]
        players = (
            ["A", "B1", "B2"]
            if row["treatment_name"] != "treatment_dummy_player"
            else ["A1", "A2", "B"]
        )

        if row["action"] == "chat":
            content = textwrap.fill(row["body"], width=30)
        elif row["action"] == "proposal":
            content = f"Proposal #{row['offer_id']}:\n" + ", ".join(
                f"{player}: {row['allocation_' + id]}"
                for player, id in zip(players, ["1", "2", "3"])
            )
        elif row["action"] == "acceptance":
            content = f"Accept #{row['accepted_offer']}"
        else:
            raise ValueError(f"Unknown action: {row['action']}")

        text = ax.text(
            x=horizontal_location[id],
            y=yloc,
            s=content,
            horizontalalignment=alignment[id],
            verticalalignment="bottom",
        )

        bboxstyle = "round,pad=0.5" if row["action"] == "chat" else "sawtooth,pad=0.5"

        text.set_bbox(
            dict(
                facecolor=colors[id],
                alpha=0.5,
                edgecolor=colors[id],
                boxstyle=bboxstyle,
            )
        )
        top_left_corner = (
            text.get_tightbbox().transformed(ax.transData.inverted()).corners()[1, :]
        )
        player_label = ax.text(
            x=top_left_corner[0] - 0.01,
            y=top_left_corner[1] + 0.02,
            s=row["player_name"],
            horizontalalignment="left",
            verticalalignment="bottom",
            fontweight="bold",
            color=colors[id],
        )
        yloc = (
            player_label.get_tightbbox()
            .transformed(ax.transData.inverted())
            .corners()[:, 1]
            .max()
            + 0.05
        )

    ax.axis("off")

    return fig


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    row_string: str = snakemake.wildcards.rows  # noqa F821 # type: ignore

    rows = itertools.chain.from_iterable(
        [int(row_string_section)]
        if "-" not in row_string_section
        else range(
            int(row_string_section.split("-")[0]),
            int(row_string_section.split("-")[1]) + 1,
        )
        for row_string_section in row_string.split(",")
    )
    rows_corrected = [row - 2 for row in rows]

    df = prepare_dataset(actions[rows_corrected])
    fig = plot_chat(df)

    fig.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
