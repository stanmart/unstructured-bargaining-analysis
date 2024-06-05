import polars as pl
import seaborn.objects as so
from matplotlib.ticker import FixedFormatter


def prepare_dataset(actions: pl.DataFrame, outcomes: pl.DataFrame) -> pl.DataFrame:
    relevant_outcomes = (
        outcomes.select(
            "session_code",
            "round_number",
            "group_id",
            "id_in_group",
            "start_time",
            "payoff_this_round",
        )
        .with_columns(
            coalition_member=pl.col("payoff_this_round") > 0,
        )
        .filter(
            pl.col("round_number") > 1,
        )
    )

    proposals = actions.filter(
        pl.col("round_number") > 1, pl.col("action") == "proposal"
    ).select(
        "session_code",
        "round_number",
        "group_id",
        "offer_id",
        time_of_winning_proposal="timestamp",
        agreement=pl.when(
            pl.col("member_1") == 1,
            pl.col("member_2") == 1,
            pl.col("member_3") == 1,
        )
        .then(pl.lit("Full agreement"))
        .otherwise(pl.lit("Partial agreement")),
    )

    final_choices = (
        actions.filter(pl.col("round_number") > 1, pl.col("action") == "acceptance")
        .join(
            relevant_outcomes,
            on=["session_code", "round_number", "group_id", "id_in_group"],
            how="left",
        )
        .filter(
            pl.col("coalition_member"),
        )
        .sort(by=["session_code", "round_number", "group_id", "timestamp"])
        .group_by(["session_code", "round_number", "group_id"])
        .agg(
            pl.last("start_time"),
            pl.last("timestamp").alias("time_of_final_agreement"),
            pl.last("treatment_name"),
            pl.last("accepted_offer").alias("offer_id"),
        )
        .join(
            proposals,
            on=["session_code", "round_number", "group_id", "offer_id"],
            how="left",
        )
        .with_columns(
            treatment_name_nice=pl.col("treatment_name").replace(
                {
                    "treatment_dummy_player": "Dummy player",
                    "treatment_y_10": "Y = 10",
                    "treatment_y_30": "Y = 30",
                    "treatment_y_90": "Y = 90",
                }
            ),
            time_until_winning_proposal=pl.col("time_of_winning_proposal")
            - pl.col("start_time"),
            time_until_final_agreement=pl.col("time_of_final_agreement")
            - pl.col("start_time"),
        )
        .with_columns(
            row_number=pl.col("time_until_winning_proposal")
            .rank(method="ordinal", descending=True)
            .over("treatment_name_nice"),
            round_number_nice=pl.format("Round {}", pl.col("round_number") - 1),
        )
    )

    return final_choices


def timing_until_decision(df: pl.DataFrame) -> so.Plot:
    empty_formatter = FixedFormatter([])

    plot = (
        so.Plot(df)
        .add(
            so.Range(),
            y="row_number",
            xmin="time_until_winning_proposal",
            xmax="time_until_final_agreement",
            color="agreement",
        )
        .label(x="Time (m)", y="", color="")
        .facet(
            col="treatment_name_nice",
            order=["Dummy player", "Y = 10", "Y = 30", "Y = 90"],
        )
        .scale(
            x=so.Continuous()
            .tick(at=[0, 60, 120, 180, 240, 300])
            .label(like=lambda x, _: f"{x/60:.0f}"),  # type: ignore
            y=so.Continuous().tick().label(formatter=empty_formatter),
            color=so.Nominal(order=["Full agreement", "Partial agreement"]),
        )
    )
    return plot


def timing_until_agreement_scatterplot(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(df, y="treatment_name_nice")
        .add(
            so.Dot(),
            so.Jitter(y=0.2),
            x="time_until_final_agreement",
            color="agreement",
        )
        .add(
            so.Dot(marker="|", pointsize=20, stroke=2, color="black"),
            so.Agg(),
            x="time_until_final_agreement",
        )
        .label(x="Time (m)", y="", color="")
        .scale(
            x=so.Continuous()
            .tick(at=[0, 60, 120, 180, 240, 300])
            .label(like=lambda x, _: f"{x/60:.0f}"),  # type: ignore
            color=so.Nominal(order=["Full agreement", "Partial agreement"]),
            y=so.Nominal(order=["Dummy player", "Y = 10", "Y = 30", "Y = 90"]),
        )
    )
    return plot


def timing_until_agreement_by_round(df: pl.DataFrame) -> so.Plot:
    plot = (
        so.Plot(df, y="treatment_name_nice")
        .add(
            so.Bar(),
            so.Agg(),
            so.Dodge(),
            x="time_until_final_agreement",
            color="round_number_nice",
        )
        .label(x="Time (m)", y="", color="")
        .scale(
            x=so.Continuous()
            .tick(at=[0, 60, 120, 180, 240, 300])
            .label(like=lambda x, _: f"{x/60:.0f}"),  # type: ignore
            y=so.Nominal(order=["Dummy player", "Y = 10", "Y = 30", "Y = 90"]),
            color=so.Nominal(order=[f"Round {i + 1}" for i in range(5)]),
        )
    )
    return plot


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    df = prepare_dataset(actions, outcomes)
    width = float(snakemake.wildcards.width)  # noqa F821 # type: ignore
    height = float(snakemake.wildcards.height)  # noqa F821 # type: ignore

    try:
        funcname = "timing_" + snakemake.wildcards.plot  # noqa F821 # type: ignore
        plot = globals()[funcname](df)
    except KeyError:
        raise ValueError(f"Unknown plot: {snakemake.wildcards.plot}")  # noqa F821 # type: ignore

    plot.layout(size=(width, height)).save(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
