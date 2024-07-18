import polars as pl
import seaborn as sns


def prepare_final_choices(
    actions: pl.DataFrame, outcomes: pl.DataFrame
) -> pl.DataFrame:
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
            time_until_winning_proposal=pl.col("time_of_winning_proposal")
            - pl.col("start_time"),
            time_until_final_agreement=pl.col("time_of_final_agreement")
            - pl.col("start_time"),
        )
        .with_columns(
            row_number=pl.col("time_until_winning_proposal")
            .rank(method="ordinal", descending=True)
            .over("treatment_name"),
        )
    )

    return final_choices


def plot_chat_topics(
    chat: pl.DataFrame, final_choices: pl.DataFrame, until_agreement: bool = True
) -> sns.FacetGrid:
    treatment_names = pl.Enum(["Dummy player", "Y = 10", "Y = 30", "Y = 90"])

    main_topics = pl.Enum(["bargaining", "meta-talk", "small talk"])
    sub_topics = pl.Enum(
        [
            "fairness-based",
            "non-fairness-based",
            "rules",
            "purpose",
            "identification",
            "greetings and farewells",
            "other",
        ]
    )

    df_chat = (
        chat.filter(
            pl.col("main_topic").is_in(["bargaining", "meta-talk", "small talk"])
        )
        .with_columns(
            sub_topic_cleaned=pl.when(pl.col("sub_topic") == "farewells")
            .then(pl.lit("greetings and farewells"))
            .otherwise(pl.col("sub_topic"))
            .cast(sub_topics),
            main_topic=pl.col("main_topic").cast(main_topics),
        )
        .join(
            final_choices,
            on=["treatment_name", "round_number", "group_id"],
            how="left",
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
            .cast(treatment_names),
        )
    )

    if until_agreement:
        df_chat = df_chat.filter(
            pl.when(
                pl.col(
                    "time_of_final_agreement"
                ).is_null()  # include chats with coordination failures
            )
            .then(True)
            .otherwise(pl.col("timestamp") <= pl.col("time_of_final_agreement"))
        )

    g = sns.displot(
        df_chat,
        x="main_topic",
        hue="sub_topic_cleaned",
        multiple="stack",
        col="treatment_name_nice",
        col_wrap=2,
        height=3.5,
    )

    for ax in g.axes.flat:
        ax.xaxis.grid(False)

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Topics", "Message counts by subtopic")
    g.legend.set_title("Subtopics")  # type: ignore

    return g


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    final_choices = prepare_final_choices(actions, outcomes)
    chat = pl.read_csv(snakemake.input.chat)  # noqa F821 # type: ignore

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
    if snakemake.wildcards.sample == "all":  # noqa F821 # type: ignore
        plot = plot_chat_topics(chat, final_choices, until_agreement=False)
    elif snakemake.wildcards.sample == "until_agreement":  # noqa F821 # type: ignore
        plot = plot_chat_topics(chat, final_choices, until_agreement=True)
    else:
        raise ValueError(f"Unknown sample: {snakemake.wildcards.sample}")  # noqa F821 # type: ignore

    plot.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
