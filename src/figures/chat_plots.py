import string

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import spacy
from matplotlib.figure import Figure


def prepare_dataset(outcomes: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    player_names = pl.Enum(["P1", "P2", "P3"])
    agreement_types = pl.Enum(["Breakdown", "Partial agreement", "Full agreement"])
    roles = {
        1: "P1",
        2: "P2",
        3: "P3",
    }
    treatment_names = pl.Enum(
        [
            "Dummy player",
            "Y = 10",
            "Y = 30",
            "Y = 90",
        ]
    )

    chat = (
        pl.read_csv("data/clean/_collected/actions.csv")
        .filter(pl.col("action").eq("chat"))
        .select(
            "session_code",
            "treatment_name",
            "round_number",
            "group_id",
            "id_in_group",
            message="body",
        )
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
    )

    outcome_properties = (
        outcomes.filter(
            pl.col("round_number") > 1,
        )
        .with_columns(
            max_payoff=pl.col("payoff_this_round")
            .max()
            .over(["session_code", "round_number", "group_id"]),
            min_payoff=pl.col("payoff_this_round")
            .min()
            .over(["session_code", "round_number", "group_id"]),
            total_payoff=pl.col("payoff_this_round")
            .sum()
            .over(["session_code", "round_number", "group_id"]),
        )
        .with_columns(
            role=pl.col("id_in_group").replace(roles).cast(player_names),
            agreement=(
                pl.when(pl.col("total_payoff") == 0)
                .then(pl.lit("Breakdown"))
                .otherwise(
                    pl.when(pl.col("min_payoff") > 0)
                    .then(pl.lit("Full agreement"))
                    .otherwise(pl.lit("Partial agreement"))
                )
            ).cast(agreement_types),
            equal_split=(
                (pl.col("max_payoff") - pl.col("min_payoff") <= 1)
                & (pl.col("total_payoff") > 0)
            ),
        )
    )

    chat_with_outcomes = chat.filter(pl.col("round_number") > 1).join(
        outcome_properties.select(
            [
                "session_code",
                "round_number",
                "group_id",
                "id_in_group",
                "payoff_this_round",
                "agreement",
                "equal_split",
            ]
        ),
        on=["session_code", "round_number", "group_id", "id_in_group"],
        how="left",
    )

    return chat_with_outcomes


def lemmatize_chat(chat: pl.DataFrame, model: spacy.Language) -> pl.DataFrame:  # type: ignore
    def lemmatize(text: str) -> list[dict[str, str]]:
        nlp = model(text)
        return [
            {
                "lemma": token.lemma_,
                "pos": token.pos_,
            }
            for token in nlp
        ]

    return (
        chat.with_columns(
            words=pl.col("message").apply(
                lemmatize,
                return_dtype=pl.List(pl.Struct({"lemma": pl.Utf8, "pos": pl.Utf8})),
            ),
        )
        .explode("words")
        .unnest("words")
        .filter(pl.col("lemma").is_not_null())
    )


def count_words(
    df: pl.DataFrame, predicament: pl.Expr, word_type: str | None = None
) -> pl.DataFrame:
    if word_type is not None:
        df = df.filter(pl.col("pos") == word_type)

    exclude_words = set(string.punctuation) | {"a", "a1", "a2", "b", "b1", "b2"}

    word_counts = (
        df.filter(
            pl.col("lemma").is_in(exclude_words).not_(),
            pl.col("lemma").str.contains(r"\d+$").not_(),
            pl.col("lemma").str.contains(r"id\d+$").not_(),
        )
        .with_columns(
            group_var=predicament,
        )
        .groupby(["group_var", "lemma"])
        .agg(
            count=pl.count("lemma"),
        )
    )

    freq_in_group = (
        word_counts.with_columns(
            total_count=pl.sum("count").over("group_var"),
        )
        .with_columns(
            freq=pl.col("count") / pl.col("total_count"),
        )
        .pivot(values="freq", index="lemma", columns="group_var")
        .fill_null(0)
        .select(pl.col("lemma"), pl.all().exclude("lemma").name.prefix("freq_"))
    )

    freq_total = (
        word_counts.group_by("lemma")
        .agg(
            count=pl.sum("count"),
        )
        .with_columns(
            total_count=pl.sum("count"),
        )
        .with_columns(
            freq_total=pl.col("count") / pl.col("total_count"),
        )
        .select("lemma", "freq_total")
    )

    freq_table = (
        freq_in_group.join(freq_total, on="lemma", how="left")
        .with_columns(
            freq_true_minus_false=pl.col("freq_true") - pl.col("freq_false"),
            relative_freq_true=pl.col("freq_true") / pl.col("freq_total"),
            relative_freq_false=pl.col("freq_false") / pl.col("freq_total"),
        )
        .with_columns(
            relative_freq_true_minus_false=pl.col("freq_true_minus_false")
            / pl.col("freq_total"),
            log_relative_freq_true=(pl.col("relative_freq_true") + 1).log(),
            log_relative_freq_false=(pl.col("relative_freq_false") + 1).log(),
        )
    )

    return freq_table


def create_plot(
    df: pl.DataFrame,
    predicament: pl.Expr,
    group_true_name: str,
    group_false_name: str,
    word_type: str | None = None,
    top_k: int = 5,
    total_freq_threshold: float = 0.002,
) -> Figure:
    freq_table = (
        count_words(df, predicament, word_type)
        .filter(pl.col("freq_total") > total_freq_threshold)
        .with_columns(
            relative_freq_false=-pl.col("relative_freq_false"),
        )
    )
    plot_data = (
        pl.concat(
            [
                freq_table.sort("relative_freq_true", descending=True).head(top_k),
                freq_table.sort("relative_freq_false", descending=True).tail(top_k),
            ]
        )
        .rename(
            {
                "relative_freq_true": group_true_name,
                "relative_freq_false": group_false_name,
            }
        )
        .melt(
            id_vars="lemma",
            value_vars=[
                group_true_name,
                group_false_name,
            ],
            value_name="Relative frequency",
            variable_name="group_var",
        )
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=plot_data,
        x="Relative frequency",
        hue="group_var",
        y="lemma",
        ax=ax,
    )

    return fig


def setup_spacy(model_name) -> spacy.Language:  # type: ignore
    try:
        model = spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)  # type: ignore
        model = spacy.load(model_name)

    return model


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

    model = setup_spacy("en_core_web_sm")
    df = prepare_dataset(outcomes, actions)
    df_processed = lemmatize_chat(df, model)
