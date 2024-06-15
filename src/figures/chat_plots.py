import string

import polars as pl
import seaborn as sns
import spacy


def prepare_dataset(outcomes: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    player_roles = pl.Enum(["A", "B"])
    split_types = pl.Enum(["Equal split", "Unequal split / breakdown"])
    agreement_types = pl.Enum(["Full agreement", "Partial agreement / breakdown"])
    treatment_names = pl.Enum(
        [
            "Dummy player",
            "Y = 10",
            "Y = 30",
            "Y = 90",
        ]
    )

    roles = pl.DataFrame(
        [
            {"treatment_name": "treatment_dummy_player", "id_in_group": 1, "role": "A"},
            {"treatment_name": "treatment_dummy_player", "id_in_group": 2, "role": "A"},
            {"treatment_name": "treatment_dummy_player", "id_in_group": 3, "role": "B"},
            {"treatment_name": "treatment_y_10", "id_in_group": 1, "role": "A"},
            {"treatment_name": "treatment_y_10", "id_in_group": 2, "role": "A"},
            {"treatment_name": "treatment_y_10", "id_in_group": 3, "role": "B"},
            {"treatment_name": "treatment_y_30", "id_in_group": 1, "role": "A"},
            {"treatment_name": "treatment_y_30", "id_in_group": 2, "role": "A"},
            {"treatment_name": "treatment_y_30", "id_in_group": 3, "role": "B"},
            {"treatment_name": "treatment_y_90", "id_in_group": 1, "role": "A"},
            {"treatment_name": "treatment_y_90", "id_in_group": 2, "role": "A"},
            {"treatment_name": "treatment_y_90", "id_in_group": 3, "role": "B"},
        ]
    ).with_columns(
        role=pl.col("role").cast(player_roles),
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
            agreement=(
                pl.when(pl.col("min_payoff") > 0)
                .then(pl.lit("Full agreement"))
                .otherwise(pl.lit("Partial agreement / breakdown"))
            ).cast(agreement_types),
            equal_split=(
                pl.when(
                    (pl.col("max_payoff") - pl.col("min_payoff") <= 1)
                    & (pl.col("total_payoff") > 0)
                )
                .then(pl.lit("Equal split"))
                .otherwise(pl.lit("Unequal split / breakdown"))
            ).cast(split_types),
        )
    ).join(
        roles,
        on=["treatment_name", "id_in_group"],
        how="left",
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
    df: pl.DataFrame, group_var: str, word_type: str | None = None
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
        .groupby([group_var, "lemma"])
        .agg(
            count=pl.count("lemma"),
        )
    )

    freq_in_group = (
        word_counts.with_columns(
            total_count=pl.sum("count").over(group_var),
        )
        .with_columns(
            freq=pl.col("count") / pl.col("total_count"),
        )
        .pivot(values="freq", index="lemma", columns=group_var)
        .fill_null(0)
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

    freq_table = freq_in_group.join(freq_total, on="lemma", how="left")

    freq_table_long = pl.concat(
        [
            freq_table.with_columns(type=pl.lit("absolute")),
            freq_table.with_columns(
                (pl.all().exclude("lemma", "freq_total") / pl.col("freq_total")),
                type=pl.lit("relative"),
            ),
        ],
        how="diagonal",
    )

    return freq_table_long


def create_plot(
    df: pl.DataFrame,
    group_var: str,
    word_type: str | None = None,
    top_k: int = 5,
    total_freq_threshold: float = 0.002,
    type="relative",
) -> sns.FacetGrid:
    value_name = (
        "Relative frequency (group / total)"
        if type == "relative"
        else "Frequency (group / total)"
    )
    freq_table = (
        count_words(df, group_var, word_type)
        .filter(pl.col("freq_total") > total_freq_threshold, pl.col("type") == type)
        .select(pl.all().exclude("freq_total", "type"))
    )
    group_levels = [col for col in freq_table.columns if col not in ["lemma", "facet"]]
    plot_data = pl.concat(
        [
            freq_table.sort(level, descending=True)
            .head(top_k)
            .with_columns(facet=pl.lit(level))
            for level in group_levels
        ]
    ).melt(
        id_vars=["lemma", "facet"],
        value_vars=group_levels,
        value_name=value_name,
        variable_name="group_var",
    )

    word_type_nice = {
        "NOUN": "nouns",
        "VERB": "verbs",
        "ADJ": "adjectives",
        None: "words",
    }

    palette = {level: color for level, color in zip(group_levels, sns.color_palette())}

    g = sns.FacetGrid(
        plot_data,
        col="facet",
        sharey=False,
        legend_out=True,
        aspect=1.3,
        height=2.8,
    )
    g.map_dataframe(
        sns.barplot,
        x=value_name,
        hue="group_var",
        y="lemma",
        alpha=0.9,
        palette=palette,
    )

    g.set_ylabels("")
    g.add_legend(title="")
    g.set_titles(
        col_template=f"Top {top_k} {word_type_nice[word_type]} in '{{col_name}}'"
    )

    if type == "relative":
        for ax in g.axes.flat:
            ax.axvline(1, color="black", linestyle=":")

    return g


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
