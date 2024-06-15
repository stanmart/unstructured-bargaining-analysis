import polars as pl
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
        actions.filter(pl.col("action").eq("chat")).select(
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
                "role",
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
            words=pl.col("message").map_elements(
                lemmatize,
                return_dtype=pl.List(pl.Struct({"lemma": pl.Utf8, "pos": pl.Utf8})),
            ),
        )
        .explode("words")
        .unnest("words")
        .filter(pl.col("lemma").is_not_null())
    )


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

    df_processed.write_csv(snakemake.output.lemmas)  # noqa F821 # type: ignore
