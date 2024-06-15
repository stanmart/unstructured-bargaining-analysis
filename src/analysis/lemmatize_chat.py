import polars as pl
import spacy


def prepare_dataset(actions: pl.DataFrame) -> pl.DataFrame:
    treatment_names = pl.Enum(
        [
            "Dummy player",
            "Y = 10",
            "Y = 30",
            "Y = 90",
        ]
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

    return chat


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
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

    model = setup_spacy("en_core_web_sm")
    df = prepare_dataset(actions)
    lemmas = lemmatize_chat(df, model)

    lemmas.write_csv(snakemake.output.lemmas)  # noqa F821 # type: ignore
