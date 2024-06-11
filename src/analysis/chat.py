import re
import string

import nltk
import numpy as np
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


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

    outcome_properties = outcomes.filter(
        pl.col("round_number") > 1,
    ).with_columns(
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
        equal_split=(
            pl.col("payoff_this_round")
            .max()
            .over(["session_code", "round_number", "group_id"])
            - pl.col("payoff_this_round")
            .min()
            .over(["session_code", "round_number", "group_id"])
        )
        <= 1,
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


def train_idf(df: pl.DataFrame, vectorizer_args={}) -> TfidfVectorizer:
    class LemmaTokenizer:
        def __init__(self):
            self.wnl = WordNetLemmatizer()
            self.ignore = set(stopwords.words("english")) | set(string.punctuation)
            self.exclude_re = re.compile(r"[0-9]+\.?[0-9]*")

        def __call__(self, doc):
            lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
            return [
                t
                for t in lemmas
                if t not in self.ignore and not self.exclude_re.match(t)
            ]

    vectorizer = TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        **vectorizer_args,
    )
    vectorizer.fit(df["message"])
    return vectorizer


def get_tfidf_matrix(
    df: pl.DataFrame, vectorizer: TfidfVectorizer, group_var: str
) -> tuple[np.ndarray, list]:
    documents = (
        df.group_by(group_var)
        .agg(pl.col("message").str.concat(" ").alias("message"))
        .sort(group_var)
    )
    groups = documents[group_var].to_list()
    tfidf_matrix = vectorizer.transform(documents["message"]).toarray()  # type: ignore

    return tfidf_matrix, groups


def get_top_k_words(
    df: pl.DataFrame, vectorizer: TfidfVectorizer, group_var: str, k: int
):
    tfidf_matrix, groups = get_tfidf_matrix(df, vectorizer, group_var)
    top_k_word_indices = np.argsort(tfidf_matrix, axis=1)[:, -k:][:, ::-1]

    words = vectorizer.get_feature_names_out()
    top_k_words = words[top_k_word_indices]

    return {group: list(top_words) for group, top_words in zip(groups, top_k_words)}


def setup_nltk() -> None:
    nltk.data.path.append("nltk_data")
    nltk.download("punkt", download_dir="nltk_data")
    nltk.download("stopwords", download_dir="nltk_data")
    nltk.download("wordnet", download_dir="nltk_data")


if __name__ == "__main__":
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

    setup_nltk()
    df = prepare_dataset(outcomes, actions)
    vectorizer = train_idf(df)
