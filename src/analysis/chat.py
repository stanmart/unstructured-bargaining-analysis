import re
import string

import nltk
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def prepare_dataset(outcomes: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    agreement_types = pl.Enum(["Breakdown", "Partial agreement", "Full agreement"])
    treatment_names = pl.Enum(
        [
            "Dummy player",
            "Y = 10",
            "Y = 30",
            "Y = 90",
        ]
    )

    chat = (
        (
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
        .group_by(["session_code", "round_number", "group_id"])
        .agg(pl.col("message").str.concat(" | ").alias("message"))
    )

    outcome_properties = (
        outcomes.filter(
            pl.col("round_number") > 1,
        )
        .group_by(["session_code", "round_number", "group_id"])
        .agg(
            agreement=(
                pl.when(pl.col("payoff_this_round").sum() == 0)
                .then(pl.lit("Breakdown"))
                .otherwise(
                    pl.when((pl.col("payoff_this_round") > 0).all())
                    .then(pl.lit("Full agreement"))
                    .otherwise(pl.lit("Partial agreement"))
                )
            ).cast(agreement_types),
            equal_split=(
                (pl.col("payoff_this_round").max() - pl.col("payoff_this_round").min())
                <= 1
            )
            & pl.col("payoff_this_round").sum()
            != 0,
        )
    )

    chat_with_outcomes = chat.filter(pl.col("round_number") > 1).join(
        outcome_properties,
        on=["session_code", "round_number", "group_id"],
        how="left",
    )

    return chat_with_outcomes


def make_tfifd_transformer() -> TfidfVectorizer:
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
    )
    return vectorizer


def make_classifier() -> XGBClassifier:
    return XGBClassifier(enable_categorical=True)


def make_classification_pipeline() -> Pipeline:
    return make_pipeline(make_tfifd_transformer(), make_classifier())


def fit_classifier(df: pl.DataFrame, group_var: str) -> tuple[Pipeline, LabelEncoder]:
    pipeline = make_classification_pipeline()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[group_var])
    pipeline.fit(df["message"], y)  # type: ignore
    return pipeline, label_encoder


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
    pipeline, label_encoder = fit_classifier(df, "agreement")
