import os
from typing import Any

import numpy as np
import polars as pl
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from transformers import (
    BatchEncoding,
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
)


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


def train_test_split_dataset(
    df: pl.DataFrame, outcome: str, test_size: float = 0.2
) -> tuple[pl.Series, pl.Series, pl.Series, pl.Series]:
    stratify_column = df[outcome] if outcome is not None else None
    train_df, test_df = train_test_split(  # type: ignore
        df, test_size=test_size, stratify=stratify_column, random_state=8001
    )

    X_train = train_df["message"]
    X_test = test_df["message"]
    y_train = train_df[outcome].to_physical()
    y_test = test_df[outcome].to_physical()

    return X_train, X_test, y_train, y_test


def tokenize_series(
    df: pl.Series, tokenizer: DistilBertTokenizer, max_length: int = 50
) -> BatchEncoding:
    return tokenizer(
        text=df.to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
    )


def prepare_data_for_training(
    X_train: pl.Series,
    X_test: pl.Series,
    y_train: pl.Series,
    y_test: pl.Series,
    tokenizer: DistilBertTokenizer,
    max_tokens: int = 100,
) -> tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    train_tokens = tokenize_series(X_train, tokenizer, max_length=max_tokens)
    test_tokens = tokenize_series(X_test, tokenizer, max_length=max_tokens)
    train_outcome = y_train.to_numpy()
    test_outcome = y_test.to_numpy()

    return train_tokens, test_tokens, train_outcome, test_outcome


def compile_model(
    model_name: str, learning_rate: float = 1e-5, max_tokens: int = 100
) -> TFDistilBertForSequenceClassification:
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=2, max_length=max_tokens
    )

    optimizer = Adam(learning_rate=5e-5)
    metric = CategoricalAccuracy("balanced_accuracy")

    model.compile(  # type: ignore
        optimizer=optimizer, metrics=[metric]
    )

    return model  # type: ignore


def fit_model(
    model: TFDistilBertForSequenceClassification,
    train_tokens: BatchEncoding,
    test_tokens: BatchEncoding,
    train_outcome: np.ndarray,
    test_outcome: np.ndarray,
    epochs: int = 1,
    batch_size: int = 32,
) -> Any:
    history = model.fit(  # type: ignore
        x={
            "input_ids": train_tokens["input_ids"],
            "attention_mask": train_tokens["attention_mask"],
        },
        y=train_outcome,
        validation_data=(
            {
                "input_ids": test_tokens["input_ids"],
                "attention_mask": test_tokens["attention_mask"],
            },
            test_outcome,
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )

    return history


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    outcome_var = snakemake.wildcards.outcome_var  # noqa F821 # type: ignore
    task = snakemake.params.task  # noqa F821 # type: ignore
    model_path = snakemake.output.model  # noqa F821 # type: ignore

    model_dir = os.path.dirname(model_path)
    model_name = "distilbert-base-uncased"

    df = prepare_dataset(outcomes, actions)
    X_train, X_test, y_train, y_test = train_test_split_dataset(
        df, test_size=0.2, outcome=outcome_var
    )

    max_tokens = 50
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    if task == "train":
        (
            train_tokens,
            test_tokens,
            train_outcome,
            test_outcome,
        ) = prepare_data_for_training(
            X_train, X_test, y_train, y_test, tokenizer=tokenizer, max_tokens=max_tokens
        )
        model = compile_model(model_name, max_tokens=max_tokens)
        fit_model(
            model,
            train_tokens,
            test_tokens,
            train_outcome,
            test_outcome,
            epochs=1,
            batch_size=32,
        )
        model.save_pretrained(model_dir)
