import os
import pickle

import evaluate
import numpy as np
import polars as pl
import shap
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)


class ChatDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length: int = 64):
        self.tokens = [
            tokenizer(msg, max_length=max_length, truncation=True, padding="max_length")
            for msg in X
        ]
        self.labels = y

    def __getitem__(self, idx):
        return self.tokens[idx] | {"labels": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


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


def compile_trainer(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
) -> Trainer:
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,  # type: ignore
        tokenizer=tokenizer,
    )

    return trainer


if __name__ == "__main__":
    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore
    outcomes = pl.read_csv(snakemake.input.outcomes)  # noqa F821 # type: ignore
    outcome_var = snakemake.wildcards.outcome_var  # noqa F821 # type: ignore
    task = snakemake.params.task  # noqa F821 # type: ignore
    model_name = snakemake.wildcards.model_name  # noqa F821 # type: ignore

    df = prepare_dataset(outcomes, actions).filter(
        pl.col("treatment_name") != "treatment_dummy_player"
    )
    X_train, X_test, y_train, y_test = train_test_split_dataset(
        df, test_size=0.2, outcome=outcome_var
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task == "train":
        model_path = snakemake.output.model  # noqa F821 # type: ignore
        model_dir = os.path.dirname(model_path)

        training_args = TrainingArguments(
            output_dir=model_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            eval_strategy="epoch",
        )

        train_dataset = ChatDataset(X_train, y_train, tokenizer)
        eval_dataset = ChatDataset(X_test, y_test, tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        trainer = compile_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,  # type: ignore
        )

        trainer.train()
        trainer.save_model(model_dir)

    elif task == "shap":
        model_path = snakemake.input.model  # noqa F821 # type: ignore
        model_dir = os.path.dirname(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        prediction_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            max_length=64,
            truncation=True,
            device=0,
        )

        explainer = shap.Explainer(prediction_pipeline)
        shap_values = explainer(df["message"])
        with open(snakemake.output.shap_values, "wb") as f:  # noqa F821 # type: ignore
            pickle.dump(shap_values, f)

    else:
        raise ValueError(f"Task {task} not recognized")
