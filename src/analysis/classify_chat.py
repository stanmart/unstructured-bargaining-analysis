import os
from pathlib import Path
from typing import Any

import openai
import polars as pl

BASE_PROMPT = """
You are going to receive a log containing messages between three players from an economics lab experiment. Players bargained how to split an amount of money. They could additionally use an interface for submitting and accepting proposals. Before the bargaining, players did a slider task and their performance determined their bargaining position.

The log format is the following:
MSG #[MESSAGE_ID] @[PLAYER_NAME]: [MESSAGE]
PROP #[PROPOSAL_ID] @[PLAYER_NAME]: [distribution of the money]
ACC #[ACCEPTANCE_ID] @[PLAYER_NAME]: PROP #[PROPOSAL_ID]
separated by newlines.

Please classify which TOPIC each message (MSG) belongs to. You only have to classify messages, not proposals or acceptances (those latter two are only included for context). Possible topics are: bargaining-related, small-talk, experiment-related, fairness-related, meta-talk, identifying-each-other. The classification should also take into account the context of the message.

Your response should look like the following:
#[MESSAGE_ID]: [TOPIC]
for each message, separated by newlines.
Please do not return anything else.

Chat log:

"""


def prepare_dataset(actions: pl.DataFrame) -> pl.DataFrame:
    actions_clean = actions.with_columns(
        player_name=pl.when(
            pl.col("treatment_name") == "treatment_dummy_player",
        )
        .then(
            pl.col("id_in_group").replace(
                {
                    1: "A1",
                    2: "A2",
                    3: "B",
                }
            )
        )
        .otherwise(
            pl.col("id_in_group").replace(
                {
                    1: "A",
                    2: "B1",
                    3: "B2",
                }
            )
        ),
        treatment_name_nice=pl.col("treatment_name").replace(
            {
                "treatment_dummy_player": "Dummy player",
                "treatment_y_10": "Y = 10",
                "treatment_y_30": "Y = 30",
                "treatment_y_90": "Y = 90",
            }
        ),
        action_id=pl.lit(0),
    ).with_columns(
        action_id=pl.col("action_id")
        .cum_count()
        .over(["treatment_name", "round_number", "group_id", "action"])
    )

    return actions_clean


def create_chat_log(df: pl.DataFrame) -> str:
    example_log_str = "\n".join(
        (format_chat_row(i, row) for i, row in enumerate(df.iter_rows(named=True)))
    )
    return example_log_str


def format_chat_row(i: int, row: dict[str, Any]) -> str:
    players = (
        ["A", "B1", "B2"]
        if row["treatment_name"] != "treatment_dummy_player"
        else ["A1", "A2", "B"]
    )

    if row["action"] == "chat":
        return f"MSG #{row['action_id']} @{row['player_name']}: {row['body']}"
    elif row["action"] == "proposal":
        content = ", ".join(
            f"{player}: {row['allocation_' + id]}"
            for player, id in zip(players, ["1", "2", "3"])
        )
        return f"PROP #{row['offer_id']} @{row['player_name']}: {content}"
    elif row["action"] == "acceptance":
        return f"ACC #{row['action_id']} @{row['player_name']}: PROP #{row['accepted_offer']}"
    else:
        raise ValueError(f"Unknown action: {row['action']}")


def create_openai_client() -> openai.Client:
    return openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_model_response(client: openai.Client, chat_log: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": BASE_PROMPT + chat_log,
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0.0,
    )

    if response.choices[0].message.content is None:
        raise ValueError("OpenAI response is None")

    return response.choices[0].message.content


def parse_model_response_row(row: str) -> dict[str, int | str]:
    message_id, topic = row.split(":")
    return {
        "action_id": int(message_id.lstrip("#")),
        "topic": topic.strip(),
    }


def parse_model_response(model_response: str) -> pl.DataFrame:
    rows = model_response.split("\n")
    return pl.DataFrame(
        (parse_model_response_row(row) for row in rows),
        schema={
            "action_id": pl.UInt32,
            "topic": pl.String,
        },
    )


def analyze_log(client: openai.Client, df: pl.DataFrame) -> pl.DataFrame:
    chat_log = create_chat_log(df)
    model_response = get_model_response(client, chat_log)
    model_response_df = parse_model_response(model_response)

    result_df = (
        df.filter(pl.col("action") == "chat")
        .join(model_response_df, on="action_id")
        .select(
            [
                "treatment_name",
                "round_number",
                "group_id",
                "id_in_group",
                "timestamp",
                "body",
                "topic",
            ]
        )
    )

    return result_df


if __name__ == "__main__":
    output_file = Path(snakemake.output[0])  # noqa F821 # type: ignore
    output_exists = output_file.is_file()
    execution_allowed = os.environ.get("ALLOW_OPENAI_REQUESTS") == "true"

    if output_exists:
        print("Output file already exists, but Snakemake wants to recreate it.")
        if not execution_allowed:
            print("OpenAI requests are disabled. Using existing file.")
            output_file.touch()
            exit(0)
        else:
            print("OpenAI requests are enabled. Recreating file.")
    else:
        print("Output file does not exist.")
        if not execution_allowed:
            print("OpenAI requests are disabled. Exiting.")
            exit(1)
        else:
            print("OpenAI requests are enabled. Creating file.")

    actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

    df = prepare_dataset(actions)[:100]

    result_dfs = []
    for _, dfi in df.group_by("treatment_name", "group_id", "round_number"):
        result_dfs.append(analyze_log(create_openai_client(), dfi))

    result_df_long = pl.concat(result_dfs)
    result_df_long.write_csv(output_file)

    print("Done classyfing chat messages.")
    print(
        "Consider committing the output to the reository to avoid unnecessary OpenAI requests."
    )
