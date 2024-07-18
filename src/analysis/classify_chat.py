import glob
import os
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

Please classify which TOPIC each message (MSG) belongs to. You only have to classify messages, not proposals or acceptances (those latter two are only included for context). The classification should also take into account the context of the message (e.g. when a message is a reply to another).
Each message should be classified into one main and one subtopic. The topics are given in the following nested list:
 - small talk: messages that are not directly related to the experiment
    - greetings and farewells: e.g. saying hello, goodbye, etc.
    - other: e.g. talking about the weather, how to spend the remaining time, etc.
 - bargaining: messages discussing the distribution of the money, making and reacting to proposals, counter-proposals, etc.
    - fairness-based: using arguments based on fairness or justice ideas
    - non-fairness-based: using arguments based on other considerations
 - meta-talk: talking about the experiment itself
    - purpose: discussing what the experimenters are trying to find out
    - rules: discussing and clarifying the rules of the experiment
    - identification: identifying each other, e.g. trying to figure out if players met each other in previous rounds, or identifying information for later

Your response should be of the following format:
#[MESSAGE_ID]: [MAIN_TOPIC], [SUB_TOPIC]
for each message, separated by newlines.
It should look like the contents of a dictionary, but without the surrounding curly braces and apostrophes.
Do not include any other lines, such as code block delimiters or comments.
If there are no rows of type MSG, please respond with NO_MESSAGES without any additional content, such as IDs or comments.
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


def get_model_response(
    client: openai.Client, chat_log: str, model: str = "gpt-4o"
) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": BASE_PROMPT,
            },
            {
                "role": "user",
                "content": chat_log,
            },
        ],
        model=model,
        temperature=0.0,
    )

    if response.choices[0].message.content is None:
        raise ValueError("OpenAI response is None")

    return response.choices[0].message.content


def parse_model_response_row(row: str) -> dict[str, int | str]:
    try:
        message_id, topic = row.split(":")
        main_topic, sub_topic = topic.split(",")
        return {
            "action_id": int(message_id.lstrip("#")),
            "main_topic": main_topic.strip(),
            "sub_topic": sub_topic.strip(),
        }
    except ValueError as e:
        print(f"Could not parse row: {row}")
        raise e


def parse_model_response(model_response: str) -> pl.DataFrame:
    model_response = model_response.strip().lstrip("{").rstrip("}").strip()
    if model_response == "NO_MESSAGES":
        rows = []
    else:
        rows = (parse_model_response_row(row) for row in model_response.split("\n"))
    return pl.DataFrame(
        rows,
        schema={
            "action_id": pl.UInt32,
            "main_topic": pl.String,
            "sub_topic": pl.String,
        },
    )


def analyze_log(
    client: openai.Client, df: pl.DataFrame, model: str = "gpt-4o"
) -> pl.DataFrame:
    chat_log = create_chat_log(df)
    model_response = get_model_response(client, chat_log, model)
    model_response_df = parse_model_response(model_response)

    result_df = model_response_df.join(
        df.filter(pl.col("action") == "chat"), on="action_id", how="left"
    ).select(
        [
            "treatment_name",
            "round_number",
            "group_id",
            "id_in_group",
            "timestamp",
            "body",
            "main_topic",
            "sub_topic",
        ]
    )

    return result_df


if __name__ == "__main__":
    cache_files = glob.glob(
        os.path.join(snakemake.params["cache_dir"], "*.parquet")  # noqa F821 # type: ignore
    )
    execution_allowed = os.environ.get("ALLOW_OPENAI_REQUESTS") == "true"
    delete_cache = os.environ.get("DELETE_CACHE") == "true"

    if execution_allowed:
        print("OpenAI requests are enabled.")
        client = create_openai_client()

        if cache_files and delete_cache:
            print("Cache directory is not empty. Deleting cache files.")
            for file in cache_files:
                os.remove(file)

        os.makedirs(snakemake.params["cache_dir"], exist_ok=True)  # noqa F821 # type: ignore

        actions = pl.read_csv(snakemake.input.actions)  # noqa F821 # type: ignore

        df = prepare_dataset(actions)

        for (treatment_name, group_id, round_number), dfi in df.group_by(  # type: ignore
            "treatment_name", "group_id", "round_number"
        ):
            cache_file_path = os.path.join(
                snakemake.params["cache_dir"],  # noqa F821 # type: ignore
                f"{treatment_name}_g{group_id}_r{round_number}.parquet",
            )
            if os.path.exists(cache_file_path):
                print(
                    f"Using cached file for {treatment_name}, group {group_id}, round {round_number}."
                )
            else:
                print(
                    f"Classifying chat messages for {treatment_name}, round {round_number}, group {group_id}."
                )
                analyze_log(client, dfi).write_parquet(cache_file_path)

        print("Done classyfing chat messages.")
        print(
            "Consider committing the cache directory to avoid unnecessary OpenAI requests."
        )

    else:
        print("OpenAI requests are disabled.")
        if cache_files:
            print(f"Using {len(cache_files)} cached files only.")
        else:
            print("No cache files found. Exiting.")
            exit(1)

    new_cache_files = glob.glob(
        os.path.join(snakemake.params["cache_dir"], "*.parquet")  # noqa F821 # type: ignore
    )
    result_df_long = pl.concat(pl.read_parquet(file) for file in new_cache_files)
    result_df_long.write_csv(snakemake.output.chat_classified)  # noqa F821 # type: ignore
