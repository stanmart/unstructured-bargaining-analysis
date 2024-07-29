import json

import polars as pl


def read_session_details(session_details_path: str) -> dict:
    with open(session_details_path, "r") as file:
        return json.load(file)


def concatenate_actions(
    action_data_paths: list[str], sessions_details: list[dict]
) -> pl.LazyFrame:
    action_data = [
        pl.scan_csv(path).select(
            pl.lit(session_details["session_code"]).alias("session_code"),
            pl.lit(session_details["treatment_name"]).alias("treatment_name"),
            pl.all(),
        )
        for path, session_details in zip(action_data_paths, sessions_details)
    ]
    return pl.concat(action_data, how="diagonal").sort(
        "treatment_name", "session_code", "round_number", "group_id", "timestamp"
    )


def concatenate_outcomes(
    outcome_data_paths: list[str], sessions_details: list[dict]
) -> pl.LazyFrame:
    outcome_data = [
        pl.scan_csv(path).select(
            pl.lit(session_details["session_code"]).alias("session_code"),
            pl.lit(session_details["treatment_name"]).alias("treatment_name"),
            pl.all(),
        )
        for path, session_details in zip(outcome_data_paths, sessions_details)
    ]
    return pl.concat(outcome_data, how="diagonal").sort(
        "treatment_name", "session_code", "round_number", "group_id", "id_in_group"
    )


def concatenate_personal(
    personal_data_paths: list[str], sessions_details: list[dict]
) -> pl.LazyFrame:
    personal_data = [
        pl.scan_csv(path).select(
            pl.lit(session_details["session_code"]).alias("session_code"),
            pl.lit(session_details["treatment_name"]).alias("treatment_name"),
            pl.all(),
        )
        for path, session_details in zip(personal_data_paths, sessions_details)
    ]
    return pl.concat(personal_data, how="diagonal").sort(
        "treatment_name", "session_code"
    )


if __name__ == "__main__":
    sessions_details = [
        read_session_details(path)
        for path in snakemake.input.session_details  # noqa F821 # type: ignore
    ]

    action_data = concatenate_actions(snakemake.input.actions, sessions_details)  # noqa F821 # type: ignore
    action_data.sink_csv(snakemake.output.actions)  # noqa F821 # type: ignore

    outcome_data = concatenate_outcomes(snakemake.input.outcomes, sessions_details)  # noqa F821 # type: ignore
    outcome_data.sink_csv(snakemake.output.outcomes)  # noqa F821 # type: ignore

    personal_data = concatenate_personal(snakemake.input.personal, sessions_details)  # noqa F821 # type: ignore
    personal_data.sink_csv(snakemake.output.personal)  # noqa F821 # type: ignore
