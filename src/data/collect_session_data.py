import json

import polars as pl
import polars.selectors as cs


def load_session_details(path: str, session_code: str) -> dict[str, bool | int]:
    data = pl.scan_csv(path)
    session_details = (
        data.filter(pl.col("session.code") == session_code)
        .select(
            [
                "session.code",
                "session.is_demo",
                "session.config.real_world_currency_per_point",
                "session.config.seconds_per_round",
                "session.config.seconds_for_sliders",
                "session.config.name",
            ]
        )
        .rename(
            {
                "session.code": "session_code",
                "session.is_demo": "demo_session",
                "session.config.real_world_currency_per_point": "currency_per_point",
                "session.config.seconds_per_round": "seconds_for_bargaining",
                "session.config.seconds_for_sliders": "seconds_for_sliders",
                "session.config.name": "treatment_name",
            }
        )
        .unique()
        .collect()
    )

    if len(session_details) == 0:
        raise ValueError(f"No data found for session code {session_code}")

    if len(session_details) > 1:
        raise ValueError(f"Non-unique configuration found for {session_code}")

    return session_details.rows(named=True)[0]


def load_chat_data(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session_code") == session_code)
        .select(
            [
                "participant_code",
                "timestamp",
                "channel",
                "nickname",
                "body",
            ]
        )
    )
    return data


def load_live_data(
    path: str, session_code: str
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    data = pl.scan_csv(path).filter(pl.col("session_code") == session_code)

    page_loads = (
        data.filter(pl.col("event_type") == "page_load")
        .select(["session_code", "participant_code", "timestamp", "round_number"])
        .sort("round_number", "timestamp")
    )

    proposals = data.filter(pl.col("event_type") == "proposal").select(
        [
            "participant_code",
            "timestamp",
            "round_number",
            "offer_id",
            "group_id",
            "id_in_group",
            "offer_id",
            cs.starts_with("member_"),
            cs.starts_with("allocation_"),
        ]
    )

    acceptances = data.filter(pl.col("event_type") == "acceptance").select(
        [
            "participant_code",
            "timestamp",
            "round_number",
            "group_id",
            "id_in_group",
            "accepted_offer",
        ]
    )

    return page_loads, proposals, acceptances


def load_bargaining_data(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session.code") == session_code)
        .select(
            [
                "participant.code",
                "participant.label",
                "subsession.round_number",
                "group.id_in_subsession",
                "player.id_in_group",
                "player.accepted_offer",
                "player.payoff_this_round",
                "player.payoff",
                "participant.payoff",
                "subsession.start_time",
                "player.end_time",
                "subsession.expiry",
            ]
        )
        .rename(
            {
                "participant.code": "participant_code",
                "participant.label": "participant_label",
                "subsession.round_number": "round_number",
                "group.id_in_subsession": "group_id",
                "participant.payoff": "payoff_total",
                "player.id_in_group": "id_in_group",
                "player.accepted_offer": "accepted_offer",
                "player.payoff_this_round": "payoff_this_round",
                "player.payoff": "payoff_this_round_chf",
                "subsession.start_time": "start_time",
                "player.end_time": "end_time",
                "subsession.expiry": "expiry",
            }
        )
        .sort("round_number", "group_id", "id_in_group")
    )

    return data


def load_slider_data(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session.code") == session_code)
        .select(
            [
                "participant.code",
                "player.num_correct",
            ]
        )
        .rename(
            {
                "participant.code": "participant_code",
                "player.num_correct": "slider_performance",
            }
        )
    )

    return data


def load_survey_data_nonpersonal(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session.code") == session_code)
        .select(
            [
                "participant.id_in_session",
                "participant.code",
                "player.own_strategy",
                "player.other_players_strategy",
                "player.pilot_difficulty",
                "player.pilot_explanation",
                "player.pilot_interface",
                "player.pilot_time",
                "player.research_question",
                "player.comments",
                "player.dummy_player_axiom",
                "player.symmetry_axiom",
                "player.efficiency_axiom",
                "player.linearity_additivity_axiom",
                "player.linearity_HD1_axiom",
                "player.stability_axiom",
            ]
        )
        .rename(
            {
                "participant.id_in_session": "participant_id",
                "participant.code": "participant_code",
                "player.own_strategy": "own_strategy",
                "player.other_players_strategy": "other_players_strategy",
                "player.pilot_difficulty": "difficulty",
                "player.pilot_explanation": "explanation",
                "player.pilot_interface": "interface",
                "player.pilot_time": "enough_time",
                "player.research_question": "research_question",
                "player.comments": "comments",
                "player.dummy_player_axiom": "dummy_player_axiom",
                "player.symmetry_axiom": "symmetry_axiom",
                "player.efficiency_axiom": "efficiency_axiom",
                "player.linearity_additivity_axiom": "linearity_additivity_axiom",
                "player.linearity_HD1_axiom": "linearity_HD1_axiom",
                "player.stability_axiom": "stability_axiom",
            }
        )
    )

    return data


def load_survey_data_personal(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session.code") == session_code)
        .select(
            [
                "player.age",
                "player.gender",
                "player.gender_other",
                "player.degree",
                "player.degree_other",
                "player.study_field",
                "player.study_field_other",
                "player.nationality",
                "player.has_second_nationality",
                "player.second_nationality",
            ]
        )
        .rename(
            {
                "player.age": "age",
                "player.gender": "gender",
                "player.gender_other": "gender_if_other",
                "player.degree": "degree",
                "player.degree_other": "degree_if_other",
                "player.study_field": "study_field",
                "player.study_field_other": "study_field_if_other",
                "player.nationality": "nationality",
                "player.has_second_nationality": "has_secondnationality",
                "player.second_nationality": "second_nationality",
            }
        )
    )

    return data


def organize_chat_data(
    chat_data: pl.LazyFrame, bargaining_data: pl.LazyFrame
) -> pl.LazyFrame:
    data_tomerge = bargaining_data.select(
        [
            "participant_code",
            "round_number",
            "group_id",
            "id_in_group",
            "start_time",
        ]
    ).sort("participant_code", "start_time")

    merged_data = (
        chat_data.sort("participant_code", "timestamp")
        .join_asof(
            data_tomerge,
            by="participant_code",
            left_on="timestamp",
            right_on="start_time",
            strategy="backward",
        )
        .drop("start_time")
        .sort("round_number", "timestamp")
    )
    return merged_data


if __name__ == "__main__":
    session_details = load_session_details(
        snakemake.input.wide_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    chat_data_raw = load_chat_data(
        snakemake.input.chat_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    page_loads, proposals, acceptances = load_live_data(
        snakemake.input.live_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    bargaining_data = load_bargaining_data(
        snakemake.input.bargaining_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    slider_data = load_slider_data(
        snakemake.input.slider_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    survey_data_nonpersonal = load_survey_data_nonpersonal(
        snakemake.input.survey_data_nonpersonal,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    survey_data_personal = load_survey_data_personal(
        snakemake.input.survey_data_personal,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    chat_data = organize_chat_data(chat_data_raw, bargaining_data).collect()

    chat_data.write_csv(snakemake.output.chat)  # noqa F821 # type: ignore
    page_loads.sink_csv(snakemake.output.page_loads)  # noqa F821 # type: ignore
    proposals.sink_csv(snakemake.output.proposals)  # noqa F821 # type: ignore
    acceptances.sink_csv(snakemake.output.acceptances)  # noqa F821 # type: ignore
    bargaining_data.sink_csv(snakemake.output.bargaining_data)  # noqa F821 # type: ignore
    slider_data.sink_csv(snakemake.output.slider_data)  # noqa F821 # type: ignore
    survey_data_nonpersonal.sink_csv(snakemake.output.survey_data_nonpersonal)  # noqa F821 # type: ignore
    survey_data_personal.sink_csv(snakemake.output.survey_data_personal)  # noqa F821 # type: ignore

    with open(snakemake.output.session_details, "w") as file:  # noqa F821 # type: ignore
        json.dump(session_details, file, indent=4)
