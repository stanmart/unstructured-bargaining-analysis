import polars as pl
import polars.selectors as cs


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
            "offer_id",
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


def load_survey_data(path: str, session_code: str) -> pl.LazyFrame:
    data = (
        pl.scan_csv(path)
        .filter(pl.col("session.code") == session_code)
        .select(
            [
                "participant.code",
                "player.age",
                "player.gender",
                "player.degree",
                "player.study_field",
                "player.nationality",
                "player.reflection",
                "player.pilot_difficulty",
                "player.pilot_explanation",
                "player.pilot_interface",
                "player.pilot_time",
                "player.comments",
            ]
        )
        .rename(
            {
                "participant.code": "participant_code",
                "player.age": "age",
                "player.gender": "gender",
                "player.degree": "degree",
                "player.study_field": "study_field",
                "player.nationality": "nationality",
                "player.reflection": "reflection",
                "player.pilot_difficulty": "pilot_difficulty",
                "player.pilot_explanation": "pilot_explanation",
                "player.pilot_interface": "pilot_interface",
                "player.pilot_time": "pilot_time",
                "player.comments": "comments",
            }
        )
    )

    return data


def organize_chat_data(
    chat_data: pl.LazyFrame, bargaining_data: pl.LazyFrame
) -> pl.LazyFrame:
    # TODO
    return chat_data


if __name__ == "__main__":
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
    survey_data = load_survey_data(
        snakemake.input.survey_data,  # noqa F821 # type: ignore
        snakemake.wildcards.session_code,  # noqa F821 # type: ignore
    )
    chat_data = organize_chat_data(chat_data_raw, bargaining_data)

    chat_data.sink_csv(snakemake.output.chat)  # noqa F821 # type: ignore
    page_loads.sink_csv(snakemake.output.page_loads)  # noqa F821 # type: ignore
    proposals.sink_csv(snakemake.output.proposals)  # noqa F821 # type: ignore
    acceptances.sink_csv(snakemake.output.acceptances)  # noqa F821 # type: ignore
    bargaining_data.sink_csv(snakemake.output.bargaining_data)  # noqa F821 # type: ignore
    survey_data.sink_csv(snakemake.output.survey_data)  # noqa F821 # type: ignore
