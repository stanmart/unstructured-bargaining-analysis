import polars as pl


def merge_action_data(
    chat_data_path: str,
    acceptances_path: str,
    proposals_path: str,
):
    chat_data = pl.scan_csv(chat_data_path).with_columns(action=pl.lit("chat"))
    acceptances = pl.scan_csv(acceptances_path).with_columns(
        action=pl.lit("acceptance")
    )
    proposals = pl.scan_csv(proposals_path).with_columns(action=pl.lit("proposal"))

    action_data = pl.concat([chat_data, acceptances, proposals], how="diagonal").sort(
        "round_number", "group_id", "timestamp"
    )
    first_columns = ["round_number", "group_id", "timestamp", "action"]
    colorder = first_columns + [
        col for col in action_data.columns if col not in first_columns
    ]

    return action_data.select(colorder)


if __name__ == "__main__":
    action_data = merge_action_data(
        snakemake.input.chat_data,  # noqa F821 # type: ignore
        snakemake.input.acceptances,  # noqa F821 # type: ignore
        snakemake.input.proposals,  # noqa F821 # type: ignore
    )

    action_data.sink_csv(snakemake.output.actions)  # noqa F821 # type: ignore
