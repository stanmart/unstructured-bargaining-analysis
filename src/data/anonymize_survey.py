import polars as pl

PERSONAL_COLS = [
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


def split_survey(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    personal = raw.select("session.code", *PERSONAL_COLS)
    non_personal = raw.drop(*PERSONAL_COLS)

    return personal, non_personal


def reshuffle_personal(personal: pl.DataFrame) -> pl.DataFrame:
    return personal.with_columns(
        pl.col("*").shuffle().over("session.code"),
    )


if __name__ == "__main__":
    raw = pl.read_csv("data/raw/survey_data.csv")
    personal, non_personal = split_survey(raw)
    personal_reshuffled = reshuffle_personal(personal)
    personal_reshuffled.write_csv("data/raw/survey_data_personal.csv")
    non_personal.write_csv("data/raw/survey_data_nonpersonal.csv")

    wide = pl.read_csv("data/raw/wide_data.csv").drop(
        "survey.1." + col for col in PERSONAL_COLS
    )
    wide.write_csv("data/raw/wide_data_nonpersonal.csv")
