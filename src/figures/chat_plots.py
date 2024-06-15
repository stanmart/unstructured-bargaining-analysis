import string

import polars as pl
import seaborn as sns


def count_words(
    df: pl.DataFrame, group_var: str, word_type: str | None = None
) -> pl.DataFrame:
    if word_type is not None:
        df = df.filter(pl.col("pos") == word_type)

    exclude_words = set(string.punctuation) | {"a", "a1", "a2", "b", "b1", "b2"}

    word_counts = (
        df.filter(
            pl.col("lemma").is_in(exclude_words).not_(),
            pl.col("lemma").str.contains(r"\d+$").not_(),
            pl.col("lemma").str.contains(r"id\d+$").not_(),
        )
        .group_by([group_var, "lemma"])
        .agg(
            count=pl.count("lemma"),
        )
    )

    freq_in_group = (
        word_counts.with_columns(
            total_count=pl.sum("count").over(group_var),
        )
        .with_columns(
            freq=pl.col("count") / pl.col("total_count"),
        )
        .pivot(values="freq", index="lemma", columns=group_var)
        .fill_null(0)
    )

    freq_total = (
        word_counts.group_by("lemma")
        .agg(
            count=pl.sum("count"),
        )
        .with_columns(
            total_count=pl.sum("count"),
        )
        .with_columns(
            freq_total=pl.col("count") / pl.col("total_count"),
        )
        .select("lemma", "freq_total")
    )

    freq_table = freq_in_group.join(freq_total, on="lemma", how="left")

    freq_table_long = pl.concat(
        [
            freq_table.with_columns(type=pl.lit("absolute")),
            freq_table.with_columns(
                (pl.all().exclude("lemma", "freq_total") / pl.col("freq_total")),
                type=pl.lit("relative"),
            ),
        ],
        how="diagonal",
    )

    return freq_table_long


def create_plot(
    df: pl.DataFrame,
    group_var: str,
    word_type: str | None = None,
    top_k: int = 5,
    total_freq_threshold: float = 0.002,
    type="relative",
) -> sns.FacetGrid:
    value_name = (
        "Relative frequency (group / total)"
        if type == "relative"
        else "Frequency (group / total)"
    )
    freq_table = (
        count_words(df, group_var, word_type)
        .filter(pl.col("freq_total") > total_freq_threshold, pl.col("type") == type)
        .select(pl.all().exclude("freq_total", "type"))
    )
    group_levels = [col for col in freq_table.columns if col not in ["lemma", "facet"]]
    plot_data = pl.concat(
        [
            freq_table.sort(level, descending=True)
            .head(top_k)
            .with_columns(facet=pl.lit(level))
            for level in group_levels
        ]
    ).melt(
        id_vars=["lemma", "facet"],
        value_vars=group_levels,
        value_name=value_name,
        variable_name="group_var",
    )

    word_type_nice = {
        "NOUN": "nouns",
        "VERB": "verbs",
        "ADJ": "adjectives",
        None: "words",
    }

    palette = {level: color for level, color in zip(group_levels, sns.color_palette())}

    g = sns.FacetGrid(
        plot_data,
        col="facet",
        sharey=False,
        legend_out=True,
        aspect=1.3,
        height=2.8,
        col_wrap=2,
    )
    g.map_dataframe(
        sns.barplot,
        x=value_name,
        hue="group_var",
        y="lemma",
        alpha=0.9,
        palette=palette,
    )

    g.set_ylabels("")
    g.add_legend(title="")
    g.set_titles(
        col_template=f"Top {top_k} {word_type_nice[word_type]} for '{{col_name}}'"
    )

    if type == "relative":
        for ax in g.axes.flat:
            ax.axvline(1, color="black", linestyle=":")

    return g


if __name__ == "__main__":
    df_processed = pl.read_csv(snakemake.input.lemmas)  # noqa F821 # type: ignore

    word_type = snakemake.wildcards.word_type  # noqa F821 # type: ignore
    if word_type == "all":
        word_type = None
    group_var = snakemake.wildcards.group_var  # noqa F821 # type: ignore
    dummy = snakemake.wildcards.dummy  # noqa F821 # type: ignore
    if dummy == "nodummy":
        df_processed = df_processed.filter(
            pl.col("treatment_name") != "treatment_dummy_player"
        )

    fig = create_plot(
        df_processed.filter(pl.col("treatment_name") != "treatment_dummy_player"),
        group_var=group_var,
        word_type=word_type,
        top_k=10,
        total_freq_threshold=0.002,
        type="relative",
    )

    fig.savefig(snakemake.output.figure, bbox_inches="tight")  # noqa F821 # type: ignore
