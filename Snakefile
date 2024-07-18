from src.util.makeutils import find_quarto_images, find_opened_files


SESSION_CODES = ["ykdzfw2h", "5r4374w0", "v0bpsxm2", "m7xcm95f"]
WORD_TYPES = ["all", "VERB", "ADJ", "NOUN"]
PRESENTATIONS, *_ = glob_wildcards("src/presentation/{presentation}.qmd")


rule prepare_to_deploy:
    input:
        presentations = expand("out/presentation/{presentation}.html", presentation=PRESENTATIONS),
    output:
        presentations = expand("gh-pages/{presentation}.html", presentation=PRESENTATIONS),
        nojekyll = "gh-pages/.nojekyll"
    run:
        from shutil import copy2
        from pathlib import Path
        for file in input.presentations:
            copy2(file, "gh-pages")
        Path(output.nojekyll).touch()

rule presentations:
    input:
        expand("out/presentation/{presentation}.html", presentation=PRESENTATIONS)

rule blm_presentation:
    input:
        qmd = "src/presentation/blm.qmd",
        analysis_results = find_opened_files("src/presentation/blm.qmd"),
        images = find_quarto_images("src/presentation/blm.qmd"),
        bibliography = "src/paper/references.bib",
        css = "src/presentation/include/custom.scss",
        marhjax_js = "src/presentation/include/mathjax-settings.html",
        section_js = "src/presentation/include/sections-in-footer.html",
    output:
        "out/presentation/blm.html",
    shell:
        "quarto render {input.qmd}"


rule figures:
    input:
        "out/figures/payoff_scatterplot.pdf",
        "out/figures/payoff_average.pdf",
        "out/figures/payoff_by_agreement_type.pdf",
        "out/figures/payoff_share_of_agreement_types.pdf",
        "out/figures/payoff_share_of_agreement_types_by_round.pdf",
        "out/figures/payoff_equal_splits_by_round.pdf",
        "out/figures/payoff_matching_group_average.pdf",
        "out/figures/payoff_groups.pdf",
        "out/figures/timing_until_decision.pdf",
        "out/figures/timing_until_agreement_scatterplot.pdf",
        "out/figures/timing_until_agreement_by_round.pdf",
        "out/figures/proposal_number_per_round.pdf",
        "out/figures/proposal_gini.pdf",
        "out/figures/axioms_survey_efficiency.pdf",
        "out/figures/axioms_survey_symmetry.pdf",
        "out/figures/axioms_survey_dummy_player.pdf",
        "out/figures/axioms_survey_linearity_HD1.pdf",
        "out/figures/axioms_survey_linearity_additivity.pdf",
        "out/figures/axioms_survey_stability.pdf",
        "out/figures/axioms_outcomes_efficiency.pdf",
        "out/figures/axioms_outcomes_symmetry.pdf",
        "out/figures/axioms_outcomes_dummy_player.pdf",
        "out/figures/axioms_outcomes_linearity_additivity.pdf",
        "out/figures/axioms_outcomes_stability.pdf",
        "out/figures/plot_chat_topics_until_agreement.pdf",
        "out/figures/survey_difficulty_rating.pdf", 
        "out/figures/survey_age.pdf",
        "out/figures/survey_gender.pdf",
        "out/figures/survey_degree.pdf",
        "out/figures/survey_study_fields.svg",
        "out/figures/survey_nationality.pdf",
        "out/figures/values_theory.pdf",
        "out/figures/chat_excerpt-8,10,11,17-21.pdf",
        "out/figures/allocations_outcomes.pdf",
        "out/figures/allocations_proposals.pdf",


rule nonparametric_table:
    input:
        mann_whitney = "out/analysis/mann_whitney.json",
    output:
        table = "out/tables/nonparametric_table.tex",
    script:
        "src/tables/nonparametric_table.py"


rule regression_table:
    input:
        regression = "out/analysis/regression.pkl",
        regression_dummies = "out/analysis/regression_dummies.pkl",
    output:
        table = "out/tables/regression_table.tex",
    script:
        "src/tables/regression_table.py"

rule run_analysis: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv", 
    output: 
        summary = "out/analysis/analysis_results.txt",
        mann_whitney = "out/analysis/mann_whitney.json",
        regression = "out/analysis/regression.pkl",
        regression_dummies = "out/analysis/regression_dummies.pkl",
        mse = "out/analysis/mse.json",
        axiom_results =  "out/analysis/axiom_test_results.pkl",
    script: 
        "src/analysis/analysis.py"

rule create_chat_plot: 
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
        actions =  "data/clean/_collected/actions.csv",
        chat = "out/analysis/chat_classified.csv",
    output: 
        figure = "out/figures/plot_chat_{plot}.{ext}",
    script: 
        "src/figures/chat_plots.py"


rule classify_chat_messages:
    input:
        actions = "data/clean/_collected/actions.csv",
    output:
        chat_classified = "out/analysis/chat_classified.csv",
    params:
        cache_dir = "data/cached/chat_classified",
    script:
        "src/analysis/classify_chat.py"

rule create_values_theory_plot:
    output:
        figure = "out/figures/values_theory.{ext}",
    script:
        "src/figures/values_theory_plot.py"

rule create_chat_excerpt:
    input:
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/chat_excerpt-{rows}.{ext}",
    script:
        "src/figures/chat_excerpts.py"

rule create_survey_plot: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv",
    output: 
        figure = "out/figures/survey_{plot}.{ext}",
    script: 
        "src/figures/survey_plots.py"

rule create_axiom_survey_plot: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv",
    wildcard_constraints:
        ncol = r"\-?.*",
        axiom = "\w+"
    output: 
        figure = "out/figures/axioms_survey_{axiom}{ncol}.{ext}",
    script: 
        "src/figures/axiom_plots.py"

rule create_axiom_outcomes_plot: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv",
    output: 
        figure = "out/figures/axioms_outcomes_{axiom}.{ext}",
    script: 
        "src/figures/axiom_outcomes_plots.py"

rule create_proposal_plot:
    input:
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/proposal_{plot}.{ext}",
    script:
        "src/figures/proposal_plots.py"


rule create_timing_plot:
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/timing_{plot}.{ext}",
    script:
        "src/figures/timing_plots.py"


rule create_allocation_plot:
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/allocations_{type}.{ext}",
    script:
        "src/figures/allocation_scatterplots.py"


rule create_payoff_plot:
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
    output:
        figure = "out/figures/payoff_{plot}.{ext}",
    script:
        "src/figures/payoff_plots.py"


rule create_datasets:
    input:
        actions = "data/clean/_collected/actions.csv",
        outcomes = "data/clean/_collected/outcomes.csv",


rule concatenate_sessions:
    input:
        actions = expand("data/clean/session_{session_code}/actions.csv", session_code=SESSION_CODES),
        outcomes = expand("data/clean/session_{session_code}/outcomes.csv", session_code=SESSION_CODES),
        session_details = expand("data/clean/session_{session_code}/session_details.txt", session_code=SESSION_CODES),
    output:
        actions = "data/clean/_collected/actions.csv",
        outcomes = "data/clean/_collected/outcomes.csv",
    script:
        "src/data/concatenate_sessions.py"


rule merge_session_data:
    input:
        chat_data = "data/clean/session_{session_code}/chat.csv",
        acceptances = "data/clean/session_{session_code}/acceptances.csv",
        proposals = "data/clean/session_{session_code}/proposals.csv",
        bargaining_data = "data/clean/session_{session_code}/bargaining.csv",
        slider_data = "data/clean/session_{session_code}/slider_data.csv",
        survey_data = "data/clean/session_{session_code}/survey_data.csv",
    output:
        actions = "data/clean/session_{session_code}/actions.csv",
        outcomes = "data/clean/session_{session_code}/outcomes.csv",
    script:
        "src/data/merge_session_data.py"


rule collect_session_data:
    input:
        wide_data = "data/raw/wide_data.csv",
        bargaining_data = "data/raw/bargaining_data.csv",
        live_data = "data/raw/live_data.csv",
        chat_data = "data/raw/chat_data.csv",
        slider_data = "data/raw/slider_data.csv",
        survey_data = "data/raw/survey_data.csv",
    output:
        session_details = "data/clean/session_{session_code}/session_details.txt",
        chat = "data/clean/session_{session_code}/chat.csv",
        page_loads = "data/clean/session_{session_code}/page_loads.csv",
        proposals = "data/clean/session_{session_code}/proposals.csv",
        acceptances = "data/clean/session_{session_code}/acceptances.csv",
        bargaining_data = "data/clean/session_{session_code}/bargaining.csv",
        slider_data = "data/clean/session_{session_code}/slider_data.csv",
        survey_data = "data/clean/session_{session_code}/survey_data.csv",
    script:
        "src/data/collect_session_data.py"


rule power_analysis:
    input:
        "src/power_analysis/power.ipynb",
    output:
        "out/power_analysis/power.html"
    shell:
        "jupyter nbconvert --execute --to html -- {input} --output {output} --output-dir ."
