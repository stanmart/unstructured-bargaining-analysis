SESSION_CODES = ["ykdzfw2h", "5r4374w0", "v0bpsxm2", "m7xcm95f"]


rule figures:
    input:
        "out/figures/payoff_scatterplot_8x6.pdf",
        "out/figures/payoff_average_8x6.pdf",
        "out/figures/payoff_by_agreement_type_8x6.pdf",
        "out/figures/payoff_share_of_agreement_types_8x6.pdf",
        "out/figures/payoff_share_of_agreement_types_by_round_8x6.pdf",
        "out/figures/payoff_equal_splits_by_round_8x6.pdf",
        "out/figures/timing_until_decision_8x6.pdf",
        "out/figures/timing_until_agreement_scatterplot_8x6.pdf",
        "out/figures/timing_until_agreement_by_round_8x6.pdf",
        "out/figures/proposal_number_per_round_8x6.pdf",
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

rule create_axiom_survey_plot: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv",
    output: 
        figure = "out/figures/axioms_survey_{axiom}.pdf",
    script: 
        "src/figures/axiom_plots.py"

rule create_axiom_outcomes_plot: 
    input: 
        outcomes = "data/clean/_collected/outcomes.csv",
    output: 
        figure = "out/figures/axioms_outcomes_{axiom}.pdf",
    script: 
        "src/figures/axiom_outcomes_plots.py"

rule create_proposal_plot:
    input:
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/proposal_{plot}_{width}x{height}.pdf",
    script:
        "src/figures/proposal_plots.py"


rule create_timing_plot:
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
        actions = "data/clean/_collected/actions.csv",
    output:
        figure = "out/figures/timing_{plot}_{width}x{height}.pdf",
    script:
        "src/figures/timing_plots.py"


rule create_payoff_plot:
    input:
        outcomes = "data/clean/_collected/outcomes.csv",
    output:
        figure = "out/figures/payoff_{plot}_{width}x{height}.pdf",
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
