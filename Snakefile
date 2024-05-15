SESSION_CODES = ["ykdzfw2h", "5r4374w0"]


rule concatenate_sessions:
    conda: "environment.yml"
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
    conda: "environment.yml"
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
    conda: "environment.yml"
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
    conda: "environment.yml"
    input:
        "src/power_analysis/power.ipynb",
    output:
        "out/power_analysis/power.html"
    shell:
        "jupyter nbconvert --execute --to html -- {input} --output {output} --output-dir ."
