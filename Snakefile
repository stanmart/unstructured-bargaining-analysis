rule create_action_data:
    conda: "environment.yml"
    input:
        chat_data = "data/clean/session_{session_code}/chat.csv",
        acceptances = "data/clean/session_{session_code}/acceptances.csv",
        proposals = "data/clean/session_{session_code}/proposals.csv",
    output:
        actions = "data/clean/session_{session_code}/actions.csv",
    script:
        "src/data/create_action_data.py"



rule create_session_data:
    conda: "environment.yml"
    input:
        bargaining_data = "data/raw/bargaining_data.csv",
        live_data = "data/raw/live_data.csv",
        chat_data = "data/raw/chat_data.csv",
        survey_data = "data/raw/survey_data.csv"
    output:
        chat = "data/clean/session_{session_code}/chat.csv",
        page_loads = "data/clean/session_{session_code}/page_loads.csv",
        proposals = "data/clean/session_{session_code}/proposals.csv",
        acceptances = "data/clean/session_{session_code}/acceptances.csv",
        bargaining_data = "data/clean/session_{session_code}/bargaining_outcomes.csv",
        survey_data = "data/clean/session_{session_code}/survey_data.csv"
    script:
        "src/data/create_session_data.py"


rule power_analysis:
    conda: "environment.yml"
    input:
        "src/power_analysis/power.ipynb",
    output:
        "out/power_analysis/power.html"
    shell:
        "jupyter nbconvert --execute --to html -- {input} --output {output} --output-dir ."
