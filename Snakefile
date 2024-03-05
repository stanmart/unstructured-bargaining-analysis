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
