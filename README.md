# Unstructured bargaining analysis
Code for analyzing the results of the unstructured bargaining experiment

## Setup

The project is set up so that snakemake handles the installation of the required dependencies into a local virtual environment. Snakemake and conda/mamba/micromamba are the only dependencies that need to be installed on the host system.

 - Install your favorite conda package manager (e.g. conda, mamba, micromamba).
 - It is recommended to install snakemake in its own separate conda virtual environment (e.g. `conda create -c conda-forge -c bioconda -n snakemake snakemake`).

The steps to build the project are described in its snakemake file. If snakemake is installed it can be compiled from scratch by running the snakemake command in its root directory:

Furthermore, data exported from otree must be placed into the `data/raw` directory. First, navigate to the `Data` tab in the otree admin interface and export the following files to the desired location. Always chose the plain format.

 - Per-app: live_bargaining (custom_export) → `data/raw/live_data.csv`
 - Per-app: live_bargaining → `data/raw/bargaining_data.csv`
 - Per-app: survey → `data/raw/survey_data.csv`
 - Chat logs → `data/raw/chat_data.csv`

## Running the analysis

To organize the data for a given session with code `session_code`, simply tell snakemake that you'd like to get one of the target files (the others will be generated automatically):

```bash
snakemake --use-conda --cores 1 data/clean/session_p15obeom/chat.csv
```
