[![checks](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/ci.yml)
[![publish](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/publish.yml/badge.svg)](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

# Unstructured bargaining analysis
Code for analyzing the results of the unstructured bargaining experiment

## Setup

### Software requirements

The project is set up so that [pixi](https://pixi.sh/latest/) handles the installation of the required dependencies into a local virtual environment. First, install pixi as described in the [documentation](https://pixi.sh/latest/#installation). Then, you can run the commands as described in the [running the analysis](#running-the-analysis) section.

<details>
<summary>Using pixi</summary>
With pixi installed, you have three main commands at your disposal:

 - `pixi run [TASK]` runs the pixi task `[TASK]`. For a list of available tasks, run `pixi task list`.
 - `pixi run [COMMAND]` runs the command `[COMMAND]` in the pixi environment. For example, `pixi run python` starts a Python shell in the pixi environment.
 - `pixi shell` starts a shell in the pixi environment. It is analogous to `conda activate`. Note, that there is no need to activate the environment before using the `run` command. Also, there is no `deactivate` command. To exit the shell, simply type `exit`.

All of these commands take care of setting up the virtual environment and installing the required dependencies. If you want to add a new dependency, simply run `pixi add [PACKAGE]=[VERSION]`. It will then be added to the `pixi.toml` file and installed in the virtual environment.

Pixi also uses a lockfile. This lockfile is updated automatically when you add a new dependency. If you want to update the lockfile manually, you can delete the `pixi.lock` file and run `pixi install` to recreate it. The lockfile should normally be committed to the repository to make sure that everyone uses the same versions of the dependencies.
</details>

### Data requirements

Data exported from otree is already placed into the `data/raw` directory. Please do not modify it under any circumstances. Automated checks are in place to ensure that the data is not modified.

<details>
<summary>Exporting data from otree</summary>
The data in the `data/raw` directory is exported from the otree admin interface. The following steps describe how it can be obtained.
First, navigate to the `Data` tab in the otree admin interface and export the following files to the indicated location. Always chose the plain format.

 - All apps → `data/raw/wide_data.csv`
 - Per-app: live_bargaining (custom_export) → `data/raw/live_data.csv`
 - Per app: introduction → `data/raw/intro_data.csv`
 - Per-app: live_bargaining → `data/raw/bargaining_data.csv`
 - Per-app: survey → `data/raw/survey_data.csv`
 - Per-app: sliders: → `data/raw/slider_data.csv`
 - Chat logs → `data/raw/chat_data.csv`
 - Page times → `data/raw/page_time_data.csv`

Checksums for the raw data are stored in the `RAW_DATA_CHECKSUMS` repository variable.
</details>

## Running the analysis

 - To create the collected datasets in `data/clean_collected`, run: `pixi run create-datasets`
 - To rerun the power analysis at `src/power_analysis/power.ipynb`, run `pixi run power-analysis`
 - To create a chart of the analysis steps at `build-graphs/filegraph.svg`, run `pixi run filegraph` (or similar commands for the `dag` or the `rulegraph`)

## Automated checks

The project is set up with GitHub Actions to run automated checks on every push and pull request to the main branch. The checks include:
 - `ruff check` for Python code style
 - `ruff format` for Python code formatting
 - `codespell` for spell checking
 - `pyright` for static type checking
 - Data integrity checks for the raw data based on the `RAW_DATA_CHECKSUMS` repository variable

These checks can also be run locally using `pixi`. Simply run `pixi run checks` to run all checks. If you only want to run a specific check, you can do so by running `pixi run [CHECK]`, where `[CHECK]` is one of `lint`, `format`, `spellcheck`, `typecheck` or `data-check`.

Alternatively, `ruff` and `codespell` checks can also be installed as pre-commit hooks. First, install [`pre-commit`](https://pre-commit.com/#install), e.g. using `pipx`:

```bash
pipx install pre-commit
```

Then, you can install the hooks by running
```bash
pre-commit install
```
After this, the checks will be performed automatically before every commit.