[![checks](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/stanmart/unstructured-bargaining-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

# Unstructured bargaining analysis
Code for analyzing the results of the unstructured bargaining experiment

## Setup

The project is set up so that [`pixi`](https://pixi.sh/latest/) handles the installation of the required dependencies into a local virtual environment. First, install `pixi` as described in the [documentation](https://pixi.sh/latest/#installation). Then, you have the following options to run the analysis (all of them take care of downloading and installing all required dependencies):

 - `pixi run [TASK]` runs the pixi task `[TASK]`. For a list of available tasks, run `pixi task list`.
 - `pixi run [COMMAND]` runs the command `[COMMAND]` in the pixi environment. For example, `pixi run python` starts a Python shell in the pixi environment.
 - `pixi shell` starts a shell in the pixi environment. It is analogous to `conda activate`. Note, that there is no need to activate the environment before using the `run` command. Also, there is no `deactivate` command. To exit the shell, simply type `exit`.

Furthermore, data exported from otree must be placed into the `data/raw` directory. First, navigate to the `Data` tab in the otree admin interface and export the following files to the desired location. Always chose the plain format.

 - Per-app: live_bargaining (custom_export) → `data/raw/live_data.csv`
 - Per-app: live_bargaining → `data/raw/bargaining_data.csv`
 - Per-app: survey → `data/raw/survey_data.csv`
 - Per-app: sliders: → `data/raw/slider_data.csv`
 - Chat logs → `data/raw/chat_data.csv`

## Running the analysis

 - To create the collected datasets in `data/clean_collected`, run: `pixi run create_datasets`
 - To rerun the power analysis in `src/power_analysis/power.ipynb`, run `pixi run power_analysis`

## Automated checks

The project is set up with GitHub Actions to run automated checks on every push and pull request to the main branch. The checks include:
 - `ruff check` for Python code style
 - `ruff format` for Python code formatting
 - `codespell` for spell checking

These checks can also be run locally as pre-commit hooks. First, install [`pre-commit`](https://pre-commit.com/#install), e.g. using `pipx`:

```bash
pipx install pyright
pipx install pre-commit
```

Then, you can install the hooks by running
```bash
pre-commit install
```
After this, the checks will be performed automatically before every commit. If you want to run the checks manually, you can do so by running

```bash
pre-commit run --all-files
```
