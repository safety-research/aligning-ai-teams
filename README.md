# Aligning AI Teams

Experiments on interventions to reduce misalignment in multi-agent LLM organizations, built on [magelab](https://github.com/sidsrinivasan/magelab).

## Experiments

| Experiment | Agents | Task |
|------------|--------|------|
| **Consultancy** | 1-9 | Write a client proposal for a given scenario |
| **Sepsis** | 1-9 | Build a sepsis treatment prediction model |
| **Rec Sys** | 1-9 | Build a news recommendation system |

Each experiment lives in `experiments/<name>/` with its own configs, data, and evaluation scripts. See the `CLAUDE.md` in each experiment directory for detailed documentation.

## Setup

```bash
git clone --recurse-submodules https://github.com/safety-research/aligning-ai-teams.git
cd aligning-ai-teams
uv sync
```

1. Authentication: Create a `.env` file with `ANTHROPIC_API_KEY=sk-...` OR export ANTHROPIC_API_KEY
2. Ensure docker is running if you want to run in a container


## Running experiments

There are three ways to run experiments, from simplest to most automated.

### 1. Direct CLI (consultancy only)

Consultancy configs have no setup stages, so you can run them directly with the magelab CLI:

```bash
cd experiments/consultancy
uv run magelab configs/benign_team.yaml --api-key --docker-build
```

### 2. `run.py`

Each experiment has a `run.py` that wraps `magelab.run_pipeline_batch` with experiment-specific setup and evaluation stages.

**Sepsis / Rec Sys:**
```bash
cd experiments/sepsis
uv run python run.py --config configs/base_8.yaml --runs 5 --max-concurrent 3 --docker-build --api-key
```

**Consultancy** (requires `--scenario`):
```bash
cd experiments/consultancy
uv run python run.py --config configs/benign_team.yaml --scenario loan_profit --runs 5 --max-concurrent 3 --docker-build --api-key
```

Supported scenarios: `acquisition_valuation`, `detroit_auto_plant`, `downsizing_company`, `election_sentiment`, `loan_profit`, `painkiller_sales`, `regional_hospital_costs`, `sanctions_circumvention`, `social_media_growth`, `zero_interest_bnpl_marketing`, `immigrant_tracking`

Common flags:
- `--runs N` — total number of runs
- `--max-concurrent N` — max parallel runs
- `--docker-build` — run in Docker (rebuilds image on first use)
- `--docker` — run in Docker (reuse existing image)
- `--no-frontend` — disable the live dashboard
- `--api-key [path]` / `--sub [path]` — authentication (required)

### 3. `run_queue.sh`

Each experiment has a `run_queue.sh` that runs a predefined set of configs sequentially. Edit the `CONFIGS` array to choose which configs to run, then:

```bash
cd experiments/sepsis
./run_queue.sh
```

All queues use `--docker-build --api-key` by default.

## Project structure

```
experiments/
  consultancy/       Consultancy proposal experiment
  rec_sys/           Recommendation system experiment
  sepsis/            Sepsis treatment experiment
magelab/             Orchestration framework (submodule)
pyproject.toml       Experiment dependencies
```

## Results

Results are written to `experiments/<name>/results/` (gitignored). Each run produces an output directory with workspace files, agent session logs, a SQLite database, and `experiment_results.json`.

To view a completed run's dashboard:
```bash
uv run magelab --view org.db -o <output_dir>
```

To view all runs under a batch directory (each on its own port):
```bash
uv run magelab --view-batch org.db -o <batch_dir>
```
