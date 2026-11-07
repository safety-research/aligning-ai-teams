# Sepsis intervention analysis

Aggregates per-run results from the sepsis experiment and produces the
analysis figures used in the paper.

## Pipeline

```
results/                                    aggregate_results.py
{model}/{N}_agent_experiments/    ────────►  all_results.csv
{org}/{run}/...                              (one row per run)
                                                  │
                                                  ▼
                                           walkthrough.ipynb  (uses analysis.py)
                                                  │
                                                  ▼
                                           ../figures/*.png
```

`aggregate_results.py` walks the results tree, reads each run's
`experiment_results.json` (test-set metrics) plus the run's SQLite DB
(outcome, duration, cost, etc.), and writes `all_results.csv`.
`walkthrough.ipynb` consumes that CSV and produces all the figures by
calling helpers in `analysis.py`.

## Files

| File | Purpose |
|---|---|
| `aggregate_results.py` | Walk results tree → produce `all_results.csv`. Run as a standalone script when results change. |
| `analysis.py` | All analysis helpers: data loading + filtering, per-cell stats, plotting functions. Imported by `walkthrough.ipynb`. |
| `walkthrough.ipynb` | The analysis notebook. Thin: each section is a one-liner that calls into `analysis.py` and renders a figure. Read this to understand the flow. |
| `all_results.csv` | Output of `aggregate_results.py`. Source of truth for everything in `walkthrough.ipynb`. |

## Figures

All figures land in `../figures/` (the shared `analysis/figures/` folder, also used by the DoR pipeline).

- **Headline (paper)**: `safety_summary.png` — cross-model bars showing curated multi-agent interventions vs single-agent baseline. Top row is mean Δ sensitivity (Welch CI); bottom row is 10th-percentile Δ sensitivity (BCa bootstrap CI).
- **Per-model mean (appendix)**: `mean_<model>.png` — family-faceted absolute sensitivity + normalized savings, mean with Welch CI, sig vs single agent.
- **Per-model worst-case (appendix)**: `worst_<model>.png` — same family-faceted layout but bar values are the 10th-percentile of each metric, with BCa bootstrap 95% CIs.
- **Cost-benefit**: `cost_benefit.png` — per-cell mean API cost (× model's multi-baseline) vs Δ sensitivity vs multi-baseline, with per-model Pareto frontiers.
- **Per-(model, org) scatters**: `<model>/<org>.png` — miss-rate vs avg-cost on the test set, with single-agent overlay, perfect-classifier frontier, and a random-classifier band.

## Running

```bash
# 1. Re-aggregate the results CSV from the on-disk results tree
uv run python aggregate_results.py

# 2. Open walkthrough.ipynb and run cells. Figures land in ../figures/.
```

The two `plot_worst` / `plot_safety_bars_vs_single` cells in the
walkthrough do BCa bootstraps with `B=10000` and take ~30s per model —
expect a few minutes total on the first run. Cells fail safely if the
CSV is missing or empty.

## Default analysis filter

`load_and_annotate` defaults to:
- `exclude={'outcome': ['timeout', 'failure']}` — drop runs that timed out or crashed
- `require_eval=True` — drop runs missing `experiment_results.json`
- `min_n=3` — drop (model, org) cells with fewer than 3 surviving runs

`build_outcome_table` (walkthrough §2b) shows pre-filter outcome counts so
the dropped fraction is visible.
