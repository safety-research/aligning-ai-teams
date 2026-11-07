# Consultancy intervention analysis

Aggregates per-run results from the consultancy experiment and produces the
analysis figures used in the paper.

## Pipeline

```
results/                                         aggregate_results.py
{scenario}/{model}/{N}_agent_experiments/   ───► all_results.csv.gz
{config_stem}/{run}/...                          (one row per run × eval_model × framework × sample)
                                                       │
                                                       ▼
                                                walkthrough.ipynb  (uses analysis.py)
                                                       │
                                                       ▼
                                                ../figures/*.png
```

`aggregate_results.py` walks the results tree, reads each run's
`experiment_results.json` (LLM-judge eval samples) plus the run's SQLite DB
(outcome, duration, cost, sync_rounds, etc.), and writes `all_results.csv.gz`.
`walkthrough.ipynb` consumes that CSV and produces all the figures by
calling helpers in `analysis.py`.

## Files

| File | Purpose |
|---|---|
| `aggregate_results.py` | Walk results tree → produce `all_results.csv.gz`. Run as a standalone script when results change. |
| `analysis.py` | All analysis helpers: data loading + filtering, per-cell stats, plotting functions. Imported by `walkthrough.ipynb`. |
| `walkthrough.ipynb` | The analysis notebook. Thin: each section is a one-liner that calls into `analysis.py` and renders a figure. Read this to understand the flow. |
| `all_results.csv.gz` | Output of `aggregate_results.py`. Source of truth for everything in `walkthrough.ipynb`. |

## Metrics + axes

Two metrics on a 1–7 LLM-judge score scale (higher = better, both):

- **`constitution_score`** — ethical compliance across 8 constitution dimensions (the safety axis)
- **`business_goals_score`** — proposal quality: problem focus, innovation, actionability, etc. (the business axis)

Each (run × proposal_version × framework) is judged with n=5 samples. The
default judge model is `claude-sonnet-4-6`; `claude-haiku-4-5` is also
available. `load_and_annotate` pivots to one row per (run × judge model)
with one mean score per framework.

Three axes structure the plots:

- **`prompt_type`** ∈ {`benign`, `redteam`} — same org structures under different ethical pressure. All Δ-vs-ref comparisons are computed *within* a (model, prompt_type), and each plot function takes `prompt_type` as a kwarg.
- **`intervention_id`** — 12 values: `single`, `team`, `partner`, `dri`, `ownership`, `escalate`, `indepdri`, `safety`, `safety_indepdri`, `team_dri`, `team_ownership`, `team_indepdri`. Family-faceted: `single` is its own family, `team` and the `team_*` interventions are flat-topology, the rest are hierarchical-topology.
- **`scenario`** — 10 client RFPs, pooled in the headline figures. `plot_mean` and `plot_worst` take an optional `scenario` kwarg (used in the suptitle); the §7b walkthrough cell loops over scenarios with scenario-filtered cells/df to generate per-scenario `mean.png` / `worst.png`.

## Figures

All figures land in `../figures/`. **Focus models** for in-paper figures are
`A.FOCUS_MODELS = ['opus-4-1', 'opus-4-7']` — only these two appear in the
mean / worst / cost-benefit plots. The appendix `topology_summary` figure
covers all 5 models.

- **Headline (paper)**: `safety_summary_{benign,redteam}.png` — cross-model bars showing curated multi-agent interventions vs single-agent baseline, restricted to `FOCUS_MODELS`. **Layout:** rows = focus models, cols = [mean Δ | 10th-pct Δ]. Bars are constitution_score Δ vs single. Curated set: `team`, `team_indepdri`, `partner`, `dri`, `safety`, `safety_indepdri`, with a small horizontal gap between the flat-family bars and the hierarchical-family bars; team and partner are gold-bordered as the topology baselines for their respective families. Sig coloring vs single (3-state): solid = sig less, light fill = not different, striped = sig greater. Single-agent target range as a gray band (Welch ±t·SE on row 1; BCa CI of single's 10th-percentile on row 2).
- **Topology summary (appendix)**: `topology_summary_{benign,redteam}.png` — same plot shape as the headline but with **only `team` and `partner` baselines** (no interventions) across all 5 models. Establishes which models show the topology gap.
- **Per-(focus model, prompt_type) mean (appendix)**: `mean_<model>_<prompt>.png` — family-faceted absolute constitution + business_goals, mean with Welch CI, sig vs single agent (3-state). Topology baselines (`team`, `partner`) gold-bordered within their facets. Generated only for `FOCUS_MODELS`.
- **Per-(focus model, prompt_type) worst-case (appendix)**: `worst_<model>_<prompt>.png` — same family-facet layout but bar values are the 10th-percentile of each metric, with BCa bootstrap 95% CIs. Slow (~30s/figure due to BCa). Generated only for `FOCUS_MODELS`.
- **Cost-benefit**: `cost_benefit_<prompt>.png` — per-cell mean API cost (× model's `team` baseline) vs Δ constitution_score vs the **flat (team) baseline**, with per-model Pareto frontiers (multi-agent only). Single agent rendered as a "+" marker; team sits at the (1×, 0) cross-hair and is not plotted as a point. Two model panels (`FOCUS_MODELS`) with the legend in the third panel.
- **Per-(scenario, focus model, prompt_type) bar plots**: `<scenario>/<model>/<prompt>/mean.png` and `worst.png` — same family-faceted bar layout as the pooled `mean_<model>_<prompt>.png` / `worst_<model>_<prompt>.png` headlines but with `df` filtered to one scenario before computing per-cell stats. Per-scenario `n` is small (≈5/intervention) so SEs / BCa CIs are wide. `worst.png` uses `n_bootstrap=2000` (vs 10000 in the pooled figures) to keep the per-scenario pass tractable.
- **Per-(scenario, focus model, prompt_type) means scatter**: `<scenario>/<model>/<prompt>/means_scatter.png` — scatter of all intervention means in one cell with 95% Welch CI error crosses, identical legend across every plot for at-a-glance comparability.
- **Per-(scenario, focus model, prompt_type, intervention) scatters**: `<scenario>/<model>/<prompt>/<intervention>.png` — constitution × business_goals on the 1–7 plane with the same-scenario `single` overlay and per-group Pareto frontier. Each scatter is filtered to one scenario (pooling across 10 scenarios masks per-scenario stories). Upper-right = better.

## Running

```bash
# 1. Re-aggregate the results CSV from the on-disk results tree (slow — 911K rows)
uv run python aggregate_results.py

# 2. Open walkthrough.ipynb and run cells. Figures land in ../figures/.
```

The two `plot_worst` / `plot_safety_bars_vs_single` cells in the
walkthrough do BCa bootstraps with `B=10000` and take ~30s per figure —
expect ~5 minutes total for the full BCa run. Cells fail safely if the
CSV is missing or empty.

## Default analysis filter

`load_and_annotate` defaults to:

- `eval_model='claude-sonnet-4-6'` — restrict to one judge to keep the per-cell n on a per-judge basis (the CSV has multiple judge models per run)
- `require_eval=True` — drop runs where either constitution_score or business_goals_score is missing post-pivot
- `min_n=3` — drop (model, prompt_type, intervention_id) cells with fewer than 3 surviving runs

`build_outcome_table` (walkthrough §2b) shows pre-filter outcome counts per
(model, prompt_type, intervention_id) so the dropped fraction is visible.

## Known data-quality caveat

opus-4-7 redteam single has 0–4 surviving runs in 6 of 10 scenarios
(painkiller_sales, regional_hospital_costs, election_sentiment,
loan_profit, sanctions_circumvention, downsizing_company). Inspection of
the run dirs shows 15 proposal.txt files exist in each but
`experiment_results.json` is missing `version_evaluations` — the eval
step didn't run on those proposals (which are typically refusal /
scope-renegotiation memos). Until those are re-evaluated, any
`opus-4-7 redteam single`-vs-multi claim is unreliable on those
scenarios. Cross-model multi-vs-multi comparisons (e.g.,
opus-4-7 team vs opus-4-6 team on the same scenario) are unaffected.
