"""Analysis helpers for the consultancy walkthrough notebook.

Mirrors the sepsis / rec_sys interventions analysis, adapted for the
consultancy task's specifics:
  - Two metrics on the 1–7 LLM-judge score scale: `business_goals_score`
    (proposal quality) and `constitution_score` (ethical compliance), with
    n=5 evaluation samples per (run × proposal_version × framework) which
    `load_and_annotate` averages to one value per (run × framework).
  - Two prompt types per intervention: `benign` and `redteam` are the *same*
    org structure under different ethical pressure. All Δ-vs-ref comparisons
    are computed *within* a (model, prompt_type) so the two pressures stay
    separate; plot functions take `prompt_type` as a kwarg and emit one
    figure per pressure.
  - Three structural baselines: `single` (1 agent, universal anchor), `team`
    (flat 11-agent multi), `partner` (hierarchical 12-agent multi). The
    flat-family interventions reference `team`; the hierarchical-family
    interventions reference `partner`. Δ-vs-single is the bar value on the
    headline figure; per-family baselines are shown gold-bordered within
    their respective family facets.
  - 10 scenarios are pooled in the headline / per-(model, prompt) figures.

Data pipeline
-------------
load_and_annotate(...)             -> per-run df (one row per run × judge model)
compute_config_stats(...)          -> per-(model, prompt_type, intervention) cells
build_summary_table(...)           -> rich summary
build_outcome_table(...)           -> outcome counts (pre-filter)

Plots
-----
plot_safety_bars_vs_single(cells, df, prompt_type, out_path)
                                   -> headline: rows=models, cols=[mean Δ, 10th-pct Δ]
                                      Constitution score Δ vs single, curated bars.
plot_mean(cells, model, prompt_type, out_path)
                                   -> per-(model, prompt) family-faceted bars,
                                      two metric rows (constitution + business_goals).
plot_worst(df, model, prompt_type, out_path)
                                   -> per-(model, prompt) family-faceted worst-10% (BCa).
plot_cost_benefit(df, cells, prompt_type, out_path)
                                   -> faceted cost-benefit Pareto frontier per prompt.
plot_scatter(df, model, prompt_type, intervention_id, out_path)
                                   -> per-(model, prompt, intervention) constitution × business_goals scatter.
"""

import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _stats


# ── Canonical metric list and constants ──────────────────────────────────
METRICS = ["business_goals_score", "constitution_score"]
SCORE_RANGE = (1, 7)

FAMILY_ORDER = ["single", "flat", "hierarchical"]
FAMILY_DISPLAY = {
    "single":       "Single-agent",
    "hierarchical": "Hierarchical topology",
    "flat":         "Flat topology",
}

# intervention_id → (family, structure)
INTERVENTION_FAMILY = {
    "single":           ("single",       "single"),
    "partner":          ("hierarchical", "partner"),
    "dri":              ("hierarchical", "partner"),
    "ownership":        ("hierarchical", "partner"),
    "escalate":         ("hierarchical", "partner"),
    "indepdri":         ("hierarchical", "partner"),
    "safety":           ("hierarchical", "partner"),
    "safety_indepdri":  ("hierarchical", "partner"),
    "team":             ("flat",         "team"),
    "team_dri":         ("flat",         "team"),
    "team_ownership":   ("flat",         "team"),
    "team_indepdri":    ("flat",         "team"),
}
# Within-family ordering: baselines first, then by responsibility locus / intervention type
INTERVENTION_RANK = {
    "single":           -1,  # baseline of single family
    "partner":          -1,  # baseline of hierarchical family
    "dri":              0,   # designated (existing)
    "ownership":        1,   # distributed
    "escalate":         2,   # centralized (chain)
    "indepdri":         3,   # independent designated (new dedicated)
    "safety":           4,   # safety doc
    "safety_indepdri":  5,   # compound
    "team":             -1,  # baseline of flat family
    "team_dri":         0,
    "team_ownership":   1,
    "team_indepdri":    2,
}

CLEAN_LABEL = {
    "single":           "baseline",
    "partner":          "baseline",
    "team":             "baseline",
    "dri":              "embedded ethics lead",
    "ownership":        "distributed resp.",
    "escalate":         "centralized resp.",
    "indepdri":         "ind. ethics lead",
    "safety":           "safety doc",
    "safety_indepdri":  "safety doc + ind. ethics lead",
    "team_dri":         "embedded ethics lead",
    "team_ownership":   "distributed resp.",
    "team_indepdri":    "ind. ethics lead",
}

# Multi-line label for in-panel x-ticks (plot_mean / plot_worst). Used because
# horizontal labels need to fit ~1 inch per bar; long single-line labels
# overlap. Headline plot has its own label dict in plot_safety_bars_vs_single.
BAR_LABEL = {
    "single":           "baseline",
    "partner":          "baseline",
    "team":             "baseline",
    "dri":              "embedded\nethics lead",
    "ownership":        "distributed\nresp.",
    "escalate":         "centralized\nresp.",
    "indepdri":         "ind.\nethics lead",
    "safety":           "safety\ndoc",
    "safety_indepdri":  "safety doc\n+ ind.\nethics lead",
    "team_dri":         "embedded\nethics lead",
    "team_ownership":   "distributed\nresp.",
    "team_indepdri":    "ind.\nethics lead",
}

MODEL_COLOR = {  # Tol vibrant — CVD-safe categorical palette
    "opus-4-1":       "#0077BB",
    "sonnet-4-6":     "#EE7733",
    "opus-4-6":       "#CC3311",
    "opus-4-7":       "#009988",
    "mythos-preview": "#AA3377",
}
MODEL_ORDER = ["opus-4-1", "sonnet-4-6", "opus-4-6", "opus-4-7", "mythos-preview"]
# Two models that show real intervention effects worth featuring.
# Used as default in plot_mean / plot_worst / plot_safety_bars_vs_single /
# plot_cost_benefit. plot_topology_summary deliberately ignores this and
# shows all models as the appendix cross-model effect-size summary.
FOCUS_MODELS = ["opus-4-1", "opus-4-7"]
INTERVENTION_CODE = {
    "single":          "S",
    "team":            "Tb",
    "team_dri":        "Td",
    "team_ownership":  "To",
    "team_indepdri":   "Ti",
    "partner":         "Pb",
    "dri":             "Dr",
    "ownership":       "Ow",
    "escalate":        "Es",
    "indepdri":        "Id",
    "safety":          "Sd",
    "safety_indepdri": "",  # star marker is unique; no inside text
}

# Consistent visual identity for each intervention, identical legend across every
# per-cell mean scatter (plot_means_per_cell). Mirrors the (color, marker)-rotation
# style used in explore.ipynb but with a CB-safe categorical palette
# (Okabe-Ito 8 + Tol vibrant 4) paired with 10 distinct marker shapes.
LEGEND_ORDER = [
    "single",
    "team", "team_dri", "team_ownership", "team_indepdri",
    "partner", "dri", "ownership", "escalate", "indepdri", "safety", "safety_indepdri",
]
_PALETTE = [
    "#000000",  # black             — single
    "#0072B2",  # blue              — team
    "#56B4E9",  # sky blue          — team_dri
    "#009E73",  # bluish green      — team_ownership
    "#44AA99",  # teal              — team_indepdri
    "#D55E00",  # vermillion        — partner
    "#E69F00",  # orange            — dri
    "#CC79A7",  # reddish purple    — ownership
    "#882255",  # dark red          — escalate
    "#F0E442",  # yellow            — indepdri
    "#117733",  # dark green        — safety
    "#6699CC",  # light blue        — safety_indepdri
]
_MARKERS = ["o", "^", "s", "D", "v", "p", "h", "*", "X", "P", "<", ">"]
INTERVENTION_STYLE = {
    iid: {"color": _PALETTE[i],
          "marker": _MARKERS[i % len(_MARKERS)],
          "size": 220}
    for i, iid in enumerate(LEGEND_ORDER)
}
# Disambiguated labels for the per-cell scatter legend. CLEAN_LABEL collapses
# topology baselines to "baseline" (correct in their own family-faceted plots)
# but here we have all 12 interventions on one figure, so qualify them.
LEGEND_LABEL = {
    "single":           "single agent",
    "team":             "team (flat baseline)",
    "team_dri":         "team + designated",
    "team_ownership":   "team + distributed",
    "team_indepdri":    "team + des. ethics lead",
    "partner":          "partner (hier. baseline)",
    "dri":              "designated",
    "ownership":        "distributed",
    "escalate":         "centralized",
    "indepdri":         "des. ethics lead",
    "safety":           "safety doc",
    "safety_indepdri":  "safety doc + des. ethics lead",
}


# ── Data pipeline ────────────────────────────────────────────────────────
def clean_model_name(name: str) -> str:
    """'claude-opus-4-1-20250805' -> 'opus-4-1'."""
    if not isinstance(name, str):
        return str(name)
    if name.startswith("claude-"):
        name = name[len("claude-"):]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        name = parts[0]
    return name


def parse_config(config_name: str) -> dict:
    """Parse a config_name like 'benign_team_dri_o41' into:
      prompt_type     : 'benign' | 'redteam'
      intervention_id : structural tag, e.g. 'team_dri', 'partner', 'safety_indepdri'
      family          : family taxonomy bucket (FAMILY_ORDER values)
      structure       : 'single' | 'team' | 'partner'
    Strips the optional model suffix (_o41/_o46/_s46), which is redundant
    with the separate `model` column.
    """
    parts = config_name.split("_")
    if parts and parts[-1] in ("o41", "o46", "s46"):
        parts = parts[:-1]
    prompt_type = parts[0]
    intervention_id = "_".join(parts[1:])
    fam, struct = INTERVENTION_FAMILY.get(intervention_id, ("other", "unknown"))
    return {
        "prompt_type":     prompt_type,
        "intervention_id": intervention_id,
        "family":          fam,
        "structure":       struct,
    }


def _unique_run_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Make run_id unique across the whole results tree.

    run_id is the run's timestamp folder name, which is only unique within one
    (scenario, model, config) batch: two batches launched in the same second
    (e.g. parallel run queues) produce identical names. Prefix it so runs from
    different batches are never merged.
    """
    df = df.copy()
    df["run_id"] = (df["scenario"].astype(str) + "/" + df["model"].astype(str) + "/"
                    + df["config_name"].astype(str) + "/" + df["run_id"].astype(str))
    return df


def load_and_annotate(csv_path, include=None, exclude=None,
                      eval_model: str = "claude-sonnet-4-6",
                      require_eval: bool = True, min_n: int = 3,
                      return_drops: bool = False):
    """Load all_results.csv.gz and reduce to one row per (run × judge model).

    Steps:
      1. Restrict to is_final=True rows (the final proposal version per run).
      2. Restrict to a single judge model (eval_model). Defaults to sonnet-4-6.
      3. Apply user include/exclude filters.
      4. Aggregate eval samples to mean score per (run × framework).
      5. Pivot framework so each run has business_goals_score and
         constitution_score columns.
      6. Annotate with prompt_type, intervention_id, family, structure.
      7. Drop runs missing either metric (require_eval=True), then min_n filter.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["is_final"].fillna(False).astype(bool)]
    df = _unique_run_ids(df)
    df["model"] = df["model"].apply(clean_model_name)
    df["eval_model"] = df["eval_model"].apply(clean_model_name)
    eval_model_short = clean_model_name(eval_model)
    df = df[df["eval_model"] == eval_model_short]

    include = include or {}
    exclude = exclude or {}
    for col, vals in include.items():
        if col == "model":
            vals = [clean_model_name(v) for v in vals]
        df = df[df[col].isin(vals)]
    for col, vals in exclude.items():
        df = df[~df[col].isin(vals)]

    # Mean-aggregate eval samples to one score per (run, framework)
    score_agg = (df.dropna(subset=["score"])
                 .groupby(["run_id", "framework"], as_index=False)
                 ["score"].mean())
    score_wide = score_agg.pivot(index="run_id", columns="framework", values="score")
    score_wide.columns = [f"{c}_score" for c in score_wide.columns]
    score_wide = score_wide.reset_index()

    # Run-level metadata (one row per run)
    run_cols = ["scenario", "model", "config_name", "run_id",
                "duration_seconds", "total_cost_usd", "total_messages_sent",
                "rate_limited_429", "api_overloaded_529", "api_error_other",
                "outcome", "proposal_outcome"]
    run_cols = [c for c in run_cols if c in df.columns]
    run_meta = df[run_cols].drop_duplicates(subset=["run_id"])

    out = run_meta.merge(score_wide, on="run_id", how="inner")

    ann = out["config_name"].apply(parse_config).apply(pd.Series)
    out = pd.concat([out, ann], axis=1)

    dropped_counts = (out[out["business_goals_score"].isna()
                          | out["constitution_score"].isna()]
                      .groupby(["model", "prompt_type", "intervention_id"]).size())

    if require_eval:
        out = out.dropna(subset=METRICS)

    if min_n and min_n > 0:
        counts = out.groupby(["model", "prompt_type", "intervention_id"]).size()
        keep = counts[counts >= min_n].index
        out = out.set_index(["model", "prompt_type", "intervention_id"]).loc[keep].reset_index()

    out = out.reset_index(drop=True)
    if return_drops:
        return out, dropped_counts
    return out


def build_summary_table(df: pd.DataFrame, dropped_counts: pd.Series | None = None) -> pd.DataFrame:
    """Per-(model, prompt_type, intervention_id) summary."""
    g = df.groupby(["model", "prompt_type", "intervention_id"])
    summary = g.agg(
        n=("run_id", "size"),
        business_goals=("business_goals_score", "mean"),
        constitution=("constitution_score", "mean"),
        total_cost_usd=("total_cost_usd", "mean"),
        duration_seconds=("duration_seconds", "mean"),
        messages=("total_messages_sent", "mean"),
    )
    summary["duration_min"] = summary["duration_seconds"] / 60
    summary = summary.drop(columns=["duration_seconds"])

    if dropped_counts is not None:
        summary["dropped"] = dropped_counts.reindex(summary.index, fill_value=0).astype(int)
    else:
        summary["dropped"] = 0

    col_order = ["n", "dropped", "business_goals", "constitution",
                 "total_cost_usd", "duration_min", "messages"]
    return summary[col_order].round(3)


def build_outcome_table(
    csv_or_df,
    eval_model: str = "claude-sonnet-4-6",
    include: dict | None = None,
    exclude: dict | None = None,
) -> pd.DataFrame:
    """Outcome breakdown: counts of each outcome per (model, prompt_type, intervention_id).

    Counts at the run level (not per-eval-sample). Restricts to the same
    eval_model the rest of the pipeline uses, then dedupes to one row per run.
    Applies include/exclude filters EXCEPT any `outcome` exclude (the whole
    point of this table is to see what an outcome filter would drop).
    Useful as a pre-flight diagnostic before `load_and_annotate`.
    """
    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(csv_or_df, low_memory=False)
    else:
        df = csv_or_df.copy()
    df = df[df["is_final"].fillna(False).astype(bool)]
    df = _unique_run_ids(df)
    df["model"] = df["model"].apply(clean_model_name)
    df["eval_model"] = df["eval_model"].apply(clean_model_name)
    df = df[df["eval_model"] == clean_model_name(eval_model)]

    for col, vals in (include or {}).items():
        if col == "model":
            vals = [clean_model_name(v) for v in vals]
        df = df[df[col].isin(vals)]
    for col, vals in (exclude or {}).items():
        if col == "outcome":
            continue
        df = df[~df[col].isin(vals)]

    # One row per run for outcome counting
    runs = df[["run_id", "model", "config_name", "outcome"]].drop_duplicates(subset=["run_id"])
    ann = runs["config_name"].apply(parse_config).apply(pd.Series)
    runs = pd.concat([runs, ann], axis=1)

    table = runs.groupby(["model", "prompt_type", "intervention_id", "outcome"]).size().unstack(fill_value=0)
    desired = ["success", "partial", "timeout", "failure"]
    cols = [c for c in desired if c in table.columns]
    cols += [c for c in table.columns if c not in desired]
    table = table[cols]
    table["total"] = table.sum(axis=1)
    return table


def compute_config_stats(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Collapse per-run df to one row per (model, prompt_type, intervention_id)
    with means, SEs, and Δ vs single / team / partner with CIs + sign flags.

    Δ comparisons stay WITHIN a (model, prompt_type) — so benign_dri compares
    against benign_single/benign_team, redteam_dri against redteam_*.
    """
    metrics = metrics or METRICS
    rows = []
    for (model, prompt, intervention), grp in df.groupby(["model", "prompt_type", "intervention_id"]):
        r = {"model": model, "prompt_type": prompt, "intervention_id": intervention,
             "n": len(grp)}
        for m in metrics:
            v = grp[m].dropna().values
            r[f"{m}_mean"] = float(np.mean(v)) if len(v) else np.nan
            r[f"{m}_se"]   = (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                              if len(v) > 1 else np.nan)
        rows.append(r)
    cells = pd.DataFrame(rows)

    meta = (df[["intervention_id", "family", "structure"]]
            .drop_duplicates().set_index("intervention_id"))
    cells = cells.join(meta, on="intervention_id")

    cells = _add_delta_vs_ref(cells, "single", "vs_single", metrics)
    cells = _add_delta_vs_ref(cells, "team",   "vs_team",   metrics)
    cells = _add_delta_vs_ref(cells, "partner", "vs_partner", metrics)

    # 3-class signed comparisons vs single — used by plot_mean / plot_safety_bars_vs_single.
    for met, key in [("business_goals_score", "bg"), ("constitution_score", "const")]:
        cells[f"{key}_sign"] = cells.apply(
            lambda r: _sig_sign(r[f"{met}_ci_lo_vs_single"],
                                r[f"{met}_ci_hi_vs_single"],
                                direction="higher_better"), axis=1)

    return cells


def _add_delta_vs_ref(cells, ref_intervention, suffix, metrics):
    """Compute Δ vs reference intervention WITHIN each (model, prompt_type)."""
    ref = (cells[cells["intervention_id"] == ref_intervention]
           .set_index(["model", "prompt_type"]))
    out_rows = []
    for _, r in cells.iterrows():
        key = (r["model"], r["prompt_type"])
        if key not in ref.index:
            for met in metrics:
                r[f"{met}_delta_{suffix}"] = np.nan
                r[f"{met}_se_{suffix}"]    = np.nan
                r[f"{met}_ci_lo_{suffix}"] = np.nan
                r[f"{met}_ci_hi_{suffix}"] = np.nan
            out_rows.append(r)
            continue
        rf = ref.loc[key]

        if r["intervention_id"] == ref_intervention:
            for met in metrics:
                r[f"{met}_delta_{suffix}"] = 0.0
                r[f"{met}_se_{suffix}"]    = 0.0
                r[f"{met}_ci_lo_{suffix}"] = 0.0
                r[f"{met}_ci_hi_{suffix}"] = 0.0
            out_rows.append(r)
            continue

        n_a = r["n"]
        n_b = rf["n"]
        for met in metrics:
            mu_a = r[f"{met}_mean"]
            mu_b = rf[f"{met}_mean"]
            se_a = r[f"{met}_se"]
            se_b = rf[f"{met}_se"]
            d = mu_a - mu_b
            if pd.isna(d) or pd.isna(se_a) or pd.isna(se_b) or (se_a == 0 and se_b == 0):
                se = np.nan if pd.isna(se_a) or pd.isna(se_b) else 0.0
                ci_lo = ci_hi = d if se == 0 else np.nan
            else:
                se = np.sqrt(se_a**2 + se_b**2)
                num = (se_a**2 + se_b**2) ** 2
                den = (se_a**4 / max(n_a - 1, 1)) + (se_b**4 / max(n_b - 1, 1))
                df_wa = num / den if den > 0 else max(n_a + n_b - 2, 1)
                tcrit = _stats.t.ppf(0.975, df_wa)
                ci_lo = d - tcrit * se
                ci_hi = d + tcrit * se
            r[f"{met}_delta_{suffix}"] = d
            r[f"{met}_se_{suffix}"]    = se
            r[f"{met}_ci_lo_{suffix}"] = ci_lo
            r[f"{met}_ci_hi_{suffix}"] = ci_hi
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def _sig_sign(ci_lo, ci_hi, direction: str = "higher_better") -> str:
    """3-class verdict by sign of two-sided CI."""
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return "inconclusive"
    if ci_hi < 0:
        return "worse" if direction == "higher_better" else "better"
    if ci_lo > 0:
        return "better" if direction == "higher_better" else "worse"
    return "same"


def _intervention_sort_key(intervention_id):
    """Sort key for ordering rows within a family."""
    return (INTERVENTION_RANK.get(intervention_id, 99), intervention_id)


def _ordered_rows(cells_subset):
    """Return cells_subset sorted by family, then by intervention intensity."""
    cells = cells_subset.copy()
    cells["family"] = cells["intervention_id"].map(
        lambda iid: INTERVENTION_FAMILY.get(iid, ("other", "unknown"))[0])
    out = []
    for fam in FAMILY_ORDER:
        sub = cells[cells["family"] == fam]
        if sub.empty:
            continue
        sub = sub.sort_values("intervention_id", key=lambda s: s.map(_intervention_sort_key))
        out.append(sub.reset_index(drop=True))
    if not out:
        return cells.iloc[0:0]
    return pd.concat(out, ignore_index=True)


# ── Bootstrap helpers (BCa) ──────────────────────────────────────────────
def _bca_pct(a: np.ndarray, q: float, B: int, rng):
    """BCa CI of the q-percentile of a single sample."""
    if len(a) < 2:
        return np.nan, np.nan, np.nan
    point = float(np.percentile(a, q * 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = _stats.bootstrap(
                (a,), lambda aa: np.percentile(aa, q * 100),
                n_resamples=B, method="BCa", vectorized=False,
                confidence_level=0.95, random_state=rng,
            )
            return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            return point, np.nan, np.nan


def _bca_delta(a: np.ndarray, b: np.ndarray, q: float, B: int, rng):
    """BCa CI for pct(a) − pct(b) between two independent samples."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    point = float(np.percentile(a, q * 100) - np.percentile(b, q * 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = _stats.bootstrap(
                (a, b),
                lambda aa, bb: np.percentile(aa, q * 100) - np.percentile(bb, q * 100),
                n_resamples=B, method="BCa", vectorized=False, paired=False,
                confidence_level=0.95, random_state=rng,
            )
            return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            return point, np.nan, np.nan


# ── Plots ────────────────────────────────────────────────────────────────


def plot_mean(cells: pd.DataFrame, model: str, prompt_type: str, out_path,
              scenario: str | None = None) -> plt.Figure | None:
    """Per-(model, prompt_type) faceted bars: one column per topology family,
    two rows (constitution top, business_goals bottom). Each non-single facet
    includes its topology baseline (gold-bordered) plus its interventions.
    Single facet has just the single bar (gold-bordered).

    Single-agent baseline shown as a horizontal dashed line + gray band of
    ±t·SE on each panel. Sig coloring vs single (3-state):
      - solid model color   = sig. less than single
      - light fill          = not different from single
      - light fill + hatch  = sig. greater than single
    """
    sub = cells[(cells["model"] == model) & (cells["prompt_type"] == prompt_type)]
    if sub.empty:
        return None
    rows_all = _ordered_rows(sub)
    if rows_all.empty:
        return None

    facets = []
    for fam in FAMILY_ORDER:
        if fam == "single":
            continue  # single shown only as horizontal reference + gray band
        fam_rows = rows_all[rows_all["family"] == fam].reset_index(drop=True)
        if fam_rows.empty:
            continue
        facets.append((fam, fam_rows))
    if not facets:
        return None

    facet_sizes = [len(r) for _, r in facets]
    n_cols = len(facets)
    fig_w = max(11, 1.1 * sum(facet_sizes) + 3)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(fig_w, 9),
        gridspec_kw={"width_ratios": facet_sizes},
        sharey="row", squeeze=False,
    )

    c = MODEL_COLOR.get(model, "#555")
    light_fill = "#dddddd"
    gold = "#DAA520"

    sb = sub[sub["intervention_id"] == "single"]
    sb_row = sb.iloc[0] if not sb.empty else None

    def _abs_ci(row, metric):
        if row is None:
            return None, None, None
        mu = row[f"{metric}_mean"]
        se = row[f"{metric}_se"]
        if pd.isna(mu):
            return None, None, None
        if pd.isna(se):
            return float(mu), None, None
        t = _stats.t.ppf(0.975, max(int(row["n"]) - 1, 1))
        return float(mu), float(mu - t * se), float(mu + t * se)

    def _topology_baseline_id(family):
        if family == "flat":
            return "team"
        if family == "hierarchical":
            return "partner"
        return "single"

    def draw_panel(ax, fam_rows, family, metric, sign_col, ylim, is_first_panel: bool):
        sb_m, sb_lo, sb_hi = _abs_ci(sb_row, metric)
        band_pts = [v for v in (sb_m, sb_lo, sb_hi) if v is not None]
        if band_pts:
            ax.axhspan(min(band_pts), max(band_pts),
                       color="#cccccc", alpha=0.35, zorder=0)
        if sb_m is not None:
            ax.axhline(sb_m, color="#444", linestyle="--", linewidth=1.2,
                       alpha=0.9, zorder=4,
                       label="single agent baseline" if is_first_panel else None)

        topology_baseline = _topology_baseline_id(family)
        bar_rows = [(r["intervention_id"], r) for _, r in fam_rows.iterrows()]
        for i, (iid, r) in enumerate(bar_rows):
            mu = r[f"{metric}_mean"]
            se = r[f"{metric}_se"]
            sign = r.get(sign_col, "inconclusive")
            if pd.isna(mu):
                continue
            is_baseline = iid == topology_baseline
            if sign == "worse":
                face, hatch = c, None
            elif sign == "better":
                face, hatch = light_fill, "//"
            else:
                face, hatch = light_fill, None
            edge = gold if is_baseline else c
            edge_w = 2.8 if is_baseline else 1.2
            err = 0.0
            if not pd.isna(se):
                t_row = _stats.t.ppf(0.975, max(int(r["n"]) - 1, 1))
                err = t_row * se
            ax.bar(i, mu, width=0.7, color=face, edgecolor=edge,
                   linewidth=edge_w, hatch=hatch, zorder=2)
            if err > 0:
                ax.errorbar(i, mu, yerr=err, fmt="none",
                            ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)
            label_text = f"{mu:.1f}"
            text_color = "white" if sign == "worse" else "#222"
            text_kwargs = dict(ha="center", va="bottom", fontsize=9,
                               fontweight="bold", zorder=6, color=text_color)
            if hatch is not None:
                text_kwargs["bbox"] = dict(facecolor="white", edgecolor="none",
                                           pad=1.0, alpha=0.85)
            ax.text(i, ylim[0] + 0.05, label_text, **text_kwargs)

        ax.set_xticks(range(len(bar_rows)))
        ax.set_xlim(-0.5, len(bar_rows) - 0.5)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        return bar_rows

    for col, (fam, fam_rows) in enumerate(facets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        is_first = col == 0
        draw_panel(ax_top, fam_rows, fam, "constitution_score", "const_sign",
                   SCORE_RANGE, is_first_panel=is_first)
        bar_rows = draw_panel(ax_bot, fam_rows, fam, "business_goals_score", "bg_sign",
                              SCORE_RANGE, is_first_panel=False)

        ax_top.set_title(FAMILY_DISPLAY[fam], fontsize=11, fontweight="bold")
        xtick_labels = [f"{BAR_LABEL.get(iid, CLEAN_LABEL.get(iid, iid))}\n(n={int(r['n'])})"
                        for iid, r in bar_rows]
        ax_top.set_xticks(range(len(bar_rows)))
        ax_top.set_xticklabels(xtick_labels, fontsize=10, rotation=0, ha="center")
        ax_bot.set_xticks(range(len(bar_rows)))
        ax_bot.set_xticklabels(xtick_labels, fontsize=10, rotation=0, ha="center")
        ax_top.tick_params(axis="y", labelsize=11)
        ax_bot.tick_params(axis="y", labelsize=11)
        if is_first:
            ax_top.set_ylabel("Ethics goal (constitution score)",
                              fontsize=11, fontweight="bold")
            ax_bot.set_ylabel("Business goal (proposal quality)",
                              fontsize=11, fontweight="bold")
        ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))

    title_suffix = f" · {scenario}" if scenario else ""
    fig.suptitle(f"Consultancy ({prompt_type}{title_suffix}) interventions on {model} — mean score",
                 fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.17, right=0.92, hspace=0.20, wspace=0.10)

    # Right-side annotation for the gray band on each row
    for row_idx, metric in [(0, "constitution_score"), (1, "business_goals_score")]:
        ax_right = axes[row_idx, -1]
        sb_m, sb_lo, sb_hi = _abs_ci(sb_row, metric)
        band_pts = [v for v in (sb_m, sb_lo, sb_hi) if v is not None]
        if not band_pts:
            continue
        band_mid = (min(band_pts) + max(band_pts)) / 2
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, band_mid), xycoords=("axes fraction", "data"),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#555",
            ha="left", va="center",
        )

    # Two-row legend, col-major fill: row1=[single, gold-bordered topology],
    # row2=[E, F, G] (sig states).
    ax_handles, ax_lbls = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(ax_lbls, ax_handles))
    A = label_to_handle.get("single agent baseline")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8,
                       label="topology baseline (highlighted)")
    E = mpatches.Patch(facecolor=c, edgecolor=c, linewidth=1.2,
                       label="sig. less than single agent")
    F = mpatches.Patch(facecolor=light_fill, edgecolor=c, linewidth=1.2,
                       label="not different from single agent")
    G = mpatches.Patch(facecolor=light_fill, edgecolor=c, linewidth=1.2,
                       hatch="//", label="sig. greater than single agent")
    interleaved = [A, E, D, F, mpatches.Patch(facecolor="none", edgecolor="none", label=" "), G]
    handles = [h for h in interleaved if h is not None]
    lbls = [h.get_label() for h in handles]
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig


def plot_worst(
    df: pd.DataFrame, model: str, prompt_type: str, out_path,
    pct: float = 0.10, n_bootstrap: int = 10000, seed: int = 0,
    scenario: str | None = None,
) -> plt.Figure | None:
    """Worst-case companion to ``plot_mean``: same family-facet layout
    on the same 1–7 scale, but bar values are the pct·100-percentile of
    constitution_score / business_goals_score with BCa bootstrap 95% CIs.

    Sig is tested against ``single`` via the BCa CI of the Δ-of-percentiles
    (CI excludes 0 → sig.). Gold-bordered topology baseline within each
    family facet (`team` for flat, `partner` for hierarchical, `single` for
    single).
    """
    sub_df = df[(df["model"] == model) & (df["prompt_type"] == prompt_type)]
    if sub_df.empty:
        return None
    rng = np.random.default_rng(seed)
    metrics = ["constitution_score", "business_goals_score"]

    runs = {}
    for iid, grp in sub_df.groupby("intervention_id"):
        runs[iid] = {m: grp[m].dropna().values for m in metrics}

    sb_runs = runs.get("single")
    if sb_runs is None:
        return None

    records = []
    for iid, r in runs.items():
        rec = {"intervention_id": iid,
               "n": int(min(len(r[metrics[0]]), len(r[metrics[1]])))}
        for met in metrics:
            pt, lo, hi = _bca_pct(r[met], pct, n_bootstrap, rng)
            rec[f"{met}_abs"] = pt
            rec[f"{met}_abs_lo"] = lo
            rec[f"{met}_abs_hi"] = hi
            if iid == "single":
                rec[f"{met}_delta_lo"] = 0.0
                rec[f"{met}_delta_hi"] = 0.0
            else:
                _, dlo, dhi = _bca_delta(r[met], sb_runs[met], pct, n_bootstrap, rng)
                rec[f"{met}_delta_lo"] = dlo
                rec[f"{met}_delta_hi"] = dhi
        records.append(rec)
    pct_df = pd.DataFrame(records)

    pct_df["family"] = pct_df["intervention_id"].map(
        lambda iid: INTERVENTION_FAMILY.get(iid, ("other", "unknown"))[0])

    facets = []
    for fam in FAMILY_ORDER:
        if fam == "single":
            continue  # single shown only as horizontal reference + gray band
        fam_rows = pct_df[pct_df["family"] == fam]
        if fam_rows.empty:
            continue
        fam_rows = fam_rows.sort_values("intervention_id",
                                        key=lambda s: s.map(_intervention_sort_key)
                                        ).reset_index(drop=True)
        facets.append((fam, fam_rows))
    if not facets:
        return None

    facet_sizes = [len(r) for _, r in facets]
    n_cols = len(facets)
    fig_w = max(11, 1.1 * sum(facet_sizes) + 3)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(fig_w, 9),
        gridspec_kw={"width_ratios": facet_sizes},
        sharey="row", squeeze=False,
    )

    c = MODEL_COLOR.get(model, "#555")
    light_fill = "#dddddd"
    gold = "#DAA520"

    sb_row = (pct_df[pct_df["intervention_id"] == "single"].iloc[0]
              if (pct_df["intervention_id"] == "single").any() else None)

    def _topology_baseline_id(family):
        if family == "flat":
            return "team"
        if family == "hierarchical":
            return "partner"
        return "single"

    def draw_panel(ax, fam_rows, family, metric, is_first_panel: bool):
        sb_abs = sb_lo = sb_hi = None
        if sb_row is not None:
            sb_abs = sb_row.get(f"{metric}_abs")
            sb_lo = sb_row.get(f"{metric}_abs_lo")
            sb_hi = sb_row.get(f"{metric}_abs_hi")
        band_pts = [v for v in (sb_abs, sb_lo, sb_hi)
                    if v is not None and not pd.isna(v)]
        if band_pts:
            ax.axhspan(min(band_pts), max(band_pts),
                       color="#cccccc", alpha=0.35, zorder=0)
        if sb_abs is not None and not pd.isna(sb_abs):
            ax.axhline(sb_abs, color="#444", linestyle="--", linewidth=1.2,
                       alpha=0.9, zorder=4,
                       label="single agent baseline" if is_first_panel else None)

        topology_baseline = _topology_baseline_id(family)
        bar_rows = [(r["intervention_id"], r) for _, r in fam_rows.iterrows()]
        for i, (iid, r) in enumerate(bar_rows):
            mu = r[f"{metric}_abs"]
            lo = r[f"{metric}_abs_lo"]
            hi = r[f"{metric}_abs_hi"]
            d_lo = r[f"{metric}_delta_lo"]
            d_hi = r[f"{metric}_delta_hi"]
            if pd.isna(mu):
                continue
            if pd.isna(d_lo) or pd.isna(d_hi):
                sign = "inconclusive"
            elif d_lo > 0:
                sign = "better"
            elif d_hi < 0:
                sign = "worse"
            else:
                sign = "same"
            if sign == "worse":
                face, hatch = c, None
            elif sign == "better":
                face, hatch = light_fill, "//"
            else:
                face, hatch = light_fill, None
            is_baseline = iid == topology_baseline
            edge = gold if is_baseline else c
            edge_w = 2.8 if is_baseline else 1.2
            err_lo = max(0.0, mu - lo) if not pd.isna(lo) else 0
            err_hi = max(0.0, hi - mu) if not pd.isna(hi) else 0
            ax.bar(i, mu, width=0.7, color=face, edgecolor=edge,
                   linewidth=edge_w, hatch=hatch, zorder=2)
            if (err_lo + err_hi) > 0:
                ax.errorbar(i, mu, yerr=[[err_lo], [err_hi]], fmt="none",
                            ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)
            text_color = "white" if sign == "worse" else "#222"
            text_kwargs = dict(ha="center", va="bottom", fontsize=9,
                               fontweight="bold", zorder=6, color=text_color)
            if hatch is not None:
                text_kwargs["bbox"] = dict(facecolor="white", edgecolor="none",
                                           pad=1.0, alpha=0.85)
            ax.text(i, SCORE_RANGE[0] + 0.05, f"{mu:.1f}", **text_kwargs)

        ax.set_xticks(range(len(bar_rows)))
        ax.set_xlim(-0.5, len(bar_rows) - 0.5)
        ax.set_ylim(*SCORE_RANGE)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        return bar_rows

    for col, (fam, fam_rows) in enumerate(facets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        is_first = col == 0
        draw_panel(ax_top, fam_rows, fam, "constitution_score", is_first_panel=is_first)
        bar_rows = draw_panel(ax_bot, fam_rows, fam, "business_goals_score", is_first_panel=False)

        ax_top.set_title(FAMILY_DISPLAY[fam], fontsize=11, fontweight="bold")
        xtick_labels = [f"{BAR_LABEL.get(iid, CLEAN_LABEL.get(iid, iid))}\n(n={int(r['n'])})"
                        for iid, r in bar_rows]
        ax_top.set_xticks(range(len(bar_rows)))
        ax_top.set_xticklabels(xtick_labels, fontsize=10, rotation=0, ha="center")
        ax_bot.set_xticks(range(len(bar_rows)))
        ax_bot.set_xticklabels(xtick_labels, fontsize=10, rotation=0, ha="center")
        ax_top.tick_params(axis="y", labelsize=11)
        ax_bot.tick_params(axis="y", labelsize=11)
        if is_first:
            q_pct = int(round(pct * 100))
            ax_top.set_ylabel(f"Ethics goal (worst-{q_pct}% constitution score)",
                              fontsize=11, fontweight="bold")
            ax_bot.set_ylabel(f"Business goal (worst-{q_pct}% proposal quality)",
                              fontsize=11, fontweight="bold")
        ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))

    q_pct = int(round(pct * 100))
    title_suffix = f" · {scenario}" if scenario else ""
    fig.suptitle(
        f"Consultancy ({prompt_type}{title_suffix}) interventions on {model} — {q_pct}th-percentile score",
        fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.17, right=0.92, hspace=0.32, wspace=0.10)

    # Right-side annotation
    for row_idx, metric in [(0, "constitution_score"), (1, "business_goals_score")]:
        ax_right = axes[row_idx, -1]
        sb_abs = sb_row.get(f"{metric}_abs") if sb_row is not None else None
        sb_lo  = sb_row.get(f"{metric}_abs_lo") if sb_row is not None else None
        sb_hi  = sb_row.get(f"{metric}_abs_hi") if sb_row is not None else None
        band_pts = [v for v in (sb_abs, sb_lo, sb_hi)
                    if v is not None and not pd.isna(v)]
        if not band_pts:
            continue
        band_mid = (min(band_pts) + max(band_pts)) / 2
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, band_mid), xycoords=("axes fraction", "data"),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#555",
            ha="left", va="center",
        )

    ax_handles, ax_lbls = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(ax_lbls, ax_handles))
    A = label_to_handle.get("single agent baseline")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8,
                       label="topology baseline (highlighted)")
    E = mpatches.Patch(facecolor=c, edgecolor=c, linewidth=1.2,
                       label="sig. less than single agent")
    F = mpatches.Patch(facecolor=light_fill, edgecolor=c, linewidth=1.2,
                       label="not different from single agent")
    G = mpatches.Patch(facecolor=light_fill, edgecolor=c, linewidth=1.2,
                       hatch="//", label="sig. greater than single agent")
    interleaved = [A, E, D, F, mpatches.Patch(facecolor="none", edgecolor="none", label=" "), G]
    handles = [h for h in interleaved if h is not None]
    lbls = [h.get_label() for h in handles]
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig


def plot_safety_bars_vs_single(
    cells: pd.DataFrame, df: pd.DataFrame, prompt_type: str, out_path,
    orgs: list[str] | None = None,
    models: list[str] | None = None,
    title: str | None = None,
    pct: float = 0.10, n_bootstrap: int = 10000, seed: int = 0,
) -> plt.Figure | None:
    """Headline figure for one prompt_type: rows = models, cols = [mean Δ,
    10th-pct Δ]. Bars are constitution_score Δ vs single, for the curated
    set: team, team_indepdri | partner, dri, safety, safety_indepdri.

    A small horizontal gap separates the flat-family bars (team*) from the
    hierarchical-family bars (partner*). team and partner are gold-bordered
    as the topology baselines for their respective families. Single-agent
    target range is a gray band of ±t·SE around 0 on the mean panel and the
    BCa CI of single's 10th-pct on the worst panel.

    Sig coloring (3-state, both columns, vs single):
      - solid model color   = sig. less than single
      - light fill          = not different from single
      - light fill + hatch  = sig. greater than single
    """
    if orgs is None:
        orgs = ["team", "team_indepdri", "partner", "indepdri", "safety", "safety_indepdri"]
    pt_cells = cells[cells["prompt_type"] == prompt_type]
    pt_df = df[df["prompt_type"] == prompt_type]
    if pt_cells.empty or pt_df.empty:
        return None

    present = set(pt_cells["model"].unique())
    if models is None:
        ordered = [m for m in MODEL_ORDER if m in present]
        ordered += [m for m in sorted(present) if m not in MODEL_ORDER]
        models = ordered

    # Bar x positions with a small gap between flat and hierarchical groups.
    # Anchor the visual divider relative to which org index is the first hierarchical one.
    flat_orgs = [o for o in orgs if INTERVENTION_FAMILY.get(o, ("other",))[0] == "flat"]
    hier_orgs = [o for o in orgs if INTERVENTION_FAMILY.get(o, ("other",))[0] == "hierarchical"]
    n_flat = len(flat_orgs)
    n_hier = len(hier_orgs)
    gap = 0.7
    xs = list(range(n_flat)) + [n_flat - 1 + gap + 1 + i for i in range(n_hier)]
    org_to_x = {o: xs[i] for i, o in enumerate(flat_orgs + hier_orgs)}
    # Reorder orgs to match the layout
    layout_orgs = flat_orgs + hier_orgs

    headline_label = {
        "team":              "baseline",
        "team_indepdri":     "ind.\nethics lead",
        "partner":           "baseline",
        "indepdri":          "ind.\nethics lead",
        "safety":            "safety\ndoc",
        "safety_indepdri":   "safety doc\n+ ind.\nethics lead",
    }

    rng = np.random.default_rng(seed)

    # Pre-compute BCa per-(model, org) for the 10th-pct column, and also
    # single's own band info per model.
    worst = {}  # (model, "__band__") = (sb_band_lo, sb_band_hi); (model, org) = (d, lo, hi)
    for model in models:
        sub_df = pt_df[pt_df["model"] == model]
        sb_runs = sub_df[sub_df["intervention_id"] == "single"]["constitution_score"].dropna().values
        if len(sb_runs) < 2:
            continue
        sb_pt, sb_lo_a, sb_hi_a = _bca_pct(sb_runs, pct, n_bootstrap, rng)
        sb_band_lo = (sb_lo_a - sb_pt) if not (pd.isna(sb_lo_a) or pd.isna(sb_pt)) else 0.0
        sb_band_hi = (sb_hi_a - sb_pt) if not (pd.isna(sb_hi_a) or pd.isna(sb_pt)) else 0.0
        worst[(model, "__band__")] = (sb_band_lo, sb_band_hi)
        for org in layout_orgs:
            org_runs = sub_df[sub_df["intervention_id"] == org]["constitution_score"].dropna().values
            d, lo, hi = _bca_delta(org_runs, sb_runs, pct, n_bootstrap, rng)
            worst[(model, org)] = (d, lo, hi)

    n_models = len(models)
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(max(8.5, 1.0 * len(layout_orgs) + 4.0), 2.6 * n_models + 1.6),
        sharex="col", sharey=False, squeeze=False,
    )

    light_fill = "#dddddd"
    gold = "#DAA520"

    # Determine y-limits per column (shared across rows for that col)
    all_lo_r1, all_hi_r1 = [], []
    all_lo_r2, all_hi_r2 = [], []
    for model in models:
        sub = pt_cells[pt_cells["model"] == model].set_index("intervention_id")
        for org in layout_orgs:
            if org in sub.index:
                r = sub.loc[org]
                lo = r.get("constitution_score_ci_lo_vs_single", r["constitution_score_delta_vs_single"])
                hi = r.get("constitution_score_ci_hi_vs_single", r["constitution_score_delta_vs_single"])
                if not pd.isna(lo): all_lo_r1.append(lo)
                if not pd.isna(hi): all_hi_r1.append(hi)
        # single's own SE band on row 1
        sb = sub.loc["single"] if "single" in sub.index else None
        if sb is not None and not pd.isna(sb.get("constitution_score_se")):
            t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
            all_lo_r1.append(-t_sb * sb["constitution_score_se"])
            all_hi_r1.append(+t_sb * sb["constitution_score_se"])

        for org in layout_orgs:
            d, lo, hi = worst.get((model, org), (np.nan, np.nan, np.nan))
            if not pd.isna(lo): all_lo_r2.append(lo)
            if not pd.isna(hi): all_hi_r2.append(hi)
        band = worst.get((model, "__band__"))
        if band is not None:
            all_lo_r2.append(band[0]); all_hi_r2.append(band[1])

    pad = 0.4
    y_min_r1 = min(all_lo_r1 + [0]) - pad
    y_max_r1 = max(all_hi_r1 + [0]) + pad
    y_min_r2 = min(all_lo_r2 + [0]) - pad
    y_max_r2 = max(all_hi_r2 + [0]) + pad

    for row, model in enumerate(models):
        c = MODEL_COLOR.get(model, "#555")
        sub = pt_cells[pt_cells["model"] == model].set_index("intervention_id")

        # ── Left column (mean Δ) ─────────────────────────────────────────
        ax = axes[row, 0]
        ax.axhline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4,
                   label="single agent baseline (Δ=0)" if row == 0 else None)
        sb = sub.loc["single"] if "single" in sub.index else None
        if sb is not None and not pd.isna(sb.get("constitution_score_se")):
            t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
            ax.axhspan(-t_sb * sb["constitution_score_se"], +t_sb * sb["constitution_score_se"],
                       color="#cccccc", alpha=0.35, zorder=0,
                       label="single-agent target range (95% CI)" if row == 0 else None)

        for org in layout_orgs:
            if org not in sub.index:
                continue
            r = sub.loc[org]
            mu = r["constitution_score_delta_vs_single"]
            ci_lo = r.get("constitution_score_ci_lo_vs_single", mu)
            ci_hi = r.get("constitution_score_ci_hi_vs_single", mu)
            sign = r.get("const_sign", "inconclusive")
            if sign == "worse":
                face, hatch = c, None
            elif sign == "better":
                face, hatch = light_fill, "//"
            else:
                face, hatch = light_fill, None
            is_topology_baseline = org in ("team", "partner")
            edge = gold if is_topology_baseline else c
            edge_w = 2.8 if is_topology_baseline else 1.2
            err_lo = mu - ci_lo if not pd.isna(ci_lo) else 0
            err_hi = ci_hi - mu if not pd.isna(ci_hi) else 0
            x = org_to_x[org]
            ax.bar(x, mu, width=0.7, color=face, edgecolor=edge,
                   linewidth=edge_w, hatch=hatch, zorder=2)
            ax.errorbar(x, mu, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)

        ax.set_ylim(y_min_r1, y_max_r1)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        # Family separator (vertical thin line in the gap)
        if n_flat > 0 and n_hier > 0:
            sep_x = (xs[n_flat - 1] + xs[n_flat]) / 2
            ax.axvline(sep_x, color="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=0.5)
        ax.set_ylabel(f"{model}\nΔ vs single (score)", fontsize=10, fontweight="bold")

        # ── Right column (10th-pct Δ) ────────────────────────────────────
        ax = axes[row, 1]
        ax.axhline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
        band = worst.get((model, "__band__"))
        if band is not None:
            ax.axhspan(band[0], band[1], color="#cccccc", alpha=0.35, zorder=0)

        for org in layout_orgs:
            d, lo, hi = worst.get((model, org), (np.nan, np.nan, np.nan))
            if pd.isna(d):
                continue
            if pd.isna(lo) or pd.isna(hi):
                sign = "inconclusive"
            elif lo > 0:
                sign = "better"
            elif hi < 0:
                sign = "worse"
            else:
                sign = "same"
            if sign == "worse":
                face, hatch = c, None
            elif sign == "better":
                face, hatch = light_fill, "//"
            else:
                face, hatch = light_fill, None
            is_topology_baseline = org in ("team", "partner")
            edge = gold if is_topology_baseline else c
            edge_w = 2.8 if is_topology_baseline else 1.2
            err_lo = max(0.0, d - lo) if not pd.isna(lo) else 0
            err_hi = max(0.0, hi - d) if not pd.isna(hi) else 0
            x = org_to_x[org]
            ax.bar(x, d, width=0.7, color=face, edgecolor=edge,
                   linewidth=edge_w, hatch=hatch, zorder=2)
            if (err_lo + err_hi) > 0:
                ax.errorbar(x, d, yerr=[[err_lo], [err_hi]], fmt="none",
                            ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)

        ax.set_ylim(y_min_r2, y_max_r2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        if n_flat > 0 and n_hier > 0:
            sep_x = (xs[n_flat - 1] + xs[n_flat]) / 2
            ax.axvline(sep_x, color="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=0.5)

    # X-tick labels on bottom row of each column
    bot_left = axes[-1, 0]
    bot_right = axes[-1, 1]
    bot_left.set_xticks(xs)
    bot_left.set_xticklabels([headline_label.get(o, o) for o in layout_orgs],
                             fontsize=9.5, ha="center")
    bot_right.set_xticks(xs)
    bot_right.set_xticklabels([headline_label.get(o, o) for o in layout_orgs],
                              fontsize=9.5, ha="center")

    # Column headers
    axes[0, 0].set_title("Mean Δ constitution score", fontsize=12, fontweight="bold", pad=18)
    axes[0, 1].set_title("10th-percentile Δ constitution score",
                          fontsize=12, fontweight="bold", pad=18)

    # Family group labels above the top row of each column
    if n_flat > 0 and n_hier > 0:
        flat_x = sum(xs[:n_flat]) / n_flat
        hier_x = sum(xs[n_flat:]) / n_hier
        for col in (0, 1):
            ax_top = axes[0, col]
            trans = ax_top.get_xaxis_transform()
            ax_top.text(flat_x, 1.02, "Flat", transform=trans,
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#444")
            ax_top.text(hier_x, 1.02, "Hierarchical", transform=trans,
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#444")

    suptitle = title or f"Consultancy ({prompt_type}): constitution Δ vs single agent"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.0)

    plt.tight_layout()
    # Single-model figure is short; need extra bottom margin for x-tick labels
    # (3-line "safety doc + ethics lead" labels) plus the legend below.
    if n_models == 1:
        plt.subplots_adjust(top=0.78, bottom=0.32, hspace=0.18, wspace=0.18)
    else:
        plt.subplots_adjust(top=0.90, bottom=0.16, hspace=0.18, wspace=0.18)

    # Two-row legend, col-major fill: row1 = [single line, gold-bordered],
    # row2 = sig states
    pick_color = MODEL_COLOR.get(models[0], "#555")
    A = mpatches.Patch(facecolor="#cccccc", edgecolor="none", alpha=0.5,
                       label="single-agent target range (95% CI)")
    Aline = mlines.Line2D([0], [0], color="#444", linestyle="--", linewidth=1.5,
                          label="single-agent baseline (Δ=0)")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8,
                       label="multi-agent baseline")
    E = mpatches.Patch(facecolor=pick_color, edgecolor=pick_color, linewidth=1.2,
                       label="sig. less than single agent")
    F = mpatches.Patch(facecolor=light_fill, edgecolor=pick_color, linewidth=1.2,
                       label="not different from single agent")
    G = mpatches.Patch(facecolor=light_fill, edgecolor=pick_color, linewidth=1.2,
                       hatch="//", label="sig. greater than single agent")
    interleaved = [Aline, E, A, F, D, G]
    handles = interleaved
    lbls = [h.get_label() for h in handles]
    legend_y = -0.08 if n_models == 1 else 0.0
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, legend_y), frameon=False, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_safety_summary_condensed(
    cells: pd.DataFrame, df: pd.DataFrame, out_path,
    orgs: list[str] | None = None,
    models: list[str] | None = None,
    title: str | None = None,
) -> plt.Figure | None:
    """Condensed headline figure: rows = models, cols = [benign mean Δ,
    redteam mean Δ]. Drops the 10th-percentile column. Same curated bars and
    sig coloring as plot_safety_bars_vs_single's left column.

    Default curated set on the hierarchical side uses `indepdri` (independent
    ethics lead) instead of `dri` (designated responsibility), so both family
    panels share parallel intervention semantics: baseline + ethics lead.

    Defaults to opus-4-1 only — this is the in-paper headline figure and was
    deliberately scoped to one model. Pass `models=...` to override.
    """
    if orgs is None:
        orgs = ["team", "team_indepdri", "partner", "indepdri", "safety", "safety_indepdri"]
    if cells.empty or df.empty:
        return None

    present = set(cells["model"].unique())
    if models is None:
        models = [m for m in ["opus-4-1"] if m in present]
    else:
        models = [m for m in models if m in present]
    if not models:
        return None

    flat_orgs = [o for o in orgs if INTERVENTION_FAMILY.get(o, ("other",))[0] == "flat"]
    hier_orgs = [o for o in orgs if INTERVENTION_FAMILY.get(o, ("other",))[0] == "hierarchical"]
    n_flat = len(flat_orgs)
    n_hier = len(hier_orgs)
    gap = 0.7
    xs = list(range(n_flat)) + [n_flat - 1 + gap + 1 + i for i in range(n_hier)]
    org_to_x = {o: xs[i] for i, o in enumerate(flat_orgs + hier_orgs)}
    layout_orgs = flat_orgs + hier_orgs

    headline_label = {
        "team":              "baseline",
        "team_indepdri":     "ind.\nethics lead",
        "partner":           "baseline",
        "indepdri":          "ind.\nethics lead",
        "safety":            "safety\ndoc",
        "safety_indepdri":   "safety doc\n+ ind.\nethics lead",
    }

    prompts = ["benign", "redteam"]
    prompts = [p for p in prompts if p in df["prompt_type"].unique()]
    if not prompts:
        return None

    n_models = len(models)
    n_cols = len(prompts)
    fig_h = 4.6 if n_models == 1 else 2.6 * n_models + 1.4
    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(max(7.5, 1.0 * len(layout_orgs) + 3.5) * n_cols / 2, fig_h),
        sharex="col", sharey="row", squeeze=False,
    )

    light_fill = "#dddddd"
    gold = "#DAA520"

    # Per-row y-limits (shared across the two prompt columns for each model)
    row_ylims = {}
    for model in models:
        all_lo, all_hi = [0.0], [0.0]
        for prompt in prompts:
            sub = cells[(cells["prompt_type"] == prompt)
                        & (cells["model"] == model)].set_index("intervention_id")
            for org in layout_orgs:
                if org in sub.index:
                    r = sub.loc[org]
                    lo = r.get("constitution_score_ci_lo_vs_single",
                               r["constitution_score_delta_vs_single"])
                    hi = r.get("constitution_score_ci_hi_vs_single",
                               r["constitution_score_delta_vs_single"])
                    if not pd.isna(lo): all_lo.append(lo)
                    if not pd.isna(hi): all_hi.append(hi)
            if "single" in sub.index:
                sb = sub.loc["single"]
                if not pd.isna(sb.get("constitution_score_se")):
                    t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
                    all_lo.append(-t_sb * sb["constitution_score_se"])
                    all_hi.append(+t_sb * sb["constitution_score_se"])
        pad = 0.4
        row_ylims[model] = (min(all_lo) - pad, max(all_hi) + pad)

    for row, model in enumerate(models):
        c = MODEL_COLOR.get(model, "#555")
        for col, prompt in enumerate(prompts):
            ax = axes[row, col]
            sub = cells[(cells["prompt_type"] == prompt)
                        & (cells["model"] == model)].set_index("intervention_id")
            ax.axhline(0, color="#444", linestyle="--", linewidth=1.2,
                       alpha=0.9, zorder=4)
            if "single" in sub.index:
                sb = sub.loc["single"]
                if not pd.isna(sb.get("constitution_score_se")):
                    t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
                    ax.axhspan(-t_sb * sb["constitution_score_se"],
                               +t_sb * sb["constitution_score_se"],
                               color="#cccccc", alpha=0.35, zorder=0)

            for org in layout_orgs:
                if org not in sub.index:
                    continue
                r = sub.loc[org]
                mu = r["constitution_score_delta_vs_single"]
                ci_lo = r.get("constitution_score_ci_lo_vs_single", mu)
                ci_hi = r.get("constitution_score_ci_hi_vs_single", mu)
                sign = r.get("const_sign", "inconclusive")
                if sign == "worse":
                    face, hatch = c, None
                elif sign == "better":
                    face, hatch = light_fill, "//"
                else:
                    face, hatch = light_fill, None
                is_topology_baseline = org in ("team", "partner")
                edge = gold if is_topology_baseline else c
                edge_w = 2.8 if is_topology_baseline else 1.2
                err_lo = mu - ci_lo if not pd.isna(ci_lo) else 0
                err_hi = ci_hi - mu if not pd.isna(ci_hi) else 0
                x = org_to_x[org]
                ax.bar(x, mu, width=0.7, color=face, edgecolor=edge,
                       linewidth=edge_w, hatch=hatch, zorder=2)
                ax.errorbar(x, mu, yerr=[[err_lo], [err_hi]], fmt="none",
                            ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)

            ax.set_ylim(*row_ylims[model])
            ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            if n_flat > 0 and n_hier > 0:
                sep_x = (xs[n_flat - 1] + xs[n_flat]) / 2
                ax.axvline(sep_x, color="#bbbbbb", linewidth=0.8,
                           alpha=0.6, zorder=0.5)
            if col == 0:
                ax.set_ylabel(f"{model}\nΔ vs single (score)",
                              fontsize=10, fontweight="bold")

    # X-tick labels on every row's bottom-row axis (shared via sharex='col')
    for col in range(n_cols):
        axes[-1, col].set_xticks(xs)
        axes[-1, col].set_xticklabels(
            [headline_label.get(o, o) for o in layout_orgs],
            fontsize=9, ha="center",
        )

    # Column headers (prompt_type)
    col_titles = {"benign": "benign prompt", "redteam": "redteam prompt"}
    for col, prompt in enumerate(prompts):
        axes[0, col].set_title(col_titles.get(prompt, prompt),
                               fontsize=12, fontweight="bold", pad=22)

    # Family group labels above row 0
    if n_flat > 0 and n_hier > 0:
        flat_x = sum(xs[:n_flat]) / n_flat
        hier_x = sum(xs[n_flat:]) / n_hier
        for col in range(n_cols):
            ax_top = axes[0, col]
            trans = ax_top.get_xaxis_transform()
            ax_top.text(flat_x, 1.02, "Flat", transform=trans,
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#444")
            ax_top.text(hier_x, 1.02, "Hierarchical", transform=trans,
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#444")

    suptitle = title or "Δ Ethics score"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.0)

    plt.tight_layout()
    if n_models == 1:
        plt.subplots_adjust(top=0.82, bottom=0.32, right=0.92,
                            hspace=0.25, wspace=0.12)
    else:
        plt.subplots_adjust(top=0.90, bottom=0.18, right=0.93,
                            hspace=0.25, wspace=0.12)

    # Right-side annotation for the gray single-agent target range, on the
    # rightmost subplot of each row.
    for row_idx in range(n_models):
        ax_right = axes[row_idx, -1]
        sub = cells[(cells["prompt_type"] == prompts[-1])
                    & (cells["model"] == models[row_idx])].set_index("intervention_id")
        if "single" not in sub.index:
            continue
        sb = sub.loc["single"]
        if pd.isna(sb.get("constitution_score_se")):
            continue
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, 0.0),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#555",
            ha="left", va="center",
        )

    pick_color = MODEL_COLOR.get(models[0], "#555")
    Aline = mlines.Line2D([0], [0], color="#444", linestyle="--", linewidth=1.5,
                          label="single agent baseline")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8,
                       label="multi-agent baseline")
    E = mpatches.Patch(facecolor=pick_color, edgecolor=pick_color, linewidth=1.2,
                       label="sig. < than single ag. baseline")
    F = mpatches.Patch(facecolor=light_fill, edgecolor=pick_color, linewidth=1.2,
                       label="not sig. diff. from single ag. baseline")
    G = mpatches.Patch(facecolor=light_fill, edgecolor=pick_color, linewidth=1.2,
                       hatch="//", label="sig. > than single ag. baseline")
    # Two stacked legends: row 1 = baseline keys (2 entries), row 2 = sig states (3 entries).
    if n_models == 1:
        leg1_y, leg2_y = 0.09, 0.03
    else:
        leg1_y, leg2_y = 0.05, 0.005
    leg1 = fig.legend([Aline, D], [Aline.get_label(), D.get_label()],
                      loc="lower center", ncol=2,
                      bbox_to_anchor=(0.5, leg1_y),
                      frameon=False, fontsize=9)
    fig.add_artist(leg1)
    fig.legend([E, F, G], [E.get_label(), F.get_label(), G.get_label()],
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, leg2_y),
               frameon=False, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_benefit(df: pd.DataFrame, cells: pd.DataFrame, prompt_type: str,
                      out_path, models: list[str] | None = None) -> plt.Figure | None:
    """Faceted cost-benefit scatter for one prompt_type: one panel per model.

    Per panel:
      X = mean total_cost / that model's `team` (flat) mean total_cost.
      Y = Δ constitution_score vs that model's `team` baseline, in score points
          (Welch CI on Y bars).

    Markers (color = model, filled = sig. ≠ team, hollow = CI crosses 0):
      ● circle = multi-agent intervention; 2-char code rendered inside.
      ✚ filled plus = single agent; code rendered offset-right.
      ★ star = compound intervention (safety_indepdri).

    Per-model Pareto frontier (multi-agent only) as a model-color step line.
    Cross-hair at (1.0×, 0) marks the per-model flat baseline. Last subplot
    slot holds the legend.
    """
    pt_cells = cells[cells["prompt_type"] == prompt_type]
    pt_df = df[df["prompt_type"] == prompt_type]
    if pt_cells.empty or pt_df.empty:
        return None

    cost_mean = (pt_df.groupby(["model", "intervention_id"])["total_cost_usd"].mean()
                 .rename("total_cost_mean").reset_index())
    base_cost = (cost_mean[cost_mean["intervention_id"] == "team"]
                 .set_index("model")["total_cost_mean"])
    pts = pt_cells.merge(cost_mean, on=["model", "intervention_id"])
    pts = pts[pts["model"].isin(base_cost.index)].copy()
    pts["frac_cost"] = pts["total_cost_mean"] / pts["model"].map(base_cost)
    pts["delta_pp"] = pts["constitution_score_delta_vs_team"]
    pts["ci_lo_pp"] = pts["constitution_score_ci_lo_vs_team"]
    pts["ci_hi_pp"] = pts["constitution_score_ci_hi_vs_team"]
    pts["family"] = pts["intervention_id"].map(
        lambda iid: INTERVENTION_FAMILY.get(iid, ("other", "unknown"))[0])
    # Drop the team point itself (it's at 1×, 0) — keep single, partner, and interventions
    pts = pts[pts["intervention_id"] != "team"].dropna(subset=["frac_cost", "delta_pp"])
    if pts.empty:
        return None

    if models is not None:
        pts = pts[pts["model"].isin(models)]
        if pts.empty:
            return None
        present_models = [m for m in models if m in pts["model"].unique()]
    else:
        present_models = [m for m in MODEL_ORDER if m in pts["model"].unique()]
        present_models += [m for m in sorted(pts["model"].unique()) if m not in MODEL_ORDER]

    # Keep 3 cols so the trailing slot can hold the legend
    n_cols = 3
    n_rows = max(1, (len(present_models) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.4 * n_cols, 4.2 * n_rows),
        sharex=False, sharey=False, squeeze=False,
    )
    axes_flat = axes.flatten()

    def _pareto(sub):
        s = sub.sort_values("frac_cost")
        front, best = [], -np.inf
        for _, r in s.iterrows():
            if r["delta_pp"] > best:
                front.append((r["frac_cost"], r["delta_pp"], r["intervention_id"]))
                best = r["delta_pp"]
        return front

    def _marker_for(intervention_id):
        if intervention_id == "single":
            return "P"
        if intervention_id == "safety_indepdri":
            return "*"
        return "o"

    for i, m in enumerate(present_models):
        ax = axes_flat[i]
        sub = pts[pts["model"] == m]
        c = MODEL_COLOR.get(m, "#555")

        ax.axhline(0, color="#8b5a3c", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1.5)
        ax.axvline(1.0, color="#8b5a3c", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1.5)

        for _, r in sub.iterrows():
            x, y = r["frac_cost"], r["delta_pp"]
            iid = r["intervention_id"]
            marker = _marker_for(iid)
            code = INTERVENTION_CODE.get(iid, "")
            if not (pd.isna(r["ci_lo_pp"]) or pd.isna(r["ci_hi_pp"])):
                sig = not (r["ci_lo_pp"] <= 0 <= r["ci_hi_pp"])
                ax.vlines(x, r["ci_lo_pp"], r["ci_hi_pp"],
                          color=c, linewidth=1.0, alpha=0.35, zorder=2)
            else:
                sig = False
            face = c if sig else "white"

            if marker == "o":
                ax.scatter(x, y, s=380, marker="o", facecolor=face, edgecolor=c,
                           linewidth=1.5, zorder=4)
                if code:
                    text_color = "white" if sig else c
                    ax.text(x, y, code, ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color=text_color, zorder=5)
            elif marker == "P":
                ax.scatter(x, y, s=180, marker="P", facecolor=face, edgecolor=c,
                           linewidth=1.5, zorder=4)
                if code:
                    ax.annotate(code, xy=(x, y), xytext=(8, 0),
                                textcoords="offset points",
                                fontsize=8, fontweight="bold", color=c,
                                va="center", zorder=5)
            else:
                ax.scatter(x, y, s=440, marker="*", facecolor=face, edgecolor=c,
                           linewidth=2.0, zorder=4,
                           path_effects=[mpe.withStroke(linewidth=3.5, foreground="#222")])

        front = _pareto(sub[sub["intervention_id"] != "single"])
        if len(front) >= 2:
            xs_, ys_ = zip(*[(f[0], f[1]) for f in front])
            ax.plot(xs_, ys_, color=c, linewidth=1.6, alpha=0.65,
                    linestyle="-", zorder=2.5)

        ax.set_title(m, fontweight="bold", fontsize=11)
        ax.grid(True, which="major", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlabel("Cost / flat baseline")
        ax.set_ylabel("Δ ethics score vs flat baseline")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.1f}"))

    for j in range(len(present_models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if len(present_models) < len(axes_flat):
        leg_ax = axes_flat[len(present_models)]
        leg_ax.set_visible(True)
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)
        leg_ax.set_xticks([])
        leg_ax.set_yticks([])
        for spine in leg_ax.spines.values():
            spine.set_visible(False)

        marker_x = 0.07
        text_x = 0.22
        gray = "#555"

        def _row(y, marker, label, *, marker_size=180, facecolor=gray,
                 edgecolor=gray, linewidth=1.5, path_effects=None):
            leg_ax.scatter(marker_x, y, s=marker_size, marker=marker,
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=linewidth, zorder=3, path_effects=path_effects)
            leg_ax.text(text_x, y, label, fontsize=9.5, va="center", ha="left")

        leg_ax.text(0.02, 0.98, "Marker", fontweight="bold", fontsize=11,
                    va="top", ha="left")
        _row(0.88, "o", "multi-agent intervention", marker_size=180)
        _row(0.78, "P", "single agent", marker_size=150)
        _row(0.68, "*", "compound (safety doc + ethics lead)",
             marker_size=220, linewidth=1.8,
             path_effects=[mpe.withStroke(linewidth=3.0, foreground="#222")])
        leg_ax.text(text_x, 0.60,
                    "filled = sig. ≠ baseline ·  hollow = CI crosses 0",
                    fontsize=8.5, fontstyle="italic", color="#444",
                    va="center", ha="left")

        leg_ax.plot([marker_x - 0.05, marker_x + 0.05], [0.53, 0.53],
                    color=gray, linewidth=1.8, alpha=0.7, zorder=3)
        leg_ax.text(text_x, 0.53,
                    "per-model Pareto frontier (multi-agent only)",
                    fontsize=9.5, va="center", ha="left")
        leg_ax.plot([marker_x - 0.05, marker_x + 0.05], [0.46, 0.46],
                    color="#8b5a3c", linewidth=1.0, linestyle="--", zorder=3)
        leg_ax.text(text_x, 0.46, "flat baseline (1×, 0)",
                    fontsize=9.5, va="center", ha="left")

        leg_ax.text(0.02, 0.42, "Code → intervention", fontweight="bold",
                    fontsize=11, va="top", ha="left")
        code_lines = [
            ("S",  "single agent"),
            ("Pb", "partner baseline (hierarchical)"),
            ("Td/To/Ti", "team: designated / distributed / ethics lead"),
            ("Dr", "designated responsibility"),
            ("Ow", "distributed responsibility"),
            ("Es", "centralized escalation"),
            ("Id", "des. ethics lead"),
            ("Sd", "safety doc"),
            ("★",  "safety doc + des. ethics lead"),
        ]
        line_y0 = 0.32
        line_dy = 0.040
        for k, (code, name) in enumerate(code_lines):
            y = line_y0 - k * line_dy
            leg_ax.text(0.04, y, code, fontsize=9, family="monospace",
                        fontweight="bold", va="center", ha="left")
            leg_ax.text(0.34, y, name, fontsize=9, va="center", ha="left")

    fig.suptitle(
        f"Consultancy ({prompt_type}): cost-benefit of interventions vs flat baseline, by model",
        fontweight="bold", fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig


def plot_scatter(
    df: pd.DataFrame, scenario: str, model: str, prompt_type: str, intervention_id: str, out_path,
) -> plt.Figure | None:
    """Per-(scenario, model, prompt_type, intervention) scatter on the
    business_goals × constitution plane (1–7 each, upper-right=better),
    with single overlay and per-group Pareto frontier. Filtered to one scenario
    so the per-scenario story is legible (otherwise pooling 10 scenarios masks
    cell-specific patterns)."""
    sub_s = df[(df["scenario"] == scenario)
               & (df["model"] == model)
               & (df["prompt_type"] == prompt_type)
               & (df["intervention_id"] == "single")]
    sub_o = df[(df["scenario"] == scenario)
               & (df["model"] == model)
               & (df["prompt_type"] == prompt_type)
               & (df["intervention_id"] == intervention_id)]
    if sub_s.empty or sub_o.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 5))

    def pareto(xs, ys):
        # Upper-right Pareto: walk by x descending, keep monotone-up in y
        pts = sorted(zip(xs, ys), key=lambda p: -p[0])
        front, best_y = [], -float("inf")
        for x, y in pts:
            if y > best_y:
                front.append((x, y))
                best_y = y
        return sorted(front)

    sx = sub_s["business_goals_score"].values
    sy = sub_s["constitution_score"].values
    ox = sub_o["business_goals_score"].values
    oy = sub_o["constitution_score"].values

    ax.scatter(sx, sy, s=50, color="#888", alpha=0.55,
               edgecolor="white", linewidth=0.5, zorder=4,
               label=f"single (n={len(sub_s)})")
    ax.scatter(ox, oy, s=60, color="#d62728", alpha=0.75,
               edgecolor="white", linewidth=0.5, zorder=5,
               label=f"{intervention_id} (n={len(sub_o)})")

    mask = ~(np.isnan(sx) | np.isnan(sy))
    if mask.any():
        pts = pareto(sx[mask], sy[mask])
        if pts:
            px, py = zip(*pts)
            ax.plot(px, py, "--", color="#888", linewidth=1.4, alpha=0.7, zorder=3)
    mask = ~(np.isnan(ox) | np.isnan(oy))
    if mask.any():
        pts = pareto(ox[mask], oy[mask])
        if pts:
            px, py = zip(*pts)
            ax.plot(px, py, "--", color="#d62728", linewidth=1.4, alpha=0.8, zorder=3)

    ax.set_xlabel("Business goal (proposal quality)")
    ax.set_ylabel("Ethics goal (constitution score)")
    ax.set_xlim(*SCORE_RANGE)
    ax.set_ylim(*SCORE_RANGE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_title(f"{scenario} · {model} · {prompt_type} · {intervention_id}",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", frameon=True, fontsize=8)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return fig


def plot_means_per_cell(
    df: pd.DataFrame, scenario: str, model: str, prompt_type: str, out_path,
) -> plt.Figure | None:
    """Per-(scenario, model, prompt) scatter of intervention *means*.

    All interventions plotted as a single point each on the
    business_goals × constitution plane (1–7 each, upper-right=better) with
    error crosses showing 95% Welch CIs. Marker shape encodes family
    (single/flat/hierarchical), color encodes intervention identity,
    INTERVENTION_CODE labels render inside each marker.

    Legend is identical on every plot (uses INTERVENTION_STYLE +
    LEGEND_ORDER) so the reader learns the mapping once and can browse
    cells comparatively.
    """
    sub = df[(df["scenario"] == scenario)
             & (df["model"] == model)
             & (df["prompt_type"] == prompt_type)]
    if sub.empty:
        return None

    rows = []
    for iid, grp in sub.groupby("intervention_id"):
        bg = grp["business_goals_score"].dropna().values
        cs = grp["constitution_score"].dropna().values
        n = min(len(bg), len(cs))
        if n < 2:
            continue
        bg_mean = float(bg.mean())
        cs_mean = float(cs.mean())
        bg_se = float(bg.std(ddof=1) / np.sqrt(len(bg)))
        cs_se = float(cs.std(ddof=1) / np.sqrt(len(cs)))
        t = _stats.t.ppf(0.975, max(n - 1, 1))
        rows.append({
            "iid": iid, "n": n,
            "bg_mean": bg_mean, "bg_lo": bg_mean - t * bg_se, "bg_hi": bg_mean + t * bg_se,
            "cs_mean": cs_mean, "cs_lo": cs_mean - t * cs_se, "cs_hi": cs_mean + t * cs_se,
        })
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in rows:
        style = INTERVENTION_STYLE.get(r["iid"], {"color": "#888", "marker": "o", "size": 200})
        c = style["color"]
        m = style["marker"]
        s = style["size"]
        ax.hlines(r["cs_mean"], r["bg_lo"], r["bg_hi"],
                  color=c, linewidth=1.0, alpha=0.45, zorder=2)
        ax.vlines(r["bg_mean"], r["cs_lo"], r["cs_hi"],
                  color=c, linewidth=1.0, alpha=0.45, zorder=2)
        ax.scatter(r["bg_mean"], r["cs_mean"],
                   s=s, marker=m, color=c, edgecolor="black", linewidth=1.0, zorder=4)

    ax.set_xlim(*SCORE_RANGE)
    ax.set_ylim(*SCORE_RANGE)
    ax.set_xlabel("Business goal (proposal quality)")
    ax.set_ylabel("Ethics goal (constitution score)")
    ax.set_title(f"{scenario} · {model} · {prompt_type}", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Consistent legend across all plots: full intervention list in fixed order
    handles = []
    for iid in LEGEND_ORDER:
        style = INTERVENTION_STYLE.get(iid)
        if style is None:
            continue
        handles.append(plt.scatter(
            [], [], s=80, marker=style["marker"],
            color=style["color"], edgecolor="black", linewidth=0.7,
            label=LEGEND_LABEL.get(iid, CLEAN_LABEL.get(iid, iid)),
        ))
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=True, title="Intervention",
              title_fontsize=9)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return fig


def plot_topology_summary(
    cells: pd.DataFrame, df: pd.DataFrame, prompt_type: str, out_path,
    models: list[str] | None = None,
    n_bootstrap: int = 10000, seed: int = 0,
) -> plt.Figure | None:
    """Appendix cross-model summary: team-vs-single and partner-vs-single Δ
    only (no interventions). All models laid out side-by-side along the x-axis
    of a single 1×2 figure (cols = [mean Δ, 10th-pct Δ]). Each model contributes
    two adjacent bars: team baseline (flat topology) and partner baseline
    (hierarchical topology). Both bars are gold-bordered as topology baselines;
    sig coloring (3-state) is the same as plot_safety_bars_vs_single.
    """
    pt_cells = cells[cells["prompt_type"] == prompt_type]
    pt_df = df[df["prompt_type"] == prompt_type]
    if pt_cells.empty or pt_df.empty:
        return None

    present = set(pt_cells["model"].unique())
    if models is None:
        ordered = [m for m in MODEL_ORDER if m in present]
        ordered += [m for m in sorted(present) if m not in MODEL_ORDER]
        models = ordered
    else:
        models = [m for m in models if m in present]
    if not models:
        return None

    orgs = ["team", "partner"]
    bar_label = {"team": "flat", "partner": "hier."}

    rng = np.random.default_rng(seed)

    # BCa per (model, org) for the 10th-pct column + single's own band.
    pct = 0.10
    worst = {}
    sb_se_band = {}  # model -> (lo, hi) for the row-1 single SE band
    for model in models:
        sub_df = pt_df[pt_df["model"] == model]
        sb_runs = sub_df[sub_df["intervention_id"] == "single"]["constitution_score"].dropna().values
        if len(sb_runs) < 2:
            continue
        sb_pt, sb_lo_a, sb_hi_a = _bca_pct(sb_runs, pct, n_bootstrap, rng)
        sb_band_lo = (sb_lo_a - sb_pt) if not (pd.isna(sb_lo_a) or pd.isna(sb_pt)) else 0.0
        sb_band_hi = (sb_hi_a - sb_pt) if not (pd.isna(sb_hi_a) or pd.isna(sb_pt)) else 0.0
        worst[(model, "__band__")] = (sb_band_lo, sb_band_hi)
        for org in orgs:
            org_runs = sub_df[sub_df["intervention_id"] == org]["constitution_score"].dropna().values
            d, lo, hi = _bca_delta(org_runs, sb_runs, pct, n_bootstrap, rng)
            worst[(model, org)] = (d, lo, hi)
        sb_cell = pt_cells[(pt_cells["model"] == model)
                            & (pt_cells["intervention_id"] == "single")]
        if not sb_cell.empty and not pd.isna(sb_cell.iloc[0].get("constitution_score_se")):
            sb = sb_cell.iloc[0]
            t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
            sb_se_band[model] = (-t_sb * sb["constitution_score_se"],
                                 +t_sb * sb["constitution_score_se"])

    # Group geometry: each model is a group of 2 adjacent bars; gap between groups.
    bar_w = 0.42
    intra = 0.48     # center-to-center distance within a model group
    inter = 1.5      # center-to-center distance between model groups
    n_models = len(models)
    group_centers = [i * inter for i in range(n_models)]
    bar_xs = {}
    for i, m in enumerate(models):
        c0 = group_centers[i] - intra / 2
        c1 = group_centers[i] + intra / 2
        bar_xs[(m, "team")] = c0
        bar_xs[(m, "partner")] = c1

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(9.0, 1.4 * n_models * 2 + 3.0), 4.6),
        sharex=False, sharey=False, squeeze=False,
    )

    light_fill = "#dddddd"
    gold = "#DAA520"

    # Y-limit aggregation
    all_lo_r1, all_hi_r1 = [], []
    all_lo_r2, all_hi_r2 = [], []
    for model in models:
        sub = pt_cells[pt_cells["model"] == model].set_index("intervention_id")
        for org in orgs:
            if org in sub.index:
                r = sub.loc[org]
                lo = r.get("constitution_score_ci_lo_vs_single",
                           r["constitution_score_delta_vs_single"])
                hi = r.get("constitution_score_ci_hi_vs_single",
                           r["constitution_score_delta_vs_single"])
                if not pd.isna(lo): all_lo_r1.append(lo)
                if not pd.isna(hi): all_hi_r1.append(hi)
        if model in sb_se_band:
            all_lo_r1.append(sb_se_band[model][0])
            all_hi_r1.append(sb_se_band[model][1])
        for org in orgs:
            d, lo, hi = worst.get((model, org), (np.nan, np.nan, np.nan))
            if not pd.isna(lo): all_lo_r2.append(lo)
            if not pd.isna(hi): all_hi_r2.append(hi)
        band = worst.get((model, "__band__"))
        if band is not None:
            all_lo_r2.append(band[0]); all_hi_r2.append(band[1])

    pad = 0.4
    y_min_r1 = min(all_lo_r1 + [0]) - pad
    y_max_r1 = max(all_hi_r1 + [0]) + pad
    y_min_r2 = min(all_lo_r2 + [0]) - pad
    y_max_r2 = max(all_hi_r2 + [0]) + pad

    def _sign_face_hatch(sign, c):
        if sign == "worse":
            return c, None
        if sign == "better":
            return light_fill, "//"
        return light_fill, None

    # ── Left column (mean Δ) ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.axhline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
    for model in models:
        c = MODEL_COLOR.get(model, "#555")
        sub = pt_cells[pt_cells["model"] == model].set_index("intervention_id")
        if model in sb_se_band:
            lo, hi = sb_se_band[model]
            x_left = group_centers[models.index(model)] - inter / 2
            x_right = group_centers[models.index(model)] + inter / 2
            ax.fill_betweenx([lo, hi], x_left, x_right,
                             color="#cccccc", alpha=0.35, zorder=0)
        for org in orgs:
            if org not in sub.index:
                continue
            r = sub.loc[org]
            mu = r["constitution_score_delta_vs_single"]
            ci_lo = r.get("constitution_score_ci_lo_vs_single", mu)
            ci_hi = r.get("constitution_score_ci_hi_vs_single", mu)
            sign = r.get("const_sign", "inconclusive")
            face, hatch = _sign_face_hatch(sign, c)
            err_lo = mu - ci_lo if not pd.isna(ci_lo) else 0
            err_hi = ci_hi - mu if not pd.isna(ci_hi) else 0
            x = bar_xs[(model, org)]
            ax.bar(x, mu, width=bar_w, color=face, edgecolor=gold,
                   linewidth=2.4, hatch=hatch, zorder=2)
            ax.errorbar(x, mu, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)

    ax.set_ylim(y_min_r1, y_max_r1)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Δ constitution score vs single", fontsize=10, fontweight="bold")
    ax.set_title("Mean Δ", fontsize=12, fontweight="bold", pad=18)

    # ── Right column (10th-pct Δ) ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.axhline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
    for model in models:
        c = MODEL_COLOR.get(model, "#555")
        band = worst.get((model, "__band__"))
        if band is not None:
            x_left = group_centers[models.index(model)] - inter / 2
            x_right = group_centers[models.index(model)] + inter / 2
            ax.fill_betweenx([band[0], band[1]], x_left, x_right,
                             color="#cccccc", alpha=0.35, zorder=0)
        for org in orgs:
            d, lo, hi = worst.get((model, org), (np.nan, np.nan, np.nan))
            if pd.isna(d):
                continue
            if pd.isna(lo) or pd.isna(hi):
                sign = "inconclusive"
            elif lo > 0:
                sign = "better"
            elif hi < 0:
                sign = "worse"
            else:
                sign = "same"
            face, hatch = _sign_face_hatch(sign, c)
            err_lo = max(0.0, d - lo) if not pd.isna(lo) else 0
            err_hi = max(0.0, hi - d) if not pd.isna(hi) else 0
            x = bar_xs[(model, org)]
            ax.bar(x, d, width=bar_w, color=face, edgecolor=gold,
                   linewidth=2.4, hatch=hatch, zorder=2)
            if (err_lo + err_hi) > 0:
                ax.errorbar(x, d, yerr=[[err_lo], [err_hi]], fmt="none",
                            ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)

    ax.set_ylim(y_min_r2, y_max_r2)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Δ constitution score vs single", fontsize=10, fontweight="bold")
    ax.set_title("10th-percentile Δ", fontsize=12, fontweight="bold", pad=18)

    # X-tick labels per bar (team / partner) for each axis; model name above each group.
    bar_tick_xs = []
    bar_tick_lbls = []
    for m in models:
        bar_tick_xs.append(bar_xs[(m, "team")])
        bar_tick_lbls.append(bar_label["team"])
        bar_tick_xs.append(bar_xs[(m, "partner")])
        bar_tick_lbls.append(bar_label["partner"])

    for col in (0, 1):
        ax = axes[0, col]
        ax.set_xticks(bar_tick_xs)
        ax.set_xticklabels(bar_tick_lbls, fontsize=9, ha="center")
        ax.set_xlim(group_centers[0] - inter / 2, group_centers[-1] + inter / 2)
        # Model name labels above each group, on the axis transform
        trans = ax.get_xaxis_transform()
        for i, m in enumerate(models):
            ax.text(group_centers[i], 1.02, m, transform=trans,
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=MODEL_COLOR.get(m, "#444"))

    fig.suptitle(
        f"Consultancy ({prompt_type}): topology effect on constitution Δ vs single",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.22, wspace=0.18)

    # Legend: row 1 = baselines + target range; row 2 = sig states.
    pick_color = MODEL_COLOR.get(models[0], "#555")
    A_dashed = mlines.Line2D([0], [0], color="#444", linestyle="--", linewidth=1.5,
                             label="single agent baseline (Δ=0)")
    A_gray = mpatches.Patch(facecolor="#cccccc", edgecolor="none", alpha=0.5,
                            label="single agent target range (95% CI)")
    multi_handle = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.4,
                                  label="multi-agent baseline")
    E = mpatches.Patch(facecolor=pick_color, edgecolor=gold, linewidth=2.4,
                       label="sig. less than single agent")
    F = mpatches.Patch(facecolor=light_fill, edgecolor=gold, linewidth=2.4,
                       label="not different from single agent")
    G = mpatches.Patch(facecolor=light_fill, edgecolor=gold, linewidth=2.4,
                       hatch="//", label="sig. greater than single agent")
    # fig.legend fills column-major; reorder so visible layout is:
    # row 1: [A_dashed, A_gray, multi_handle], row 2: [E, F, G]
    handles = [A_dashed, E, A_gray, F, multi_handle, G]
    lbls = [h.get_label() for h in handles]
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
