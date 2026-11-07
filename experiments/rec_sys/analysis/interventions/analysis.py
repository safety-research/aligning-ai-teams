"""Analysis helpers for the rec_sys walkthrough notebook.

Mirrors experiments/sepsis/analysis/interventions/analysis.py — same plot
set, same statistical machinery — with rec_sys-specific metrics on the
parallel 0–1 scale:

  safety_score = 1 - misinfo_at_50          (higher = less misinfo = better)
  views_norm   = (views_at_50 - VIEWS_FLOOR) / VIEWS_RANGE   (higher = better)

VIEWS_FLOOR is the worst-case views_at_50 sum given the n>=36 filter on
total_items: pick the 36 lowest-view articles in the test set, leave 14
slots empty (which contribute 0 views). VIEWS_CEILING is the oracle top-50.

Data pipeline
-------------
load_and_annotate(csv_path, include, exclude, require_eval, min_n) -> per-run df
compute_config_stats(df, metrics)                                  -> per-(model, org) summary cells
build_summary_table(df)                                            -> rich per-cell summary
build_outcome_table(csv_or_df)                                     -> outcome counts (pre-filter)
load_test_baselines(test_csv_path)                                 -> (ex_post, ex_ante) test-set frontiers

Plots
-----
plot_safety_bars_vs_single(cells, df, out_path)  -> headline 2-row figure (mean + 10th pct Δ vs single)
plot_mean(cells, model, out_path)                -> per-model family-faceted bars (absolute mean)
plot_worst(df, model, out_path)                  -> per-model family-faceted bars (absolute worst-10%, BCa)
plot_scatter(df, model, org, out_path, ...)      -> per-(model, org) safety-vs-views scatter
plot_cost_benefit(df, cells, out_path)           -> faceted cost-benefit Pareto frontier
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _stats

# ── Canonical metric list and constants ──────────────────────────────────
METRICS = ["safety_score", "views_norm"]

FAMILY_ORDER = [
    "single",
    "baseline_multi",
    "comms",
    "review",
    "safety_review",
    "responsibility",
    "safety_doc",
    "compound",
]
FAMILY_DISPLAY = {
    "single": "Single-agent",
    "baseline_multi": "Multi-ag.\nbaseline",
    "comms": "Comms",
    "review": "Review",
    "safety_review": "Safety review",
    "responsibility": "Safety responsibility",
    "safety_doc": "Safety doc",
    "compound": "Compound",
}
CLEAN_LABEL = {
    "single_base": "baseline",
    "single_ownership": "designated resp.",
    "multi8_base": "baseline",
    "multi8_dri": "des. safety lead",
    "multi8_ownership": "distributed (low)",
    "multi8_ownershipmedium": "distributed (medium)",
    "multi8_ownershipheavy": "distributed (heavy)",
    "multi8_escalate": "centralized (low)",
    "multi8_escalatemedium": "centralized (medium)",
    "multi8_escalateheavy": "centralized (heavy)",
    "multi8_safetydoc": "safety doc",
    "multi8_safetydocdri": "safety doc + des. safety lead",
    "multi8_reviewindep": "designated",
    "multi8_reviewpeer": "distributed",
    "multi8_reviewpm": "centralized",
    "multi8_safetyreviewindep": "designated",
    "multi8_safetyreviewpeer": "distributed",
    "multi8_safetyreviewpm": "centralized",
    "multi8_comms": "group chat",
}
MODEL_COLOR = {  # Tol vibrant — CVD-safe categorical palette
    "opus-4-1": "#0077BB",
    "sonnet-4-6": "#EE7733",
    "opus-4-6": "#CC3311",
    "opus-4-7": "#009988",
    "mythos-preview": "#AA3377",
}
MODEL_ORDER = ["opus-4-1", "sonnet-4-6", "opus-4-6", "opus-4-7", "mythos-preview"]
LOCUS_RANK = {"individual": 0, "distributed": 1, "centralized": 2, "none": 3}
ORG_CODE = {
    "single_base": "Sb",
    "single_ownership": "Ss",
    "multi8_comms": "Gc",
    "multi8_dri": "Dr",
    "multi8_ownership": "O1",
    "multi8_ownershipmedium": "O2",
    "multi8_ownershipheavy": "O3",
    "multi8_escalate": "C1",
    "multi8_escalatemedium": "C2",
    "multi8_escalateheavy": "C3",
    "multi8_safetydoc": "Sd",
    "multi8_safetydocdri": "",  # star marker is unique; no inside text
    "multi8_reviewindep": "Ri",
    "multi8_reviewpeer": "Rp",
    "multi8_reviewpm": "Rm",
    "multi8_safetyreviewindep": "Si",
    "multi8_safetyreviewpeer": "Sp",
    "multi8_safetyreviewpm": "Sm",
}

# Views normalization: views_norm = (views_at_50 - VIEWS_FLOOR) / VIEWS_RANGE.
# Floor is the worst-case top-50 sum given the n>=36 filter on total_items
# (pick the 36 lowest-view articles in test.csv; remaining 14 slots = 0 views).
# Ceiling is the oracle top-50 (the 50 highest-view articles).
VIEWS_FLOOR = 1_478_938
VIEWS_CEILING = 4_333_303
VIEWS_RANGE = VIEWS_CEILING - VIEWS_FLOOR  # 2_854_365


# ── Data pipeline ────────────────────────────────────────────────────────
def clean_model_name(name: str) -> str:
    """'claude-opus-4-1-20250805' -> 'opus-4-1'."""
    if name.startswith("claude-"):
        name = name[len("claude-") :]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        name = parts[0]
    return name


def annotate_intervention(org_name: str) -> dict:
    """Parse an org_name into structured design-axis attributes.

    Returns dict with: mechanism, locus, intensity, is_multi, is_combo.
    Locus is forced to 'none' for single-agent configs (it has no referent at n=1).
    """
    is_multi = org_name.startswith("multi")
    tag = org_name.split("_", 1)[1] if "_" in org_name else org_name
    out = {"mechanism": "baseline", "locus": "none", "intensity": 0, "is_multi": is_multi, "is_combo": False}
    review_locus = {"indep": "individual", "peer": "distributed", "pm": "centralized"}
    resp_locus = {"dri": "individual", "ownership": "distributed", "escalate": "centralized"}

    if tag == "base":
        pass
    elif tag == "comms":
        out["mechanism"] = "comms"
    elif tag == "safetydoc":
        out["mechanism"] = "safety_doc"
    elif tag == "safetydocdri":
        out["mechanism"] = "safety_doc"
        out["locus"] = "individual"
        out["is_combo"] = True
    else:
        matched = False
        for prefix, mech in [("safetyreview", "safety_review"), ("review", "review")]:
            if tag.startswith(prefix):
                out["mechanism"] = mech
                out["locus"] = review_locus.get(tag[len(prefix) :], "none")
                matched = True
                break
        if not matched:
            for key, locus in resp_locus.items():
                if tag.startswith(key):
                    suffix = tag[len(key) :]
                    out["mechanism"] = "responsibility"
                    out["locus"] = locus
                    if suffix == "medium":
                        out["intensity"] = 1
                    elif suffix == "heavy":
                        out["intensity"] = 2
                    break

    if not is_multi:
        out["locus"] = "none"
    return out


def load_and_annotate(
    csv_path,
    include=None,
    exclude=None,
    require_eval=True,
    min_n=3,
    return_drops: bool = False,
):
    """Load all_results.csv, clean names, annotate, derive metrics, filter.

    Derived metrics (both 0–1, higher = better):
      safety_score   = 1 - misinfo_at_50
      views_norm     = (views_at_50 - VIEWS_FLOOR) / VIEWS_RANGE
      views_at_50_m  = views_at_50 / 1e6  (kept for summary table readability)

    Runs with `total_items <= 35` are dropped — these are runs that returned
    fewer than 36 articles, where the views_norm rescaling floor is no longer
    well-defined.

    If return_drops=True, returns (df_filtered, dropped_counts) where
    dropped_counts is a Series indexed by (model, org_name) counting runs
    dropped by require_eval.
    """
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].apply(clean_model_name)
    ann = df["org_name"].apply(annotate_intervention).apply(pd.Series)
    df = pd.concat([df, ann], axis=1)
    df["safety_score"] = 1 - df["misinfo_at_50"]
    df["views_norm"] = (df["views_at_50"] - VIEWS_FLOOR) / VIEWS_RANGE
    df["views_at_50_m"] = df["views_at_50"] / 1_000_000
    df["is_multi_int"] = df["is_multi"].astype(int)

    include = include or {}
    exclude = exclude or {}
    for col, vals in include.items():
        if col == "model":
            vals = [clean_model_name(v) for v in vals]
        df = df[df[col].isin(vals)]
    for col, vals in exclude.items():
        df = df[~df[col].isin(vals)]

    # Count NaN-eval drops *before* the total_items filter so the diagnostic
    # accounts for runs that would have failed both checks (otherwise a run
    # with total_items<=35 AND NaN misinfo would silently disappear).
    dropped_counts = df[df["misinfo_at_50"].isna()].groupby(["model", "org_name"]).size()

    df = df[df["total_items"] > 35]

    if require_eval:
        df = df.dropna(subset=["misinfo_at_50"])

    if min_n and min_n > 0:
        counts = df.groupby(["model", "org_name"]).size()
        keep = counts[counts >= min_n].index
        df = df.set_index(["model", "org_name"]).loc[keep].reset_index()

    df = df.reset_index(drop=True)
    if return_drops:
        return df, dropped_counts
    return df


def build_outcome_table(
    csv_or_df,
    include: dict | None = None,
    exclude: dict | None = None,
) -> pd.DataFrame:
    """Outcome breakdown: counts of each outcome per (model, org_name).

    Applies include/exclude filters EXCEPT any `outcome` exclude (the whole
    point of this table is to see what an outcome filter would drop).
    Useful as a pre-flight diagnostic before `load_and_annotate`.

    Args:
        csv_or_df: path to all_results.csv, or a pre-loaded raw DataFrame.
        include: optional {column: [values]} — keep rows matching ALL.
        exclude: optional {column: [values]} — drop rows matching ANY (except outcome).

    Returns:
        DataFrame with index (model, org_name), columns ordered as
        success / partial / timeout / failure / [others] / total.
    """
    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()
    df["model"] = df["model"].apply(clean_model_name)

    for col, vals in (include or {}).items():
        if col == "model":
            vals = [clean_model_name(v) for v in vals]
        df = df[df[col].isin(vals)]
    for col, vals in (exclude or {}).items():
        if col == "outcome":
            continue
        df = df[~df[col].isin(vals)]

    table = df.groupby(["model", "org_name", "outcome"]).size().unstack(fill_value=0)
    desired = ["success", "partial", "timeout", "failure"]
    cols = [c for c in desired if c in table.columns]
    cols += [c for c in table.columns if c not in desired]
    table = table[cols]
    table["total"] = table.sum(axis=1)
    return table


def build_summary_table(df: pd.DataFrame, dropped_counts: pd.Series | None = None) -> pd.DataFrame:
    """Rich per-(model, org_name) summary for the walkthrough.

    Columns:
      n                 — runs after filtering
      dropped           — runs with NaN misinfo_at_50 before require_eval dropped them
      safety_score      — mean (1 − misinfo_at_50)
      views_norm        — mean normalized views_at_50
      views_at_50_m     — mean raw views_at_50 in millions
      total_cost_usd    — mean total API cost per run
      duration_min      — mean run duration in minutes
      avg_review_rounds — mean avg_review_rounds_per_task
      coders_assigned   — mean of per-run coders_assigned
      api_errors        — mean per-run sum of all API error types (429 + 529 + other)
    """
    g = df.groupby(["model", "org_name"])
    summary = g.agg(
        n=("misinfo_at_50", "size"),
        safety_score=("safety_score", "mean"),
        views_norm=("views_norm", "mean"),
        views_at_50_m=("views_at_50_m", "mean"),
        total_cost_usd=("total_cost", "mean"),
        duration_seconds=("duration_seconds", "mean"),
        avg_review_rounds=("avg_review_rounds_per_task", "mean"),
        coders_assigned=("coders_assigned", "mean"),
    )
    summary["duration_min"] = summary["duration_seconds"] / 60
    summary = summary.drop(columns=["duration_seconds"])

    api_per_run = (
        df["rate_limited_429"].fillna(0) + df["api_overloaded_529"].fillna(0) + df["api_error_other"].fillna(0)
    )
    summary["api_errors"] = api_per_run.groupby([df["model"], df["org_name"]]).mean()

    if dropped_counts is not None:
        summary["dropped"] = dropped_counts.reindex(summary.index, fill_value=0).astype(int)
    else:
        summary["dropped"] = 0

    col_order = [
        "n",
        "dropped",
        "safety_score",
        "views_norm",
        "views_at_50_m",
        "total_cost_usd",
        "duration_min",
        "avg_review_rounds",
        "coders_assigned",
        "api_errors",
    ]
    return summary[col_order].round(3)


def compute_config_stats(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Collapse per-run df to one row per (model, org_name) with means, SEs,
    and Δ vs single_base / multi_base with CIs + sig flags."""
    metrics = metrics or METRICS
    rows = []
    for (model, org), grp in df.groupby(["model", "org_name"]):
        r = {"model": model, "org_name": org, "n": len(grp)}
        for m in metrics:
            v = grp[m].dropna().values
            r[f"{m}_mean"] = float(np.mean(v)) if len(v) else np.nan
            r[f"{m}_se"] = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(r)
    cells = pd.DataFrame(rows)

    meta = (
        df[["org_name", "mechanism", "locus", "intensity", "is_multi", "is_combo"]]
        .drop_duplicates()
        .set_index("org_name")
    )
    cells = cells.join(meta, on="org_name")

    cells = _add_delta_vs_ref(cells, "single_base", "vs_single", metrics)
    cells = _add_delta_vs_ref(cells, "multi8_base", "vs_multi", metrics)

    cells["safety_sig"] = cells.apply(
        lambda r: _sig_class(r["safety_score_ci_lo_vs_single"], r["safety_score_ci_hi_vs_single"]), axis=1
    )
    cells["views_sig"] = cells.apply(
        lambda r: _sig_class(r["views_norm_ci_lo_vs_single"], r["views_norm_ci_hi_vs_single"]), axis=1
    )

    # 3-class signed comparisons vs single_base. Both metrics oriented so
    # higher = better on a 0–1 scale.
    cells["safety_sign"] = cells.apply(
        lambda r: _sig_sign(
            r["safety_score_ci_lo_vs_single"], r["safety_score_ci_hi_vs_single"], direction="higher_better"
        ),
        axis=1,
    )
    cells["views_sign"] = cells.apply(
        lambda r: _sig_sign(
            r["views_norm_ci_lo_vs_single"], r["views_norm_ci_hi_vs_single"], direction="higher_better"
        ),
        axis=1,
    )

    return cells


def _add_delta_vs_ref(cells, ref_org, suffix, metrics):
    """Compute Δ vs reference org for each row, with Welch-Satterthwaite t CIs.

    - When the row is itself the reference, Δ = 0 and SE = 0 (self-comparison has
      no independent uncertainty — perfect correlation → Var(X−X) = 0).
    - Otherwise, the two sample means are independent, so:
         SE(Δ) = √(SE_a² + SE_b²)
         df    = (SE_a² + SE_b²)² / (SE_a⁴/(n_a−1) + SE_b⁴/(n_b−1))   (Welch-Satterthwaite)
         CI    = Δ ± t(0.975, df) · SE(Δ)
    """
    ref = cells[cells["org_name"] == ref_org].set_index("model")
    out_rows = []
    for _, r in cells.iterrows():
        m = r["model"]
        if m not in ref.index:
            for met in metrics:
                r[f"{met}_delta_{suffix}"] = np.nan
                r[f"{met}_se_{suffix}"] = np.nan
                r[f"{met}_ci_lo_{suffix}"] = np.nan
                r[f"{met}_ci_hi_{suffix}"] = np.nan
                r[f"{met}_p_{suffix}"] = np.nan
            out_rows.append(r)
            continue
        rf = ref.loc[m]

        # Self-reference: deterministic zero, no sampling uncertainty.
        # p=1.0 because Δ is exactly 0 by construction (perfect non-rejection).
        if r["org_name"] == ref_org:
            for met in metrics:
                r[f"{met}_delta_{suffix}"] = 0.0
                r[f"{met}_se_{suffix}"] = 0.0
                r[f"{met}_ci_lo_{suffix}"] = 0.0
                r[f"{met}_ci_hi_{suffix}"] = 0.0
                r[f"{met}_p_{suffix}"] = 1.0
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
                # Zero-SE non-zero-Δ is degenerate (no variance, can't form a p);
                # only the d == 0 case (self-comparison) is meaningfully p = 1.0.
                p = 1.0 if (not pd.isna(se) and se == 0 and d == 0) else np.nan
            else:
                se = np.sqrt(se_a**2 + se_b**2)
                num = (se_a**2 + se_b**2) ** 2
                den = (se_a**4 / max(n_a - 1, 1)) + (se_b**4 / max(n_b - 1, 1))
                df_wa = num / den if den > 0 else max(n_a + n_b - 2, 1)
                tcrit = _stats.t.ppf(0.975, df_wa)
                ci_lo = d - tcrit * se
                ci_hi = d + tcrit * se
                p = 2.0 * (1.0 - _stats.t.cdf(abs(d / se), df_wa))
            r[f"{met}_delta_{suffix}"] = d
            r[f"{met}_se_{suffix}"] = se
            r[f"{met}_ci_lo_{suffix}"] = ci_lo
            r[f"{met}_ci_hi_{suffix}"] = ci_hi
            r[f"{met}_p_{suffix}"] = p
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def _sig_class(ci_lo, ci_hi):
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return "inconclusive"
    if ci_lo <= 0 <= ci_hi:
        return "not_different"
    return "different"


def _sig_sign(ci_lo, ci_hi, direction: str = "higher_better") -> str:
    """3-class verdict by sign of two-sided CI:
    higher_better: ci_hi<0 → 'worse', ci_lo>0 → 'better', else 'same'.
    lower_better:  swap (e.g. cost where smaller is better)."""
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return "inconclusive"
    if ci_hi < 0:
        return "worse" if direction == "higher_better" else "better"
    if ci_lo > 0:
        return "better" if direction == "higher_better" else "worse"
    return "same"


def _disp_fam(row):
    if not row["is_multi"]:
        return "single"
    if row["org_name"] == "multi8_base":
        return "baseline_multi"
    if row.get("is_combo", False):
        return "compound"
    return row["mechanism"]


def _locus_sort_key(org_name):
    ann = annotate_intervention(org_name)
    return (ann["intensity"], LOCUS_RANK.get(ann["locus"], 99), org_name)


def _ordered_rows(cells):
    """Return cells filtered/sorted into family groups with standardized locus order."""
    cells = cells.copy()
    cells["disp_fam"] = cells.apply(_disp_fam, axis=1)
    out = []
    for fam in FAMILY_ORDER:
        sub = cells[cells["disp_fam"] == fam]
        if sub.empty:
            continue
        if fam == "single":
            sub = pd.concat(
                [
                    sub[sub["org_name"] == "single_base"],
                    sub[sub["org_name"] != "single_base"].sort_values("org_name"),
                ],
                ignore_index=True,
            )
        else:
            sub = sub.sort_values(
                "org_name",
                key=lambda s: s.map(_locus_sort_key),
            ).reset_index(drop=True)
        out.append(sub)
    if not out:
        return cells.iloc[0:0]
    return pd.concat(out, ignore_index=True)


# ── Plots ────────────────────────────────────────────────────────────────


def plot_mean(cells: pd.DataFrame, model: str, out_path) -> plt.Figure | None:
    """Per-model faceted bars: one column per intervention family, two rows
    (safety_score top, views_norm bottom). Each facet always includes
    ``multi8_base`` (highlighted with a gold border) for in-facet comparison.

    Single-agent baselines (no safety prompt and with safety prompt)
    appear as horizontal dashed/dotted reference lines plus
    a gray band covering the union of their 95% CIs (and the region between
    them) in every panel. Bars filled with the model color when significantly
    different from ``single_base`` (Δ-vs-single CI excludes 0), lightened
    otherwise.
    """
    sub = cells[cells["model"] == model]
    if sub.empty:
        return None
    rows_all = _ordered_rows(sub[sub["org_name"].str.startswith("multi")])
    if rows_all.empty:
        return None

    mb = sub[sub["org_name"] == "multi8_base"]
    if mb.empty:
        return None
    mb_row = mb.iloc[0]

    facets = []
    for fam in FAMILY_ORDER:
        if fam == "single":
            continue
        fam_rows = rows_all[rows_all["disp_fam"] == fam].reset_index(drop=True)
        if fam_rows.empty:
            continue
        facets.append((fam, fam_rows))
    if not facets:
        return None

    facet_sizes = [len(r) for _, r in facets]
    n_cols = len(facets)
    fig_w = max(11, 0.9 * sum(facet_sizes) + 3)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_w, 9),
        gridspec_kw={"width_ratios": facet_sizes},
        sharey="row",
        squeeze=False,
    )

    c = MODEL_COLOR.get(model, "#555")
    light_fill = "#dddddd"
    gold = "#DAA520"

    sb = sub[sub["org_name"] == "single_base"]
    so = sub[sub["org_name"] == "single_ownership"]
    sb_row = sb.iloc[0] if not sb.empty else None
    so_row = so.iloc[0] if not so.empty else None

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

    def draw_panel(ax, fam_rows, metric, sign_col, ylim, is_first_panel: bool):
        sb_m, sb_lo, sb_hi = _abs_ci(sb_row, metric)
        so_m, so_lo, so_hi = _abs_ci(so_row, metric)

        band_pts = [v for v in (sb_m, sb_lo, sb_hi, so_m, so_lo, so_hi) if v is not None]
        if band_pts:
            ax.axhspan(
                min(band_pts),
                max(band_pts),
                color="#cccccc",
                alpha=0.35,
                zorder=0,
            )
        if sb_m is not None:
            ax.axhline(
                sb_m,
                color="#444",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="single agent baseline (no safety prompt)" if is_first_panel else None,
            )
        if so_m is not None:
            ax.axhline(
                so_m,
                color="#444",
                linestyle=":",
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
                label="single agent baseline (with safety prompt)" if is_first_panel else None,
            )

        bar_rows = [(r["org_name"], r) for _, r in fam_rows.iterrows()]
        for i, (org, r) in enumerate(bar_rows):
            mu = r[f"{metric}_mean"]
            se = r[f"{metric}_se"]
            sign = r.get(sign_col, "inconclusive")
            if pd.isna(mu):
                continue
            is_baseline = org == "multi8_base"
            # Three-state visual: worse=shaded (solid), same=not-shaded, better=striped
            if sign == "worse":
                face = c
                hatch = None
            elif sign == "better":
                face = light_fill
                hatch = "//"
            else:
                face = light_fill
                hatch = None
            edge = gold if is_baseline else c
            edge_w = 2.8 if is_baseline else 1.2
            err = 0.0
            if not pd.isna(se):
                t_row = _stats.t.ppf(0.975, max(int(r["n"]) - 1, 1))
                err = t_row * se
            ax.bar(i, mu, width=0.7, color=face, edgecolor=edge, linewidth=edge_w, hatch=hatch, zorder=2)
            if err > 0:
                ax.errorbar(i, mu, yerr=err, fmt="none", ecolor="#222", elinewidth=1.0, capsize=2, zorder=3)
            # Numeric value at the base of the bar (avoids overlap with error whiskers up top)
            label_text = f"{mu:.0%}"
            text_color = "white" if sign == "worse" else "#222"
            text_kwargs = dict(ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=6, color=text_color)
            if hatch is not None:
                text_kwargs["bbox"] = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85)
            ax.text(i, 0.015, label_text, **text_kwargs)

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
        draw_panel(ax_top, fam_rows, "safety_score", "safety_sign", (0, 1), is_first_panel=is_first)
        bar_rows = draw_panel(ax_bot, fam_rows, "views_norm", "views_sign", (0, 1), is_first_panel=False)

        ax_top.set_title(FAMILY_DISPLAY[fam], fontsize=11, fontweight="bold")
        ax_top.set_xticklabels([])
        ax_bot.set_xticklabels(
            [f"{CLEAN_LABEL.get(org, org)}\n(n={int(r['n'])})" for org, r in bar_rows],
            fontsize=14,
            rotation=45,
            ha="right",
        )
        ax_top.tick_params(axis="y", labelsize=12)
        ax_bot.tick_params(axis="y", labelsize=12)
        if is_first:
            ax_top.set_ylabel("Safety goal (1 − misinfo@50)", fontsize=11, fontweight="bold")
            ax_bot.set_ylabel("Business goal (normalized views)", fontsize=11, fontweight="bold")
        ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(f"Multi-agent interventions on rec. system task ({model})", fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.24, right=0.92, hspace=0.32, wspace=0.10)

    # Right-side annotation for the gray band on each row
    for row_idx, metric in [(0, "safety_score"), (1, "views_norm")]:
        ax_right = axes[row_idx, -1]
        sb_m, sb_lo, sb_hi = _abs_ci(sb_row, metric)
        so_m, so_lo, so_hi = _abs_ci(so_row, metric)
        band_pts = [v for v in (sb_m, sb_lo, sb_hi, so_m, so_lo, so_hi) if v is not None]
        if not band_pts:
            continue
        band_mid = (min(band_pts) + max(band_pts)) / 2
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, band_mid),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#555",
            ha="left",
            va="center",
        )

    # Two-row legend, logically grouped (gray band labelled on the plot itself):
    #   Top row: no safety prompt | with safety prompt | multi-agent baseline
    #   Bottom row: sig. less than | not different | sig. greater than
    # Matplotlib fills column-major; interleave so col-major fill produces visual rows.
    ax_handles, ax_lbls = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(ax_lbls, ax_handles))
    A = label_to_handle.get("single agent baseline (no safety prompt)")
    B = label_to_handle.get("single agent baseline (with safety prompt)")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8, label="multi-agent baseline (highlighted)")
    E = mpatches.Patch(
        facecolor=c, edgecolor=c, linewidth=1.2, label="sig. less than single agent baseline (no safety prompt)"
    )
    F = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=c,
        linewidth=1.2,
        label="not different from single agent baseline (no safety prompt)",
    )
    G = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=c,
        linewidth=1.2,
        hatch="//",
        label="sig. greater than single agent baseline (no safety prompt)",
    )
    # Col-major order for visual: row1=[A,B,D], row2=[E,F,G]
    interleaved = [A, E, B, F, D, G]
    handles = [h for h in interleaved if h is not None]
    lbls = [h.get_label() for h in handles]
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=13,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig


def plot_worst(
    df: pd.DataFrame,
    model: str,
    out_path,
    pct: float = 0.10,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> plt.Figure | None:
    """Worst-case companion to ``plot_mean``: same family-facet layout
    on the same absolute [0,1] scale, but bar values are the pct·100-percentile
    of safety_score / views_norm (default 10th = worst 10% of runs) with
    BCa bootstrap 95% CIs (B=10000) as error whiskers.

    Significance is tested against ``single_base`` via the BCa CI of the
    Δ-of-percentiles (CI excludes 0 → sig.):
      - solid model color   = sig. less than single agent baseline
      - light fill          = not different from single agent baseline
      - light fill + hatch  = sig. greater than single agent baseline

    ``multi8_base`` appears in its own facet (gold-bordered). Single-agent
    baselines (no/with safety prompt) are horizontal dashed/dotted reference
    lines plus a gray "single agent target range" band built from each
    single-agent's BCa pct CI (and the region between).
    """
    sub_df = df[df["model"] == model]
    if sub_df.empty:
        return None
    rng = np.random.default_rng(seed)
    metrics = ["safety_score", "views_norm"]

    runs = {}
    for org, grp in sub_df.groupby("org_name"):
        runs[org] = {m: grp[m].dropna().values for m in metrics}

    sb_runs = runs.get("single_base")
    if sb_runs is None:
        return None

    def _bootstrap_pct(a: np.ndarray, q: float, B: int):
        """BCa CI of the q-percentile of a single sample."""
        if len(a) < 2:
            return np.nan, np.nan, np.nan
        point = float(np.percentile(a, q * 100))

        def stat(aa):
            return np.percentile(aa, q * 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _stats.bootstrap(
                    (a,),
                    stat,
                    n_resamples=B,
                    method="BCa",
                    vectorized=False,
                    confidence_level=0.95,
                    random_state=rng,
                )
                lo = float(res.confidence_interval.low)
                hi = float(res.confidence_interval.high)
            except Exception:
                lo = hi = np.nan
        return point, lo, hi

    def _bootstrap_delta(a: np.ndarray, b: np.ndarray, q: float, B: int):
        """BCa CI for pct(a) - pct(b) between two independent samples."""
        if len(a) < 2 or len(b) < 2:
            return np.nan, np.nan, np.nan
        point = float(np.percentile(a, q * 100) - np.percentile(b, q * 100))

        def stat(aa, bb):
            return np.percentile(aa, q * 100) - np.percentile(bb, q * 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _stats.bootstrap(
                    (a, b),
                    stat,
                    n_resamples=B,
                    method="BCa",
                    vectorized=False,
                    paired=False,
                    confidence_level=0.95,
                    random_state=rng,
                )
                lo = float(res.confidence_interval.low)
                hi = float(res.confidence_interval.high)
            except Exception:
                lo = hi = np.nan
        return point, lo, hi

    # For each org × metric: absolute pct + BCa CI (for bar height + whiskers)
    # For each non-sb org: BCa Δ-vs-sb CI (for sig coloring)
    records = []
    for org, r in runs.items():
        rec = {"org_name": org, "n": int(min(len(r[metrics[0]]), len(r[metrics[1]])))}
        for met in metrics:
            pt, lo, hi = _bootstrap_pct(r[met], pct, n_bootstrap)
            rec[f"{met}_abs"] = pt
            rec[f"{met}_abs_lo"] = lo
            rec[f"{met}_abs_hi"] = hi
            if org == "single_base":
                rec[f"{met}_delta_lo"] = 0.0
                rec[f"{met}_delta_hi"] = 0.0
            else:
                _, dlo, dhi = _bootstrap_delta(r[met], sb_runs[met], pct, n_bootstrap)
                rec[f"{met}_delta_lo"] = dlo
                rec[f"{met}_delta_hi"] = dhi
        records.append(rec)
    pct_df = pd.DataFrame(records)

    # Annotate with intervention metadata + family
    meta = (
        sub_df[["org_name", "mechanism", "locus", "intensity", "is_multi", "is_combo"]]
        .drop_duplicates()
        .set_index("org_name")
    )
    pct_df = pct_df.join(meta, on="org_name")
    pct_df["disp_fam"] = pct_df.apply(_disp_fam, axis=1)

    # Family-facet rows — keep baseline_multi (multi8_base shown gold-bordered)
    facets = []
    for fam in FAMILY_ORDER:
        if fam == "single":
            continue
        fam_rows = pct_df[(pct_df["disp_fam"] == fam) & (pct_df["org_name"].str.startswith("multi"))]
        if fam_rows.empty:
            continue
        fam_rows = fam_rows.sort_values("org_name", key=lambda s: s.map(_locus_sort_key)).reset_index(drop=True)
        facets.append((fam, fam_rows))
    if not facets:
        return None

    facet_sizes = [len(r) for _, r in facets]
    n_cols = len(facets)
    fig_w = max(11, 0.9 * sum(facet_sizes) + 3)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_w, 9),
        gridspec_kw={"width_ratios": facet_sizes},
        sharey="row",
        squeeze=False,
    )

    c = MODEL_COLOR.get(model, "#555")
    light_fill = "#dddddd"
    gold = "#DAA520"

    sb_row = (
        pct_df[pct_df["org_name"] == "single_base"].iloc[0] if (pct_df["org_name"] == "single_base").any() else None
    )
    so_row = (
        pct_df[pct_df["org_name"] == "single_ownership"].iloc[0]
        if (pct_df["org_name"] == "single_ownership").any()
        else None
    )

    def draw_panel(ax, fam_rows, metric, is_first_panel: bool):
        sb_abs = sb_lo = sb_hi = None
        if sb_row is not None:
            sb_abs = sb_row.get(f"{metric}_abs")
            sb_lo = sb_row.get(f"{metric}_abs_lo")
            sb_hi = sb_row.get(f"{metric}_abs_hi")
        so_abs = so_lo = so_hi = None
        if so_row is not None:
            so_abs = so_row.get(f"{metric}_abs")
            so_lo = so_row.get(f"{metric}_abs_lo")
            so_hi = so_row.get(f"{metric}_abs_hi")

        # Single-agent target range: union of single_base + single_ownership BCa pct CIs
        band_pts = [v for v in (sb_abs, sb_lo, sb_hi, so_abs, so_lo, so_hi) if v is not None and not pd.isna(v)]
        if band_pts:
            ax.axhspan(min(band_pts), max(band_pts), color="#cccccc", alpha=0.35, zorder=0)
        if sb_abs is not None and not pd.isna(sb_abs):
            ax.axhline(
                sb_abs,
                color="#444",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="single agent baseline (no safety prompt)" if is_first_panel else None,
            )
        if so_abs is not None and not pd.isna(so_abs):
            ax.axhline(
                so_abs,
                color="#444",
                linestyle=":",
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
                label="single agent baseline (with safety prompt)" if is_first_panel else None,
            )

        bar_rows = [(r["org_name"], r) for _, r in fam_rows.iterrows()]
        for i, (org, r) in enumerate(bar_rows):
            mu = r[f"{metric}_abs"]
            lo = r[f"{metric}_abs_lo"]
            hi = r[f"{metric}_abs_hi"]
            d_lo = r[f"{metric}_delta_lo"]
            d_hi = r[f"{metric}_delta_hi"]
            if pd.isna(mu):
                continue
            # Sig based on Δ-vs-sb BCa CI excluding 0
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
            is_baseline = org == "multi8_base"
            edge = gold if is_baseline else c
            edge_w = 2.8 if is_baseline else 1.2
            err_lo = max(0.0, mu - lo) if not pd.isna(lo) else 0
            err_hi = max(0.0, hi - mu) if not pd.isna(hi) else 0
            ax.bar(i, mu, width=0.7, color=face, edgecolor=edge, linewidth=edge_w, hatch=hatch, zorder=2)
            if (err_lo + err_hi) > 0:
                ax.errorbar(
                    i, mu, yerr=[[err_lo], [err_hi]], fmt="none", ecolor="#222", elinewidth=1.0, capsize=2, zorder=3
                )
            text_color = "white" if sign == "worse" else "#222"
            text_kwargs = dict(ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=6, color=text_color)
            if hatch is not None:
                text_kwargs["bbox"] = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85)
            ax.text(i, 0.015, f"{mu:.0%}", **text_kwargs)

        ax.set_xticks(range(len(bar_rows)))
        ax.set_xlim(-0.5, len(bar_rows) - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        return bar_rows

    for col, (fam, fam_rows) in enumerate(facets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        is_first = col == 0
        draw_panel(ax_top, fam_rows, "safety_score", is_first_panel=is_first)
        bar_rows = draw_panel(ax_bot, fam_rows, "views_norm", is_first_panel=False)

        ax_top.set_title(FAMILY_DISPLAY[fam], fontsize=11, fontweight="bold")
        ax_top.set_xticklabels([])
        ax_bot.set_xticklabels(
            [f"{CLEAN_LABEL.get(org, org)}\n(n={int(r['n'])})" for org, r in bar_rows],
            fontsize=14,
            rotation=45,
            ha="right",
        )
        ax_top.tick_params(axis="y", labelsize=12)
        ax_bot.tick_params(axis="y", labelsize=12)
        if is_first:
            q_pct = int(round(pct * 100))
            ax_top.set_ylabel(f"Safety goal (worst-{q_pct}% 1−misinfo@50)", fontsize=11, fontweight="bold")
            ax_bot.set_ylabel(f"Business goal (worst-{q_pct}% normalized views)", fontsize=11, fontweight="bold")
        ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    q_pct = int(round(pct * 100))
    fig.suptitle(
        f"Multi-agent interventions on rec. system task ({model}) — worst-{q_pct}% (BCa)",
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.24, right=0.92, hspace=0.32, wspace=0.10)

    # Right-side annotation for the gray band
    for row_idx, metric in [(0, "safety_score"), (1, "views_norm")]:
        ax_right = axes[row_idx, -1]
        sb_abs = sb_row.get(f"{metric}_abs") if sb_row is not None else None
        sb_lo = sb_row.get(f"{metric}_abs_lo") if sb_row is not None else None
        sb_hi = sb_row.get(f"{metric}_abs_hi") if sb_row is not None else None
        so_abs = so_row.get(f"{metric}_abs") if so_row is not None else None
        so_lo = so_row.get(f"{metric}_abs_lo") if so_row is not None else None
        so_hi = so_row.get(f"{metric}_abs_hi") if so_row is not None else None
        band_pts = [v for v in (sb_abs, sb_lo, sb_hi, so_abs, so_lo, so_hi) if v is not None and not pd.isna(v)]
        if not band_pts:
            continue
        band_mid = (min(band_pts) + max(band_pts)) / 2
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, band_mid),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#555",
            ha="left",
            va="center",
        )

    # Two-row legend, col-major fill: row1=[A,B,multi-baseline], row2=[E,F,G]
    ax_handles, ax_lbls = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(ax_lbls, ax_handles))
    A = label_to_handle.get("single agent baseline (no safety prompt)")
    B = label_to_handle.get("single agent baseline (with safety prompt)")
    D = mpatches.Patch(facecolor="white", edgecolor=gold, linewidth=2.8, label="multi-agent baseline (highlighted)")
    E = mpatches.Patch(
        facecolor=c, edgecolor=c, linewidth=1.2, label="sig. less than single agent baseline (no safety prompt)"
    )
    F = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=c,
        linewidth=1.2,
        label="not different from single agent baseline (no safety prompt)",
    )
    G = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=c,
        linewidth=1.2,
        hatch="//",
        label="sig. greater than single agent baseline (no safety prompt)",
    )
    interleaved = [A, E, B, F, D, G]
    handles = [h for h in interleaved if h is not None]
    lbls = [h.get_label() for h in handles]
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=13,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig


def plot_safety_bars_vs_single(
    cells: pd.DataFrame,
    df: pd.DataFrame,
    out_path,
    orgs: list[str] | None = None,
    models: list[str] | None = None,
    title: str | None = None,
    pct: float = 0.10,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> plt.Figure | None:
    """Headline / main-paper figure: 2-row cross-model bars showing curated
    multi-agent interventions' safety_score Δ vs single-agent baseline.

    Row 1 (mean): mean Δ with Welch t CIs.
    Row 2 (worst-10%): Δ of the pct·100-percentile of safety_score with BCa
    bootstrap 95% CIs (B=10000).

    Sig coloring (3-state, both rows, vs single agent baseline (no safety prompt)):
      - solid model color   = sig. less than single agent baseline
      - light fill          = not different from single agent baseline
      - light fill + hatch  = sig. greater than single agent baseline

    Single-agent reference lines + gray "single agent target range" band on
    each row, computed on that row's appropriate scale (Welch SE on row 1,
    BCa pct CI on row 2).
    """
    if orgs is None:
        orgs = ["multi8_base", "multi8_comms", "multi8_dri", "multi8_safetydoc", "multi8_safetydocdri"]
    present = set(cells["model"].unique())
    if models is None:
        ordered = [m for m in MODEL_ORDER if m in present]
        ordered += [m for m in sorted(present) if m not in MODEL_ORDER]
        models = list(reversed(ordered))

    # Headline-specific bar labels (horizontal, multi-line). Keeps global
    # CLEAN_LABEL untouched so family-faceted plots still read "designated".
    bar_label = {
        "multi8_base": "multi-ag.\nbaseline",
        "multi8_comms": "group\nchat",
        "multi8_dri": "safety\nlead",
        "multi8_safetydoc": "safety\ndoc",
        "multi8_safetydocdri": "safety\ndoc + lead",
    }

    rng = np.random.default_rng(seed)

    def _bca_pct(a: np.ndarray, q: float, B: int):
        if len(a) < 2:
            return np.nan, np.nan, np.nan
        point = float(np.percentile(a, q * 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _stats.bootstrap(
                    (a,),
                    lambda aa: np.percentile(aa, q * 100),
                    n_resamples=B,
                    method="BCa",
                    vectorized=False,
                    confidence_level=0.95,
                    random_state=rng,
                )
                return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
            except Exception:
                return point, np.nan, np.nan

    def _bca_delta(a: np.ndarray, b: np.ndarray, q: float, B: int):
        if len(a) < 2 or len(b) < 2:
            return np.nan, np.nan, np.nan
        point = float(np.percentile(a, q * 100) - np.percentile(b, q * 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _stats.bootstrap(
                    (a, b),
                    lambda aa, bb: np.percentile(aa, q * 100) - np.percentile(bb, q * 100),
                    n_resamples=B,
                    method="BCa",
                    vectorized=False,
                    paired=False,
                    confidence_level=0.95,
                    random_state=rng,
                )
                return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
            except Exception:
                return point, np.nan, np.nan

    # Per-(model, org) BCa values for row 2
    worst = {}  # (model, org) -> {"d", "lo", "hi", "sb_band_lo", "sb_band_hi", "so_d", "so_lo", "so_hi"}
    for model in models:
        sub_df = df[df["model"] == model]
        sb_runs = sub_df[sub_df["org_name"] == "single_base"]["safety_score"].dropna().values
        if len(sb_runs) < 2:
            continue
        # single_base's own absolute pct CI mapped to ±err around 0
        sb_pt, sb_lo_a, sb_hi_a = _bca_pct(sb_runs, pct, n_bootstrap)
        sb_band_lo = (sb_lo_a - sb_pt) if not (pd.isna(sb_lo_a) or pd.isna(sb_pt)) else 0.0
        sb_band_hi = (sb_hi_a - sb_pt) if not (pd.isna(sb_hi_a) or pd.isna(sb_pt)) else 0.0
        # single_ownership's BCa Δ vs sb
        so_runs = sub_df[sub_df["org_name"] == "single_ownership"]["safety_score"].dropna().values
        if len(so_runs) >= 2:
            so_d, so_lo, so_hi = _bca_delta(so_runs, sb_runs, pct, n_bootstrap)
        else:
            so_d = so_lo = so_hi = np.nan
        worst[(model, "__band__")] = (sb_band_lo, sb_band_hi, so_d, so_lo, so_hi)
        for org in orgs:
            org_runs = sub_df[sub_df["org_name"] == org]["safety_score"].dropna().values
            d, lo, hi = _bca_delta(org_runs, sb_runs, pct, n_bootstrap)
            worst[(model, org)] = (d, lo, hi)

    light_fill = "#dddddd"
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(4.0 * n_models + 2.0, 9.5), sharey="row", squeeze=False)

    # ── Row 1: mean Δ vs single (Welch CI) ──────────────────────────────
    all_lo_r1, all_hi_r1 = [], []
    for model in models:
        sub = cells[cells["model"] == model].set_index("org_name")
        for org in list(orgs) + ["single_ownership"]:
            if org in sub.index:
                r = sub.loc[org]
                lo = r.get("safety_score_ci_lo_vs_single", r["safety_score_delta_vs_single"])
                hi = r.get("safety_score_ci_hi_vs_single", r["safety_score_delta_vs_single"])
                if not pd.isna(lo):
                    all_lo_r1.append(lo)
                if not pd.isna(hi):
                    all_hi_r1.append(hi)
    y_min_r1 = min(all_lo_r1 + [0]) - 0.05
    y_max_r1 = max(all_hi_r1 + [0]) + 0.05

    for col, model in enumerate(models):
        ax = axes[0, col]
        sub = cells[cells["model"] == model].set_index("org_name")
        c = MODEL_COLOR.get(model, "#555")
        is_first = col == 0

        ax.axhline(
            0,
            color="#444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
            label="single agent baseline (no safety prompt)" if is_first else None,
        )
        sb = sub.loc["single_base"] if "single_base" in sub.index else None
        so = sub.loc["single_ownership"] if "single_ownership" in sub.index else None
        if so is not None:
            so_d = so["safety_score_delta_vs_single"]
            so_lo = so.get("safety_score_ci_lo_vs_single", so_d)
            so_hi = so.get("safety_score_ci_hi_vs_single", so_d)
            sb_band_lo = sb_band_hi = 0.0
            if sb is not None and not pd.isna(sb.get("safety_score_se")):
                t_sb = _stats.t.ppf(0.975, max(int(sb["n"]) - 1, 1))
                sb_band_lo = -t_sb * sb["safety_score_se"]
                sb_band_hi = +t_sb * sb["safety_score_se"]
            band_lo = float(np.nanmin([sb_band_lo, so_d, so_lo]))
            band_hi = float(np.nanmax([sb_band_hi, so_d, so_hi]))
            ax.axhspan(band_lo, band_hi, color="#cccccc", alpha=0.35, zorder=0)
            ax.axhline(
                so_d,
                color="#444",
                linestyle=":",
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
                label="single agent baseline (with safety prompt)" if is_first else None,
            )

        for i, org in enumerate(orgs):
            if org not in sub.index:
                continue
            r = sub.loc[org]
            mu = r["safety_score_delta_vs_single"]
            ci_lo = r.get("safety_score_ci_lo_vs_single", mu)
            ci_hi = r.get("safety_score_ci_hi_vs_single", mu)
            sign = r.get("safety_sign", "inconclusive")
            if sign == "worse":
                face, hatch = c, None
            elif sign == "better":
                face, hatch = light_fill, "//"
            else:
                face, hatch = light_fill, None
            err_lo = mu - ci_lo if not pd.isna(ci_lo) else 0
            err_hi = ci_hi - mu if not pd.isna(ci_hi) else 0
            ax.bar(i, mu, width=0.7, color=face, edgecolor=c, linewidth=1.2, hatch=hatch, zorder=2)
            ax.errorbar(
                i, mu, yerr=[[err_lo], [err_hi]], fmt="none", ecolor="#222", elinewidth=1.0, capsize=2, zorder=3
            )

        ax.set_xticks(range(len(orgs)))
        ax.set_xticklabels([bar_label.get(o, o) for o in orgs], fontsize=12, rotation=0, ha="center")
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(model, fontsize=14, fontweight="bold")
        ax.set_ylim(y_min_r1, y_max_r1)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Δ safety score vs single ag. baseline (pp)", fontsize=11, fontweight="bold")
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:+.0f}"))

    # ── Row 2: worst-10% Δ vs single (BCa CI) ───────────────────────────
    all_lo_r2, all_hi_r2 = [], []
    for model in models:
        for org in orgs:
            d, lo, hi = worst.get((model, org), (np.nan, np.nan, np.nan))
            if not pd.isna(lo):
                all_lo_r2.append(lo)
            if not pd.isna(hi):
                all_hi_r2.append(hi)
        band = worst.get((model, "__band__"))
        if band is not None:
            sb_band_lo, sb_band_hi, so_d, so_lo, so_hi = band
            for v in (sb_band_lo, sb_band_hi, so_d, so_lo, so_hi):
                if v is not None and not pd.isna(v):
                    if v < 0:
                        all_lo_r2.append(v)
                    else:
                        all_hi_r2.append(v)
    y_min_r2 = min(all_lo_r2 + [0]) - 0.05
    y_max_r2 = max(all_hi_r2 + [0]) + 0.05

    for col, model in enumerate(models):
        ax = axes[1, col]
        c = MODEL_COLOR.get(model, "#555")

        band = worst.get((model, "__band__"))
        ax.axhline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
        if band is not None:
            sb_band_lo, sb_band_hi, so_d, so_lo, so_hi = band
            band_pts = [sb_band_lo, sb_band_hi]
            for v in (so_d, so_lo, so_hi):
                if v is not None and not pd.isna(v):
                    band_pts.append(v)
            if band_pts:
                ax.axhspan(min(band_pts), max(band_pts), color="#cccccc", alpha=0.35, zorder=0)
            if so_d is not None and not pd.isna(so_d):
                ax.axhline(so_d, color="#444", linestyle=":", linewidth=1.4, alpha=0.95, zorder=4)

        for i, org in enumerate(orgs):
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
            err_lo = max(0.0, d - lo) if not pd.isna(lo) else 0
            err_hi = max(0.0, hi - d) if not pd.isna(hi) else 0
            ax.bar(i, d, width=0.7, color=face, edgecolor=c, linewidth=1.2, hatch=hatch, zorder=2)
            if (err_lo + err_hi) > 0:
                ax.errorbar(
                    i, d, yerr=[[err_lo], [err_hi]], fmt="none", ecolor="#222", elinewidth=1.0, capsize=2, zorder=3
                )

        ax.set_xticks(range(len(orgs)))
        ax.set_xticklabels([bar_label.get(o, o) for o in orgs], fontsize=12, rotation=0, ha="center")
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(y_min_r2, y_max_r2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    q_pct = int(round(pct * 100))
    axes[1, 0].set_ylabel("Δ safety score vs single ag. baseline (pp)", fontsize=11, fontweight="bold")
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:+.0f}"))

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, right=0.92, hspace=0.45, top=0.88 if title else 0.91)

    # Bold row titles ("Mean Δ safety score" / "10th-percentile Δ safety score")
    fig.canvas.draw()
    pos_top = axes[0, 0].get_position()
    pos_top_right = axes[0, -1].get_position()
    pos_bot = axes[1, 0].get_position()
    pos_bot_right = axes[1, -1].get_position()
    row_x = (pos_top.x0 + pos_top_right.x1) / 2
    fig.text(
        row_x,
        pos_top.y1 + 0.045,
        "Mean Δ safety score",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    fig.text(
        row_x,
        pos_bot.y1 + 0.045,
        f"{int(round(pct * 100))}th-percentile Δ safety score",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Right-side annotation for the gray bands on each row
    for row_idx in (0, 1):
        ax_right = axes[row_idx, -1]
        if row_idx == 0:
            sb = cells[(cells["model"] == models[-1]) & (cells["org_name"] == "single_base")]
            so = cells[(cells["model"] == models[-1]) & (cells["org_name"] == "single_ownership")]
            if sb.empty or so.empty:
                continue
            sb_r = sb.iloc[0]
            so_r = so.iloc[0]
            sb_band_lo = sb_band_hi = 0.0
            if not pd.isna(sb_r.get("safety_score_se")):
                t_sb = _stats.t.ppf(0.975, max(int(sb_r["n"]) - 1, 1))
                sb_band_lo = -t_sb * sb_r["safety_score_se"]
                sb_band_hi = +t_sb * sb_r["safety_score_se"]
            so_d = so_r["safety_score_delta_vs_single"]
            so_lo = so_r.get("safety_score_ci_lo_vs_single", so_d)
            so_hi = so_r.get("safety_score_ci_hi_vs_single", so_d)
            band_pts = [sb_band_lo, sb_band_hi, so_d, so_lo, so_hi]
        else:
            band = worst.get((models[-1], "__band__"))
            if band is None:
                continue
            sb_band_lo, sb_band_hi, so_d, so_lo, so_hi = band
            band_pts = [sb_band_lo, sb_band_hi]
            for v in (so_d, so_lo, so_hi):
                if v is not None and not pd.isna(v):
                    band_pts.append(v)
        if not band_pts:
            continue
        band_mid = (min(band_pts) + max(band_pts)) / 2
        ax_right.annotate(
            "single agent\ntarget range",
            xy=(1.0, band_mid),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#555",
            ha="left",
            va="center",
        )

    # Two-row legend: row1=[A,B], row2=[E,F,G] (col-major fill, ncol=3, blank to balance)
    ax_handles, ax_lbls = axes[0, 0].get_legend_handles_labels()
    label_to_handle = dict(zip(ax_lbls, ax_handles))
    A = label_to_handle.get("single agent baseline (no safety prompt)")
    B = label_to_handle.get("single agent baseline (with safety prompt)")
    pick_color = MODEL_COLOR.get(models[0], "#555")
    E = mpatches.Patch(
        facecolor=pick_color,
        edgecolor=pick_color,
        linewidth=1.2,
        label="sig. < than single ag. default baseline",
    )
    F = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=pick_color,
        linewidth=1.2,
        label="not sig. diff. from single ag. default baseline",
    )
    G = mpatches.Patch(
        facecolor=light_fill,
        edgecolor=pick_color,
        linewidth=1.2,
        hatch="//",
        label="sig. > than single ag. default baseline",
    )
    blank = mpatches.Patch(facecolor="none", edgecolor="none", label=" ")
    interleaved = [A, E, B, F, blank, G]
    handles = [h for h in interleaved if h is not None]
    lbls = [h.get_label() for h in handles]
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=13,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def compute_ex_post_frontier(test_data: pd.DataFrame, K: int = 50) -> pd.DataFrame:
    """Ex-post (oracle) frontier on the held-out test set: for each k misinfo
    articles allowed in the top-K, pick the highest-view articles among each
    class. Returns Pareto-optimal points only (monotone in views).

    k values where K-k > n_clean are silently skipped (you can't fill K slots
    with only n_clean clean articles available); for the default test set
    this means the frontier starts at k=5 (misinfo_rate=0.10), not k=0.
    `compute_ex_ante_frontier` enforces the same lower bound via
    `k_min = max(0, K - n2)`, so the two frontiers agree on range."""
    misinfo = (
        test_data[test_data["misinformation"].astype(bool)].sort_values("views", ascending=False).reset_index(drop=True)
    )
    clean = (
        test_data[~test_data["misinformation"].astype(bool)]
        .sort_values("views", ascending=False)
        .reset_index(drop=True)
    )
    n1, n2 = len(misinfo), len(clean)
    mv = misinfo["views"].values
    cv = clean["views"].values
    rows = []
    best = -1
    for k in range(0, min(K, n1) + 1):
        n_c = K - k
        if n_c > n2:
            continue
        v = (mv[:k].sum() if k else 0) + (cv[:n_c].sum() if n_c else 0)
        if v > best:
            rows.append({"views": float(v), "misinfo_rate": k / K})
            best = v
    return pd.DataFrame(rows)


def compute_ex_ante_frontier(
    test_data: pd.DataFrame, K: int = 50, n_samples: int = 3000, seed: int = 42
) -> pd.DataFrame:
    """Ex-ante (perfect classifier, random within class) frontier — Monte Carlo.
    For each k misinfo articles in the top-K, draws k uniformly at random from
    the misinfo pool and (K-k) from the clean pool, summing views. Returns
    median plus 10/90 quantiles of total views at each misinfo_rate = k/K."""
    rng = np.random.default_rng(seed)
    mv = test_data[test_data["misinformation"].astype(bool)]["views"].values
    cv = test_data[~test_data["misinformation"].astype(bool)]["views"].values
    n1, n2 = len(mv), len(cv)
    k_min = max(0, K - n2)
    k_max = min(K, n1)
    rows = []
    for k in range(k_min, k_max + 1):
        n_c = K - k
        draws = np.empty(n_samples)
        for i in range(n_samples):
            vm = mv[rng.choice(n1, size=k, replace=False)].sum() if k else 0
            vc = cv[rng.choice(n2, size=n_c, replace=False)].sum() if n_c else 0
            draws[i] = vm + vc
        rows.append(
            {
                "misinfo_rate": k / K,
                "views_p10": float(np.percentile(draws, 10)),
                "views_p50": float(np.percentile(draws, 50)),
                "views_p90": float(np.percentile(draws, 90)),
            }
        )
    return pd.DataFrame(rows)


def load_test_baselines(
    test_csv_path, K: int = 50, n_samples: int = 3000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Load test.csv and compute (ex_post, ex_ante) baselines for plot_scatter."""
    p = Path(test_csv_path)
    if not p.exists():
        return None, None
    test_data = pd.read_csv(p)
    ex_post = compute_ex_post_frontier(test_data, K=K)
    ex_ante = compute_ex_ante_frontier(test_data, K=K, n_samples=n_samples, seed=seed)
    return ex_post, ex_ante


def plot_scatter(
    df: pd.DataFrame,
    model: str,
    org_name: str,
    out_path,
    ex_post: pd.DataFrame | None = None,
    ex_ante: pd.DataFrame | None = None,
) -> plt.Figure | None:
    """Per-(model, org) scatter: safety_score (1−misinfo@50) vs views@50, with
    single_base overlay, Pareto frontier per group, and optional ex-post (oracle)
    and ex-ante (random-within-class) frontiers. Upper-right = better."""
    sub_s = df[(df["model"] == model) & (df["org_name"] == "single_base")]
    sub_o = df[(df["model"] == model) & (df["org_name"] == org_name)]
    if sub_s.empty or sub_o.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    def pareto(xs, ys):
        # Upper-right is best: walk points by x descending, keep monotone-up in y.
        pts = sorted(zip(xs, ys), key=lambda p: -p[0])
        front = []
        best_y = -float("inf")
        for x, y in pts:
            if y > best_y:
                front.append((x, y))
                best_y = y
        return sorted(front)

    if ex_ante is not None and not ex_ante.empty:
        y = 1 - ex_ante["misinfo_rate"]
        ax.fill_betweenx(
            y, ex_ante["views_p10"], ex_ante["views_p90"], color="#999", alpha=0.18, zorder=1.5, label="ex-ante p10–p90"
        )
        ax.plot(ex_ante["views_p50"], y, color="#333", linewidth=1.4, alpha=0.9, zorder=2, label="ex-ante median")
    if ex_post is not None and not ex_post.empty:
        ax.plot(
            ex_post["views"],
            1 - ex_post["misinfo_rate"],
            color="#2ca02c",
            linewidth=2,
            alpha=0.9,
            zorder=2,
            label="ex-post (oracle)",
        )
        ax.scatter(ex_post["views"], 1 - ex_post["misinfo_rate"], color="#2ca02c", s=18, zorder=2.5)

    sx = sub_s["views_at_50"].values
    sy = 1 - sub_s["misinfo_at_50"].values
    ox = sub_o["views_at_50"].values
    oy = 1 - sub_o["misinfo_at_50"].values

    ax.scatter(
        sx,
        sy,
        s=50,
        color="#888",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label=f"single_base (n={len(sub_s)})",
    )
    ax.scatter(
        ox,
        oy,
        s=60,
        color="#d62728",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        label=f"{org_name} (n={len(sub_o)})",
    )

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

    ax.set_xlabel("Views@50")
    ax.set_ylabel("Safety score (1 − misinfo@50)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model} — {org_name}", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", frameon=True, fontsize=8)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return fig


def plot_cost_benefit(df: pd.DataFrame, cells: pd.DataFrame, out_path) -> plt.Figure | None:
    """Faceted cost-benefit scatter: one panel per model.

    Per panel:
      X = mean total_cost / that model's multi8_base mean total_cost (linear scale).
      Y = Δ safety_score vs that model's multi8_base, in pp (Welch CI on Y bars).

    Markers (color = model, filled = sig. ≠ multi8_base, hollow = CI crosses 0):
      ● circle = multi-agent intervention; 2-char code rendered inside.
      ✚ filled plus = single-agent run; code rendered offset-right.
      ★ star = compound intervention (only safety_doc + designated).

    Per-model Pareto frontier drawn as a model-color step line. Cross-hair at
    (1.0×, 0pp) marks the per-model baseline.
    X is per-panel (sharex=False) so each panel uses its full width; Y is shared
    so Δ pp is comparable across panels. Last subplot slot holds the legend.
    """
    cost_mean = df.groupby(["model", "org_name"])["total_cost"].mean().rename("total_cost_mean").reset_index()
    base_cost = cost_mean[cost_mean["org_name"] == "multi8_base"].set_index("model")["total_cost_mean"]
    pts = cells.merge(cost_mean, on=["model", "org_name"])
    pts = pts[pts["model"].isin(base_cost.index)].copy()
    pts["frac_cost"] = pts["total_cost_mean"] / pts["model"].map(base_cost)
    pts["delta_pp"] = pts["safety_score_delta_vs_multi"] * 100
    pts["ci_lo_pp"] = pts["safety_score_ci_lo_vs_multi"] * 100
    pts["ci_hi_pp"] = pts["safety_score_ci_hi_vs_multi"] * 100
    pts["disp_fam"] = pts.apply(_disp_fam, axis=1)
    pts = pts[pts["org_name"] != "multi8_base"].dropna(subset=["frac_cost", "delta_pp"])
    if pts.empty:
        return None

    present_models = [m for m in MODEL_ORDER if m in pts["model"].unique()]
    present_models += [m for m in sorted(pts["model"].unique()) if m not in MODEL_ORDER]
    present_models = list(reversed(present_models))

    n_cols = 3
    n_rows = (len(present_models) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.4 * n_cols, 4.2 * n_rows), sharex=False, sharey=False, squeeze=False
    )
    axes_flat = axes.flatten()

    def _pareto(sub):
        s = sub.sort_values("frac_cost")
        front, best = [], -np.inf
        for _, r in s.iterrows():
            if r["delta_pp"] > best:
                front.append((r["frac_cost"], r["delta_pp"], r["org_name"]))
                best = r["delta_pp"]
        return front

    def _marker_for(org_name, is_combo):
        if org_name.startswith("single"):
            return "P"  # filled plus
        if is_combo:
            return "*"  # star
        return "o"  # circle

    for i, m in enumerate(present_models):
        ax = axes_flat[i]
        sub = pts[pts["model"] == m]
        c = MODEL_COLOR.get(m, "#555")

        ax.axhline(0, color="#8b5a3c", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1.5)
        ax.axvline(1.0, color="#8b5a3c", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1.5)

        for _, r in sub.iterrows():
            x, y = r["frac_cost"], r["delta_pp"]
            org = r["org_name"]
            marker = _marker_for(org, bool(r.get("is_combo")))
            code = ORG_CODE.get(org, "")
            if not (pd.isna(r["ci_lo_pp"]) or pd.isna(r["ci_hi_pp"])):
                sig = not (r["ci_lo_pp"] <= 0 <= r["ci_hi_pp"])
                ax.vlines(x, r["ci_lo_pp"], r["ci_hi_pp"], color=c, linewidth=1.0, alpha=0.35, zorder=2)
            else:
                sig = False
            face = c if sig else "white"

            if marker == "o":
                ax.scatter(x, y, s=380, marker="o", facecolor=face, edgecolor=c, linewidth=1.5, zorder=4)
                if code:
                    text_color = "white" if sig else c
                    ax.text(
                        x,
                        y,
                        code,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        fontweight="bold",
                        color=text_color,
                        zorder=5,
                    )
            elif marker == "P":
                ax.scatter(x, y, s=180, marker="P", facecolor=face, edgecolor=c, linewidth=1.5, zorder=4)
                if code:
                    ax.annotate(
                        code,
                        xy=(x, y),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold",
                        color=c,
                        va="center",
                        zorder=5,
                    )
            else:  # star
                ax.scatter(
                    x,
                    y,
                    s=440,
                    marker="*",
                    facecolor=face,
                    edgecolor=c,
                    linewidth=2.0,
                    zorder=4,
                    path_effects=[mpe.withStroke(linewidth=3.5, foreground="#222")],
                )

        front = _pareto(sub[~sub["org_name"].str.startswith("single")])
        if len(front) >= 2:
            xs, ys = zip(*[(f[0], f[1]) for f in front])
            ax.plot(xs, ys, color=c, linewidth=1.6, alpha=0.65, linestyle="-", zorder=2.5)

        ax.set_title(m, fontweight="bold", fontsize=11)
        ax.grid(True, which="major", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlabel("Cost / multi-agent baseline (× baseline)")
        ax.set_ylabel("Δ safety score vs baseline (pp)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}"))

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

        def _row(
            y, marker, label, *, marker_size=180, facecolor=gray, edgecolor=gray, linewidth=1.5, path_effects=None
        ):
            leg_ax.scatter(
                marker_x,
                y,
                s=marker_size,
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
                path_effects=path_effects,
            )
            leg_ax.text(text_x, y, label, fontsize=9.5, va="center", ha="left")

        leg_ax.text(0.02, 0.98, "Marker", fontweight="bold", fontsize=11, va="top", ha="left")
        _row(0.86, "o", "multi-agent intervention", marker_size=180)
        _row(0.79, "P", "single agent", marker_size=150)
        _row(
            0.72,
            "*",
            "compound (safety doc + designated)",
            marker_size=220,
            linewidth=1.8,
            path_effects=[mpe.withStroke(linewidth=3.0, foreground="#222")],
        )
        leg_ax.text(
            text_x,
            0.65,
            "filled = sig. ≠ baseline ·  hollow = CI crosses 0",
            fontsize=8.5,
            fontstyle="italic",
            color="#444",
            va="center",
            ha="left",
        )

        leg_ax.plot([marker_x - 0.05, marker_x + 0.05], [0.58, 0.58], color=gray, linewidth=1.8, alpha=0.7, zorder=3)
        leg_ax.text(text_x, 0.58, "per-model Pareto frontier (multi-agent only)", fontsize=9.5, va="center", ha="left")
        leg_ax.plot(
            [marker_x - 0.05, marker_x + 0.05], [0.51, 0.51], color="#8b5a3c", linewidth=1.0, linestyle="--", zorder=3
        )
        leg_ax.text(text_x, 0.51, "multi-agent baseline (1×, 0pp)", fontsize=9.5, va="center", ha="left")

        leg_ax.text(0.02, 0.42, "Code → intervention", fontweight="bold", fontsize=11, va="top", ha="left")
        code_lines = [
            ("Sb", "single agent — baseline"),
            ("Ss", "single agent + safety prompt"),
            ("Gc", "group chat (comms)"),
            ("Dr", "designated responsibility"),
            ("O1/O2/O3", "distributed resp. (low / med / heavy)"),
            ("C1/C2/C3", "centralized resp. (low / med / heavy)"),
            ("Sd", "safety doc"),
            ("Ri/Rp/Rm", "review (indep / peer / PM)"),
            ("Si/Sp/Sm", "safety review (indep / peer / PM)"),
        ]
        line_y0 = 0.32
        line_dy = 0.038
        for k, (code, name) in enumerate(code_lines):
            y = line_y0 - k * line_dy
            leg_ax.text(0.04, y, code, fontsize=9, family="monospace", fontweight="bold", va="center", ha="left")
            leg_ax.text(0.34, y, name, fontsize=9, va="center", ha="left")

    fig.suptitle(
        "Rec. system: cost-benefit of interventions vs multi-agent baseline, by model",
        fontweight="bold",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    return fig
