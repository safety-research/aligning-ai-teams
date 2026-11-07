#!/usr/bin/env python3
"""
Walk the consultancy results directory tree and produce a single CSV
with one row per (run × framework × eval_sample).

Each run produces N_SAMPLES × N_FRAMEWORKS rows (e.g., 5 × 2 = 10 rows).
Run-level fields are repeated across all rows for that run.

Reads run stats from the SQLite database, eval scores from
experiment_results.json, and config metadata from the resolved YAML.

Directory convention:
  results/{scenario}/{model}/{N}_agent_experiments/{config_stem}/{run_id}/
    ├── *.yaml          (resolved config)
    ├── *.db            (SQLite database)
    ├── proposal.txt    (extracted proposal)
    └── experiment_results.json

Usage:
    python aggregate_results.py                         # all results
    python aggregate_results.py results/loan_profit     # one scenario
    python aggregate_results.py results/loan_profit/claude-opus-4-1-20250805/11_agent_experiments/benign
"""

import csv
import gzip
import json
import sqlite3
import sys
from pathlib import Path

import yaml

RESULTS_ROOT = Path(__file__).parent / ".." / ".." / "results"

COLUMNS = [
    # --- identity ---
    "scenario",
    "model",
    "num_agents",
    "config_name",
    "run_id",
    # --- evaluation (per-sample) ---
    "eval_model",  # judge model name
    "framework",
    "sample",
    "score",
    "status",
    "eval_version",  # proposal version number (int)
    "is_final",  # True if this is the highest (final) version for this run
    # --- proposal ---
    "proposal_version",
    "proposal_length",
    "proposal_outcome",
    # --- run meta (from DB) ---
    "outcome",
    "duration_seconds",
    "total_cost_usd",
    "sync_rounds",
    "tasks_succeeded",
    "tasks_failed",
    "tasks_open",
    "rate_limited_429",
    "api_overloaded_529",
    "api_error_other",
    # --- agent messaging (from DB events) ---
    "total_dispatches",
    "total_messages_sent",
    "timeout_rate",
    "error_rate",
]


def _read_db(db_path: Path) -> dict:
    """Read run_meta and messaging stats from the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # --- run_meta ---
        meta = conn.execute("SELECT * FROM run_meta ORDER BY id DESC LIMIT 1").fetchone()
        if not meta:
            return {}

        result = {
            "outcome": meta["outcome"],
            "duration_seconds": meta["duration_seconds"],
            "total_cost_usd": meta["total_cost_usd"],
            "sync_rounds": meta["sync_rounds"],
            "tasks_succeeded": meta["tasks_succeeded"],
            "tasks_failed": meta["tasks_failed"],
            "tasks_open": meta["tasks_open"],
            "rate_limited_429": meta["rate_limited_429"],
            "api_overloaded_529": meta["api_overloaded_529"],
            "api_error_other": meta["api_error_other"],
        }

        # --- dispatch stats from events ---
        dispatch_stats = conn.execute("""
            SELECT
                COUNT(*) AS n_dispatches,
                SUM(CASE WHEN timed_out THEN 1 ELSE 0 END) AS n_timed_out,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS n_errored
            FROM run_events
            WHERE outcome = 'completed'
        """).fetchone()
        n_dispatches = dispatch_stats["n_dispatches"] or 0
        result["total_dispatches"] = n_dispatches
        result["timeout_rate"] = (dispatch_stats["n_timed_out"] or 0) / n_dispatches if n_dispatches else None
        result["error_rate"] = (dispatch_stats["n_errored"] or 0) / n_dispatches if n_dispatches else None

        # --- wire message counts ---
        msg_count = conn.execute("SELECT COUNT(*) AS n FROM wire_messages").fetchone()
        result["total_messages_sent"] = msg_count["n"] if msg_count else 0

        return result
    finally:
        conn.close()


def parse_run(run_dir: Path) -> list[dict]:
    """Extract CSV rows from a single run directory.

    Returns one row per (framework × eval_sample), or a single row with
    empty eval fields if no evaluation results exist.
    """
    db_files = list(run_dir.glob("*.db"))
    results_file = run_dir / "experiment_results.json"

    if not db_files:
        return []

    # --- DB stats ---
    try:
        db = _read_db(db_files[0])
    except sqlite3.OperationalError:
        return []
    if not db:
        return []

    # --- config snapshot ---
    config_file = run_dir / "configs" / "000_start.yaml"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

    # --- identity from path ---
    # results/{scenario}/{model}/{N}_agent_experiments/{config_stem}/{run_id}
    try:
        rel = run_dir.relative_to(RESULTS_ROOT)
        parts = rel.parts
        scenario = parts[0] if len(parts) >= 1 else ""
        model = parts[1] if len(parts) >= 2 else ""
        config_name = parts[3] if len(parts) >= 4 else config.get("settings", {}).get("org_name", "")
        run_id = run_dir.name
    except ValueError:
        scenario = ""
        model = ""
        config_name = config.get("settings", {}).get("org_name", "")
        run_id = run_dir.name

    num_agents = len(config.get("agents", {}))

    # --- experiment_results.json ---
    eval_data = {}
    if results_file.exists():
        with open(results_file) as f:
            eval_data = json.load(f)

    task_eval = eval_data.get("task_evaluation") or {}
    proposal_outcome = task_eval.get("outcome", "")
    proposal_version = task_eval.get("proposal_version")
    proposal_length = task_eval.get("proposal_length", 0)

    # --- shared fields for every row from this run ---
    base = {
        "scenario": scenario,
        "model": model,
        "num_agents": num_agents,
        "config_name": config_name,
        "run_id": run_id,
        "proposal_version": proposal_version,
        "proposal_length": proposal_length,
        "proposal_outcome": proposal_outcome,
        "outcome": db.get("outcome"),
        "duration_seconds": db.get("duration_seconds"),
        "total_cost_usd": db.get("total_cost_usd"),
        "sync_rounds": db.get("sync_rounds"),
        "tasks_succeeded": db.get("tasks_succeeded"),
        "tasks_failed": db.get("tasks_failed"),
        "tasks_open": db.get("tasks_open"),
        "rate_limited_429": db.get("rate_limited_429", 0),
        "api_overloaded_529": db.get("api_overloaded_529", 0),
        "api_error_other": db.get("api_error_other", 0),
        "total_dispatches": db.get("total_dispatches", 0),
        "total_messages_sent": db.get("total_messages_sent", 0),
        "timeout_rate": db.get("timeout_rate"),
        "error_rate": db.get("error_rate"),
    }

    # --- explode into one row per (eval_model × version × framework × sample) ---
    rows = []
    version_evaluations = task_eval.get("version_evaluations") or {}

    if not version_evaluations:
        # No evaluation — emit one row so we still capture run metadata
        row = dict(base)
        row["eval_model"] = None
        row["framework"] = None
        row["sample"] = None
        row["score"] = None
        row["status"] = None
        row["eval_version"] = None
        row["is_final"] = None
        rows.append(row)
        return rows

    # Detect structure: new format is {model: {ver: {fw: ...}}}, old is {ver: {fw: ...}}
    first_key = next(iter(version_evaluations))
    first_val = version_evaluations[first_key]
    if isinstance(first_val, dict) and first_val and not any(k in first_val for k in ("business_goals", "constitution")):
        # New format: eval_model is top-level key
        model_evals = version_evaluations
    else:
        # Old format: no model key, wrap in a single "unknown" model
        model_evals = {"unknown": version_evaluations}

    for eval_model, model_ver_evals in model_evals.items():
        final_ver = max(int(v) for v in model_ver_evals)

        for ver, ver_eval in model_ver_evals.items():
            ver_int = int(ver)
            for fw, fw_data in ver_eval.items():
                samples = fw_data.get("samples", [])
                if samples:
                    for i, s in enumerate(samples):
                        row = dict(base)
                        row["eval_model"] = eval_model
                        row["framework"] = fw
                        row["sample"] = i
                        row["score"] = s.get("score")
                        row["status"] = s.get("status")
                        row["eval_version"] = ver_int
                        row["is_final"] = ver_int == final_ver
                        rows.append(row)
                else:
                    row = dict(base)
                    row["eval_model"] = eval_model
                    row["framework"] = fw
                    row["sample"] = 0
                    row["score"] = fw_data.get("score")
                    row["status"] = fw_data.get("status")
                    row["eval_version"] = ver_int
                    row["is_final"] = ver_int == final_ver
                    rows.append(row)

    return rows


def find_run_dirs(search_root: Path) -> list[Path]:
    """Find all directories containing a .db file under search_root."""
    return sorted(
        {p.parent for p in search_root.rglob("*.db") if "workspace" not in p.relative_to(search_root).parts}
    )


def build_csv(search_paths: list[Path], output: Path) -> int:
    """Build the CSV from one or more search roots. Returns row count."""
    run_dirs: list[Path] = []
    for sp in search_paths:
        run_dirs.extend(find_run_dirs(sp))

    # deduplicate while preserving order
    seen = set()
    unique = []
    for d in run_dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    run_dirs = unique

    rows = []
    for rd in run_dirs:
        rows.extend(parse_run(rd))

    with gzip.open(output, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [RESULTS_ROOT]
    paths = [p if p.is_absolute() else Path.cwd() / p for p in paths]

    output = Path(__file__).parent / "all_results.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    n = build_csv(paths, output)
    print(f"Wrote {n} rows to {output}")
