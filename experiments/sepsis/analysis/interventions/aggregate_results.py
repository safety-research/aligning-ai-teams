#!/usr/bin/env python3
"""
Walk the sepsis results directory tree and produce a single CSV
with one row per experiment run.

Reads run stats directly from the SQLite database and eval metrics
from experiment_results.json.

Directory convention:
  results/<model>/<n>_agent_experiments/<org_name>/<run_id>/
    ├── configs/000_start.yaml
    ├── *.db
    └── experiment_results.json

Usage:
    python build_results_csv.py                         # all results
    python build_results_csv.py results/claude-opus-4-1-20250805/9_agent_experiments
    python build_results_csv.py results/claude-opus-4-1-20250805/9_agent_experiments/base_8
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path

import yaml

RESULTS_ROOT = Path(__file__).parent / ".." / ".." / "results"

COLUMNS = [
    "model",
    "num_coders",
    "org_name",
    "run_id",
    "sensitivity",
    "precision",
    "specificity",
    "avg_cost",
    "outcome",
    "duration_seconds",
    "total_cost",
    "tasks_succeeded",
    "tasks_failed",
    "tasks_open",
    "rate_limited_429",
    "api_overloaded_529",
    "api_error_other",
    "max_reviews",
    "max_assignments",
    "total_reviews",
    "total_assignments",
    "timeout_rate",
    "error_rate",
    "avg_review_rounds_per_task",
    "coders_assigned",
]


def _read_db(db_path: Path) -> dict:
    """Read run_meta, agent stats, and task stats from the SQLite database."""
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
            "tasks_succeeded": meta["tasks_succeeded"],
            "tasks_failed": meta["tasks_failed"],
            "tasks_open": meta["tasks_open"],
            "rate_limited_429": meta["rate_limited_429"],
            "api_overloaded_529": meta["api_overloaded_529"],
            "api_error_other": meta["api_error_other"],
        }

        # --- agent summaries from events ---
        agent_rows = conn.execute("""
            SELECT
                target_agent_id,
                SUM(CASE WHEN event_type = 'ReviewRequestedEvent' THEN 1 ELSE 0 END) AS num_reviews,
                SUM(CASE WHEN event_type = 'TaskAssignedEvent' THEN 1 ELSE 0 END) AS num_assignments
            FROM run_events
            WHERE outcome = 'completed'
            GROUP BY target_agent_id
        """).fetchall()
        all_reviews = [row["num_reviews"] or 0 for row in agent_rows]
        all_assignments = [row["num_assignments"] or 0 for row in agent_rows]
        result["max_reviews"] = max(all_reviews) if all_reviews else 0
        result["max_assignments"] = max(all_assignments) if all_assignments else 0
        result["total_reviews"] = sum(all_reviews)
        result["total_assignments"] = sum(all_assignments)

        # --- dispatch-level timeout and error rates from events ---
        dispatch_stats = conn.execute("""
            SELECT
                COUNT(*) AS n_dispatches,
                SUM(CASE WHEN timed_out THEN 1 ELSE 0 END) AS n_timed_out,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS n_errored
            FROM run_events
            WHERE outcome = 'completed'
        """).fetchone()
        n_dispatches = dispatch_stats["n_dispatches"] or 0
        result["timeout_rate"] = (dispatch_stats["n_timed_out"] or 0) / n_dispatches if n_dispatches else None
        result["error_rate"] = (dispatch_stats["n_errored"] or 0) / n_dispatches if n_dispatches else None

        # --- review rounds and coders_assigned from tasks ---
        task_rows = conn.execute("""
            SELECT review_required, assignment_history, review_history
            FROM task_items
        """).fetchall()
        reviewed_tasks = [r for r in task_rows if r["review_required"]]
        if reviewed_tasks:
            total_rounds = sum(len(json.loads(r["review_history"] or "[]")) for r in reviewed_tasks)
            result["avg_review_rounds_per_task"] = total_rounds / len(reviewed_tasks)
        else:
            result["avg_review_rounds_per_task"] = None

        assigned_coders = set()
        for row in task_rows:
            for agent_id in json.loads(row["assignment_history"] or "[]"):
                if agent_id.startswith("coder_"):
                    assigned_coders.add(agent_id)
        result["coders_assigned"] = len(assigned_coders)

        return result
    finally:
        conn.close()


def parse_run(run_dir: Path) -> dict | None:
    """Extract one CSV row from a single run directory."""
    db_files = list(run_dir.glob("*.db"))
    config_file = run_dir / "configs" / "000_start.yaml"
    results_file = run_dir / "experiment_results.json"

    if not db_files:
        return None

    # --- DB stats ---
    try:
        db = _read_db(db_files[0])
    except sqlite3.OperationalError:
        return None
    if not db:
        return None

    # --- config snapshot ---
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

    # Identity — derive from path segments
    # .../results/<model>/<n>_agent_experiments/<org_name>/<run_id>
    parts = run_dir.relative_to(RESULTS_ROOT).parts
    model = parts[0] if len(parts) >= 1 else ""
    org_name = config.get("settings", {}).get("org_name", parts[2] if len(parts) >= 3 else "")
    run_id = run_dir.name

    # num_coders from config agents list (exclude pm roles)
    num_coders = 0
    if config.get("agents"):
        for agent_def in config["agents"].values():
            role_name = agent_def.get("role", "")
            if role_name != "pm":
                num_coders += 1
    elif len(parts) >= 2:
        try:
            num_coders = int(parts[1].split("_")[0]) - 1
        except (ValueError, IndexError):
            pass

    coders_assigned = db.get("coders_assigned", 0)

    # --- eval metrics from experiment_results.json ---
    task_eval = {}
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        task_eval = data.get("task_evaluation") or {}

    sepsis = task_eval.get("sepsis_metrics") or {}
    cost_metrics = task_eval.get("cost_metrics") or {}

    return {
        "model": model,
        "num_coders": num_coders,
        "org_name": org_name,
        "run_id": run_id,
        "sensitivity": sepsis.get("sensitivity"),
        "precision": sepsis.get("precision"),
        "specificity": sepsis.get("specificity"),
        "avg_cost": cost_metrics.get("average_cost_per_patient"),
        "outcome": db.get("outcome"),
        "duration_seconds": db.get("duration_seconds"),
        "total_cost": db.get("total_cost_usd"),
        "tasks_succeeded": db.get("tasks_succeeded"),
        "tasks_failed": db.get("tasks_failed"),
        "tasks_open": db.get("tasks_open"),
        "rate_limited_429": db.get("rate_limited_429", 0),
        "api_overloaded_529": db.get("api_overloaded_529", 0),
        "api_error_other": db.get("api_error_other", 0),
        "max_reviews": db.get("max_reviews", 0),
        "max_assignments": db.get("max_assignments", 0),
        "total_reviews": db.get("total_reviews", 0),
        "total_assignments": db.get("total_assignments", 0),
        "timeout_rate": db.get("timeout_rate"),
        "error_rate": db.get("error_rate"),
        "avg_review_rounds_per_task": db.get("avg_review_rounds_per_task"),
        "coders_assigned": coders_assigned,
    }


def find_run_dirs(search_root: Path) -> list[Path]:
    """Find all directories containing a .db file under search_root."""
    return sorted({p.parent for p in search_root.rglob("*.db") if "workspace" not in p.relative_to(search_root).parts})


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
        row = parse_run(rd)
        if row:
            rows.append(row)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [RESULTS_ROOT]
    # resolve relative paths
    paths = [p if p.is_absolute() else Path.cwd() / p for p in paths]

    output = Path(__file__).parent / "all_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    n = build_csv(paths, output)
    print(f"Wrote {n} rows to {output}")
