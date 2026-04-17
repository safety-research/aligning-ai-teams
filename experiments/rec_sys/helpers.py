"""
Shared helpers for experiment scripts.

These are experiment-side utilities, not part of the magelab package.
"""

import sqlite3
from pathlib import Path

import yaml


def read_config_metadata(config_path: Path) -> tuple[str, str, int]:
    """Read lightweight metadata from a YAML config.

    Returns:
        (config_name, model, num_agents) tuple.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    name = cfg.get("settings", {}).get("org_name", config_path.stem)
    roles = cfg.get("roles", {})
    models = {r.get("model", "unknown") for r in roles.values()}
    model = models.pop() if len(models) == 1 else ("unknown" if not models else "mixed")
    num_agents = len(cfg.get("agents", {}))
    return name, model, num_agents


def read_run_meta(output_dir: Path) -> dict:
    """Read lightweight run metadata from the run's SQLite database.

    Returns flat run_meta fields (outcome, duration, cost, task counts,
    API errors). No agent summaries or dispatch details.

    Returns an empty dict if no DB is found.
    """
    db_files = list(output_dir.glob("*.db"))
    if not db_files:
        return {}
    db_path = db_files[0]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        meta = conn.execute("SELECT * FROM run_meta ORDER BY id DESC LIMIT 1").fetchone()
        if not meta:
            return {}
        return {
            "outcome": meta["outcome"],
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
            "duration_seconds": meta["duration_seconds"],
            "total_cost_usd": meta["total_cost_usd"],
            "timed_out": bool(meta["timed_out"]),
            "sync_rounds": meta["sync_rounds"],
            "tasks_succeeded": meta["tasks_succeeded"],
            "tasks_failed": meta["tasks_failed"],
            "tasks_open": meta["tasks_open"],
            "rate_limited_429": meta["rate_limited_429"],
            "api_overloaded_529": meta["api_overloaded_529"],
            "api_error_other": meta["api_error_other"],
        }
    finally:
        conn.close()
