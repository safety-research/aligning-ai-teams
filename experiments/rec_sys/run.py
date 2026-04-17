#!/usr/bin/env python3
"""
Run rec_sys experiments using magelab pipeline.

Usage:
    uv run python run.py --config configs/base_8.yaml
    uv run python run.py --config configs/base_8.yaml --runs 10 --max-concurrent 2
"""

import argparse
import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from evaluate import run_recommendation_and_evaluate
from helpers import read_config_metadata, read_run_meta

from magelab import OrgConfig, run_in_workspace, run_pipeline_batch
from magelab.auth import resolve_api_key, resolve_sub

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
LLM_CLIENT = EXPERIMENT_DIR / "llm_client.py"
REQUIREMENTS = EXPERIMENT_DIR / "requirements.txt"
TRAIN_DATASET = "train.csv"
TEST_DATASET = "test.csv"
EVAL_CONFIG = "eval_config.json"


async def setup(output_dir: Path, logger: logging.Logger, org_config: OrgConfig) -> None:
    """Populate workspace with training data, shared utilities, and install deps."""
    ws = output_dir / "workspace"
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / TRAIN_DATASET, ws / "data" / TRAIN_DATASET)
    shutil.copy(LLM_CLIENT, ws / "src" / "llm_client.py")
    shutil.copy(REQUIREMENTS, ws / "requirements.txt")

    result = await run_in_workspace(
        ["pip", "install", "-r", "requirements.txt"],
        output_dir=output_dir,
        timeout=120,
        logger=logger,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install requirements: {result.stderr}")


async def evaluate(output_dir: Path, logger: logging.Logger, org_config: OrgConfig) -> None:
    """Run recommendations on test data, compute metrics, and write results."""
    test_path = DATA_DIR / TEST_DATASET
    eval_config_path = DATA_DIR / EVAL_CONFIG
    task_evaluation = None
    if test_path.exists() and eval_config_path.exists():
        task_evaluation = await run_recommendation_and_evaluate(
            output_dir, str(test_path), str(eval_config_path), logger=logger, quiet=True,
        )
    results = {
        "experiment_name": output_dir.name,
        "task_evaluation": task_evaluation,
        "run_meta": read_run_meta(output_dir),
    }
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def make_experiment_name(config_path: str, timestamp: datetime | None = None) -> str:
    """Generate experiment name: results/{model}/{N}_agent_experiments/{name}/{timestamp}"""
    config_name, model, num_agents = read_config_metadata(Path(config_path))
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"results/{model}/{num_agents}_agent_experiments/{config_name}/{ts}"


def main():
    parser = argparse.ArgumentParser(
        description="Run rec_sys experiments using magelab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_DIR / "configs" / "base_8.yaml"),
        help="Path to YAML config (default: configs/base_8.yaml)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of experiments to run (default: 1)")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Max concurrent experiments (default: 1)")
    parser.add_argument("--no-frontend", action="store_true", help="Disable the frontend dashboard")
    parser.add_argument(
        "--frontend-port", type=int, default=8765, help="Port for the frontend dashboard (default: 8765)"
    )
    parser.add_argument(
        "--docker",
        action="store_const",
        const="run",
        default=None,
        help="Run org phases in Docker containers",
    )
    parser.add_argument(
        "--docker-build",
        action="store_const",
        const="build",
        dest="docker",
        help="Force rebuild Docker image, then run in Docker",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model for all roles (e.g. claude-opus-4-6)",
    )
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--api-key", nargs="?", const=True, default=None, metavar="ENV_FILE",
                            help="Use API key auth (optionally from a .env file)")
    auth_group.add_argument("--sub", nargs="?", const=True, default=None, metavar="CREDS_FILE",
                            help="Use subscription/OAuth auth (optionally from a credentials file)")

    args = parser.parse_args()

    if args.api_key is not None:
        env_file = Path(args.api_key) if isinstance(args.api_key, str) else None
        auth = resolve_api_key(env_file)
    else:
        creds_file = Path(args.sub) if isinstance(args.sub, str) else None
        auth = resolve_sub(creds_file)

    # If --model is set, rewrite the config with the overridden model and use that.
    config_path = args.config
    model_tmp = None
    if args.model:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for role in cfg.get("roles", {}).values():
            role["model"] = args.model
        model_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="magelab_", delete=False,
        )
        yaml.dump(cfg, model_tmp, default_flow_style=False, sort_keys=False)
        model_tmp.flush()
        config_path = model_tmp.name

    fe_port = None if args.no_frontend else args.frontend_port
    now = datetime.now()
    stages = [setup, evaluate]
    output_dirs = [
        EXPERIMENT_DIR / make_experiment_name(config_path, now + timedelta(seconds=i))
        for i in range(args.runs)
    ]
    try:
        asyncio.run(run_pipeline_batch(
            config_path=config_path,
            stages=stages,
            output_dirs=output_dirs,
            max_concurrent=args.max_concurrent,
            base_frontend_port=fe_port,
            abort_on=frozenset(),
            docker=args.docker,
            auth=auth,
        ))
    finally:
        if model_tmp:
            Path(model_tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
