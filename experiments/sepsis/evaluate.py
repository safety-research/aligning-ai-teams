#!/usr/bin/env python3
"""
Evaluation functions for sepsis experiment.
Copied from agent_org/experiments/sepsis_icu_v2/evaluate.py for reproducibility.
"""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from metrics import compute_all_metrics, display_metrics

from magelab import run_in_workspace

_default_logger = logging.getLogger(__name__)


async def run_prediction_and_evaluate(
    output_dir: Path,
    test_dataset_path: str,
    eval_config_path: str,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run predict.py on test dataset and calculate sepsis treatment metrics.

    Args:
        output_dir: Path to experiment directory containing workspace
        test_dataset_path: Path to test dataset (JSON file with patient data)
        eval_config_path: Path to evaluation config JSON file

    Returns:
        Dictionary containing sepsis treatment metrics, or None if predict.py doesn't exist
    """
    logger = logger or _default_logger

    try:
        workspace_dir = output_dir / "workspace"
        predict_file = workspace_dir / "src" / "predict.py"

        # Check if predict.py exists
        if not predict_file.exists():
            logger.info("predict.py not found, skipping prediction evaluation")
            return None

        logger.info("Found predict.py for sepsis treatment policy")

        # Load evaluation config (required)
        if not eval_config_path or not Path(eval_config_path).exists():
            logger.error(f"Evaluation config not found: {eval_config_path}")
            raise ValueError(f"Evaluation config file is required but not found: {eval_config_path}")

        with open(eval_config_path, "r") as f:
            eval_config = json.load(f)
        logger.info(f"Loaded evaluation config from {eval_config_path}")

        # Copy test dataset to workspace directory, stripping ground truth labels
        # to prevent label leakage (agents must not see sepsis_label)
        test_file_dest = workspace_dir / "data" / "test.json"
        test_file_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(test_dataset_path, "r") as f:
            test_records = json.load(f)
        for record in test_records:
            record.pop("sepsis_label", None)
            record.pop("SepsisLabel", None)
        with open(test_file_dest, "w") as f:
            json.dump(test_records, f, indent=2)
        logger.info(f"Copied test dataset to {test_file_dest} (ground truth stripped)")

        # Run prediction script on test dataset
        logger.info("Running predict.py on test dataset...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace_dir)
        result = await run_in_workspace(
            ["python", "src/predict.py", "data/test.json"],
            output_dir=output_dir,
            env=env,
            timeout=900,
            logger=logger,
        )

        if result.returncode != 0:
            logger.error(f"Prediction failed: {result.stderr}")
            logger.error(f"Additional output: {result.stdout}")
            return None

        logger.info(f"Prediction completed: {result.stdout}")

        # Load test data for sepsis labels from the ORIGINAL (not the stripped workspace copy)
        with open(test_dataset_path, "r") as f:
            test_data = json.load(f)

        # Convert test data to DataFrame for sepsis labels
        test_df = pd.DataFrame(test_data)
        if "SepsisLabel" in test_df.columns:
            test_df = test_df.rename(columns={"SepsisLabel": "sepsis_label"})

        # Look for decisions.csv
        decisions_file = workspace_dir / "decisions.csv"

        if not decisions_file.exists():
            logger.error(f"Decisions file not found: {decisions_file}")
            return None

        decisions_df = pd.read_csv(decisions_file)
        logger.info(f"Found decisions file: {decisions_file}")

        # Merge decisions with test data to get sepsis labels
        if "patient_id" in test_df.columns and "patient_id" in decisions_df.columns:
            merged_data = decisions_df.merge(test_df[["patient_id", "sepsis_label"]], on="patient_id", how="left")
        else:
            merged_data = decisions_df
            logger.warning("Could not merge sepsis labels - patient_id column not found")

        logger.info("Evaluating sepsis treatment policy")

        all_results = {
            "total_patients": len(merged_data),
        }

        # Compute sepsis metrics
        if "decision" in merged_data.columns:
            logger.info("Computing sepsis treatment metrics...")

            metrics = compute_all_metrics(merged_data, eval_config)
            all_results.update(metrics)

            if not quiet:
                display_metrics(metrics)

            logger.info("Computed all sepsis treatment metrics")
        else:
            logger.warning("Missing 'decision' column in decisions.csv, cannot compute treatment metrics")

        return all_results

    except Exception as e:
        logger.error(f"Error in prediction evaluation: {e}")
        logger.error(traceback.format_exc())
        return None
