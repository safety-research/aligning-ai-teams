#!/usr/bin/env python3
"""
Evaluation functions for rec_sys experiment.
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


async def run_recommendation_and_evaluate(
    output_dir: Path,
    test_dataset_path: str,
    eval_config_path: str,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run recommend.py on test dataset and calculate recommendation metrics.

    Args:
        output_dir: Path to experiment directory containing workspace
        test_dataset_path: Path to test dataset (CSV file with article data)
        eval_config_path: Path to evaluation config JSON file
        quiet: Suppress metric display
        logger: Optional logger for concurrent run isolation

    Returns:
        Dictionary containing recommendation metrics, or None if recommend.py doesn't exist
    """
    logger = logger or _default_logger

    try:
        workspace_dir = output_dir / "workspace"
        recommend_file = workspace_dir / "src" / "recommend.py"

        # Check if recommend.py exists
        if not recommend_file.exists():
            logger.info("recommend.py not found, skipping recommendation evaluation")
            return None

        logger.info("Found recommend.py for prompt-based recommendation system")

        # Load evaluation config (required)
        if not eval_config_path or not Path(eval_config_path).exists():
            logger.error(f"Evaluation config not found: {eval_config_path}")
            raise ValueError(f"Evaluation config file is required but not found: {eval_config_path}")

        with open(eval_config_path, "r") as f:
            eval_config = json.load(f)
        logger.info(f"Loaded evaluation config from {eval_config_path}")

        # Copy test dataset to workspace directory, stripping ground truth labels
        # to prevent label leakage (agents must not see views or misinformation)
        test_file_dest = workspace_dir / "data" / "test.csv"
        test_file_dest.parent.mkdir(parents=True, exist_ok=True)
        test_df_full = pd.read_csv(test_dataset_path)
        test_df_full.drop(columns=["misinformation", "views"], errors="ignore").to_csv(test_file_dest, index=False)
        logger.info(f"Copied test dataset to {test_file_dest} (ground truth stripped)")

        # Run recommendation script on test dataset
        logger.info("Running recommend.py on test dataset...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace_dir)
        result = await run_in_workspace(
            ["python", "src/recommend.py", "data/test.csv"],
            output_dir=output_dir,
            env=env,
            timeout=900,
            logger=logger,
        )

        if result.returncode != 0:
            logger.error(f"Recommendation failed: {result.stderr}")
            logger.error(f"Additional output: {result.stdout}")
            return None

        logger.info(f"Recommendation completed: {result.stdout}")

        # Load test data from the ORIGINAL (not the stripped workspace copy) for ground truth
        test_data = pd.read_csv(test_dataset_path)

        # Look for recommendations.csv
        recommendations_file = workspace_dir / "recommendations.csv"

        if not recommendations_file.exists():
            logger.error(f"Recommendations file not found: {recommendations_file}")
            return None

        output_data = pd.read_csv(recommendations_file)
        logger.info(f"Found recommendations file: {recommendations_file}")

        # Merge test data with output data using configured ID column
        id_column = eval_config.get("id_column", "id")

        # Handle different ID column names in recommendations vs test data
        if "article_id" in output_data.columns and id_column not in output_data.columns:
            # Recommendations uses 'article_id' but test data uses 'id'
            merged_data = test_data.merge(output_data, left_on=id_column, right_on="article_id", how="inner")
        else:
            merged_data = test_data.merge(output_data, on=id_column, how="inner")

        logger.info("Evaluating recommendation system")

        # Initialize results
        all_results = {
            "total_items": len(merged_data),
        }

        # Compute recommendation metrics
        if "views" in merged_data.columns and "misinformation" in merged_data.columns:
            logger.info("Computing recommendation metrics...")

            # Use the rank column directly from recommendations
            if "rank" in output_data.columns:
                if "rank" not in merged_data.columns:
                    merged_data["rank"] = output_data.set_index(id_column).loc[merged_data[id_column], "rank"].values
                # Sort by rank to get top-K recommendations
                merged_data = merged_data.sort_values("rank")
                logger.info(f"Using rank column for evaluation, found {len(merged_data)} ranked items")

            # Get k_values from config or use defaults
            k_values = eval_config.get("k_values", [10, 20, 50])

            # Compute all metrics in one go
            metrics = compute_all_metrics(merged_data, k_values=k_values)
            all_results.update(metrics)

            if not quiet:
                display_metrics(metrics)

            logger.info("Computed all recommendation metrics")
        else:
            logger.warning(
                "Missing 'views' or 'misinformation' columns in test data, cannot compute recommendation metrics"
            )

        return all_results

    except Exception as e:
        logger.error(f"Error in recommendation evaluation: {e}")
        logger.error(traceback.format_exc())
        return None
