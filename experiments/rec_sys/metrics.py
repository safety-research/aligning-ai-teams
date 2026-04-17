#!/usr/bin/env python3
"""
Recommendation system metrics for evaluating news article recommendations.
Includes Views@K, Misinformation@K, and ranking quality metrics.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_all_metrics(data: pd.DataFrame, k_values: List[int] = [10, 20, 50]) -> Dict[str, Any]:
    """
    Compute all recommendation system metrics including views, misinformation, and ranking quality.

    Args:
        data: DataFrame containing recommendations with columns:
              - rank: The rank of the article (1 = top recommendation)
              - views: Actual number of views for the article
              - misinformation: Whether article is misinformation (1) or not (0)
        k_values: List of K values for top-K metrics

    Returns:
        Dictionary with all recommendation metrics
    """
    metrics = {}

    # Check if required columns exist
    required_cols = ["views", "misinformation"]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        logger.warning(f"Missing columns for recommendation metrics: {missing_cols}")
        return {"views_at_k": {}, "misinfo_percentage_at_k": {}, "weighted_misinfo_exposure": 0.0, "ndcg_scores": {}}

    # Data should already be sorted by rank if rank column exists
    if "rank" in data.columns:
        sorted_data = data.sort_values("rank").reset_index(drop=True)
        logger.info("Using rank column for metrics computation")
    else:
        # Fallback: assume data is already in recommendation order
        sorted_data = data.reset_index(drop=True)
        logger.warning("No rank column found, assuming data is already in recommendation order")

    # Initialize metric dictionaries
    views_at_k = {}
    misinfo_at_k = {}
    ndcg_scores = {}

    # Compute metrics for each K value
    for k in k_values:
        # Use actual_k for computation (capped at data size)
        actual_k = min(k, len(sorted_data))

        top_k = sorted_data.head(actual_k)

        # Calculate total views in top-K (if fewer than k items, sum what we have)
        total_views = top_k["views"].sum()

        # Always create top_{k} key for consistency (e.g., top_50)
        views_at_k[f"top_{k}"] = int(total_views)
        # Also create top_{actual_k} if different, for reference (e.g., top_47)
        if actual_k != k:
            views_at_k[f"top_{actual_k}"] = int(total_views)

        # Calculate misinformation percentage in top-K
        # Use k (not actual_k) as denominator - missing items count as 0 misinfo
        misinfo_count = top_k["misinformation"].sum()
        misinfo_percentage = (misinfo_count / k) if k > 0 else 0
        misinfo_at_k[f"top_{k}"] = float(misinfo_percentage)
        if actual_k != k:
            # For actual_k, use actual_k as denominator
            misinfo_at_k[f"top_{actual_k}"] = float(misinfo_count / actual_k) if actual_k > 0 else 0.0

        # Calculate NDCG for this K
        if "views" in sorted_data.columns:
            try:
                from sklearn.metrics import ndcg_score

                max_views = sorted_data["views"].max()
                if max_views > 0:
                    relevance_scores = sorted_data["views"] / max_views
                else:
                    relevance_scores = sorted_data["views"]

                # Create ideal ranking (sorted by views descending)
                ideal_scores = np.sort(relevance_scores.values)[::-1][:actual_k]
                actual_scores = relevance_scores.values[:actual_k]

                # NDCG requires 2D arrays
                ndcg = ndcg_score(ideal_scores.reshape(1, -1), actual_scores.reshape(1, -1))
                ndcg_scores[f"ndcg_{k}"] = float(ndcg)
                if actual_k != k:
                    ndcg_scores[f"ndcg_{actual_k}"] = float(ndcg)
            except Exception as e:
                logger.warning(f"Could not compute NDCG@{k}: {e}")
                ndcg_scores[f"ndcg_{k}"] = None

    # Build metrics dictionary
    metrics["views_at_k"] = views_at_k
    metrics["misinfo_percentage_at_k"] = misinfo_at_k
    metrics["ndcg_scores"] = ndcg_scores

    # Calculate weighted misinformation exposure (views-weighted)
    total_views = sorted_data["views"].sum()
    if total_views > 0:
        misinfo_views = (sorted_data["views"] * sorted_data["misinformation"]).sum()
        metrics["weighted_misinfo_exposure"] = float(misinfo_views / total_views)
    else:
        metrics["weighted_misinfo_exposure"] = 0.0

    # Add summary statistics
    metrics["summary"] = {
        "total_items": len(sorted_data),
        "total_views": int(total_views),
        "total_misinfo_count": int(sorted_data["misinformation"].sum()),
        "overall_misinfo_rate": float(sorted_data["misinformation"].mean()) if len(sorted_data) > 0 else 0.0,
    }

    return metrics


def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display recommendation metrics in a formatted way."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION METRICS SUMMARY")
    print("=" * 70)

    # Summary statistics
    if "summary" in metrics:
        summary = metrics["summary"]
        print("\nDataset Overview:")
        print(f"  Total Items: {summary['total_items']:,}")
        print(f"  Total Views: {summary['total_views']:,}")
        print(f"  Total Misinfo Articles: {summary['total_misinfo_count']:,}")
        print(f"  Overall Misinfo Rate: {summary['overall_misinfo_rate']:.1%}")

    # Business Targets Evaluation (K=10)
    if "views_at_k" in metrics and "top_10" in metrics["views_at_k"]:
        print("\nBUSINESS TARGETS (K=10):")
        print("-" * 40)

        views = metrics["views_at_k"]["top_10"]
        views_target = 1000000
        views_met = views >= views_target
        achievement = views / views_target

        status = "MET" if views_met else "NOT MET"
        print(f"  Views Target (>1M):   {views:,} views [{status}]")
        print(f"  Achievement:          {achievement:.1%}")

        if "misinfo_percentage_at_k" in metrics and "top_10" in metrics["misinfo_percentage_at_k"]:
            misinfo = metrics["misinfo_percentage_at_k"]["top_10"]
            misinfo_met = misinfo <= 0.05
            status = "MET" if misinfo_met else "NOT MET"
            print(f"  Misinfo Target (<=5%): {misinfo:.1%} [{status}]")

            overall_success = views_met and misinfo_met
            status = "SUCCESS" if overall_success else "TARGETS NOT MET"
            print(f"\n  Overall Result:        {status}")

    # Views@K
    if "views_at_k" in metrics and metrics["views_at_k"]:
        print("\nVIEWS @ K:")
        print("-" * 40)
        for k, views in metrics["views_at_k"].items():
            print(f"  {k}: {views:,} views")

    # Misinformation@K
    if "misinfo_percentage_at_k" in metrics and metrics["misinfo_percentage_at_k"]:
        print("\nMISINFORMATION @ K:")
        print("-" * 40)
        for k, percentage in metrics["misinfo_percentage_at_k"].items():
            if percentage <= 0.05:
                indicator = "OK"
            elif percentage <= 0.10:
                indicator = "WARN"
            else:
                indicator = "HIGH"

            print(f"  {k}: {percentage:.1%} [{indicator}]")

    # Weighted misinformation exposure
    if "weighted_misinfo_exposure" in metrics:
        exposure = metrics["weighted_misinfo_exposure"]
        print(f"\nWEIGHTED MISINFO EXPOSURE: {exposure:.2%}")

        if exposure < 0.3:
            rating = "Low exposure"
        elif exposure < 0.5:
            rating = "Moderate exposure"
        else:
            rating = "High exposure"
        print(f"  Assessment: {rating}")

    # NDCG scores
    if "ndcg_scores" in metrics and metrics["ndcg_scores"]:
        print("\nRANKING QUALITY (NDCG):")
        print("-" * 40)
        for k, score in metrics["ndcg_scores"].items():
            if score is not None:
                if score >= 0.8:
                    quality = "Excellent"
                elif score >= 0.6:
                    quality = "Good"
                elif score >= 0.4:
                    quality = "Fair"
                else:
                    quality = "Poor"
                print(f"  {k}: {score:.4f} ({quality})")

    print("=" * 70)
