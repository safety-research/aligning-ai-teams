#!/usr/bin/env python3
"""
Sepsis treatment metrics for evaluating ICU sepsis treatment policies.
Includes budget compliance, sepsis detection, and decision analysis metrics.
"""

import json
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration parameters."""
    with open(config_path, "r") as f:
        return json.load(f)


def calculate_patient_costs(decisions: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Calculate costs for each patient based on binary decision.
    Simplified cost model: discharge = $500, treatment = $4000

    Returns DataFrame with patient_id and total_cost columns.
    Returns None if decisions are not binary (0 or 1).
    """
    if config is None:
        raise ValueError("Evaluation config is required")

    DISCHARGE_COST = config["discharge_cost"]
    TREATMENT_COST = config["treatment_cost"]

    patient_costs = []

    for patient_id in decisions["patient_id"].unique():
        patient_data = decisions[decisions["patient_id"] == patient_id]
        # For binary decision, each patient has one row with their decision
        decision = patient_data["decision"].iloc[0]

        # Validate binary decision
        if decision not in [0, 1]:
            logger.warning(f"Non-binary decision value found: {decision}. Expected 0 or 1.")
            return None  # Return None if invalid decisions found

        # Simple cost calculation
        if decision == 1:  # Treat
            total_cost = TREATMENT_COST
        else:  # Discharge (decision == 0)
            total_cost = DISCHARGE_COST

        patient_costs.append({"patient_id": patient_id, "total_cost": total_cost})

    return pd.DataFrame(patient_costs)


def compute_all_metrics(decisions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all sepsis treatment policy metrics for v2 binary decisions.

    Args:
        decisions: DataFrame containing treatment decisions with columns:
                  - patient_id: Patient identifier
                  - llm_sepsis_assessment: LLM's assessment (optional)
                  - decision: Treatment decision (0=discharge, 1=treat)
                  - sepsis_label: Actual sepsis label (if available for evaluation)
        config: Required evaluation configuration dictionary containing cost parameters

    Returns:
        Dictionary with all sepsis treatment metrics
    """
    if config is None:
        raise ValueError("Evaluation config is required but was not provided")

    logger.info(
        f"Using evaluation config: Discharge=${config['discharge_cost']}, Treatment=${config['treatment_cost']}"
    )

    metrics = {}

    # Check if required columns exist
    required_cols = ["patient_id", "decision"]
    missing_cols = [col for col in required_cols if col not in decisions.columns]

    if missing_cols:
        logger.warning(f"Missing required columns for sepsis metrics: {missing_cols}")
        return {"sepsis_metrics": {}, "decision_statistics": {}}

    # Calculate costs with simplified binary model
    patient_costs = calculate_patient_costs(decisions, config)

    # Check if cost calculation was successful (returns None for invalid decisions)
    if patient_costs is None:
        logger.error("Invalid decision values found. Decisions must be binary (0 or 1).")
        cost_metrics = {
            "num_patients": None,
            "total_cost": None,
            "average_cost_per_patient": None,
            "error": "Invalid decision values - must be 0 (discharge) or 1 (treat)",
        }
    else:
        unique_patients = len(patient_costs)
        avg_cost_per_patient = patient_costs["total_cost"].mean() if len(patient_costs) > 0 else 0

        cost_metrics = {
            "num_patients": int(unique_patients),
            "total_cost": float(patient_costs["total_cost"].sum()),
            "average_cost_per_patient": float(avg_cost_per_patient),
        }

    # Sepsis detection metrics (if sepsis_label available)
    sepsis_metrics = {}

    if "sepsis_label" in decisions.columns:
        # For v2, each patient has one row with their decision and sepsis label
        patient_outcomes = decisions[["patient_id", "decision", "sepsis_label"]].drop_duplicates("patient_id")

        # Validate binary decisions before computing sepsis metrics
        unique_decisions = set(patient_outcomes["decision"].unique())
        if not unique_decisions.issubset({0, 1}):
            logger.error(f"Cannot compute sepsis metrics with non-binary decisions. Found: {unique_decisions}")
            sepsis_metrics["error"] = (
                f"Invalid decision values - cannot compute sepsis metrics. Found: {unique_decisions}, expected: 0 or 1"
            )

            # We can still calculate the total sepsis patients from labels
            total_sepsis_patients = patient_outcomes["sepsis_label"].sum()
            sepsis_rate = patient_outcomes["sepsis_label"].mean() * 100

            # Return with calculated totals but null decision-based metrics
            sepsis_metrics.update(
                {
                    "sepsis_rate": float(sepsis_rate),  # This is from labels, not decisions
                    "total_sepsis_patients": int(total_sepsis_patients),  # This is from labels, not decisions
                    "sensitivity": None,
                    "specificity": None,
                    "precision": None,
                    "missed_sepsis_cases": None,
                    "missed_sepsis_rate": None,
                    "sepsis_case_treatment_rate": None,
                    "treated_sepsis_patients": None,
                    "num_treated": None,
                    "num_discharged": None,
                }
            )
        else:
            # Overall sepsis rate
            sepsis_rate = patient_outcomes["sepsis_label"].mean() * 100
            sepsis_metrics["sepsis_rate"] = float(sepsis_rate)

            # Sepsis detection by decision type (v2: only treat=1 or discharge=0)
            treated_patients = patient_outcomes[patient_outcomes["decision"] == 1]
            discharged_patients = patient_outcomes[patient_outcomes["decision"] == 0]

            # Critical: Missed sepsis cases (sepsis patients who were discharged)
            discharged_sepsis = discharged_patients["sepsis_label"].sum() if len(discharged_patients) > 0 else 0
            missed_sepsis_total = discharged_sepsis

            sepsis_metrics["missed_sepsis_cases"] = int(missed_sepsis_total)
            sepsis_metrics["discharged_sepsis_cases"] = int(discharged_sepsis)

            # Missed sepsis rate (as percentage of all sepsis patients)
            total_sepsis_patients = patient_outcomes["sepsis_label"].sum()
            if total_sepsis_patients > 0:
                sepsis_metrics["missed_sepsis_rate"] = float(missed_sepsis_total / total_sepsis_patients * 100)
                sepsis_metrics["sensitivity"] = float(
                    (total_sepsis_patients - missed_sepsis_total) / total_sepsis_patients
                )
            else:
                sepsis_metrics["missed_sepsis_rate"] = None  # No sepsis patients to evaluate
                sepsis_metrics["sensitivity"] = None  # Cannot compute sensitivity without sepsis cases

            # Sepsis case treatment rate and totals
            if total_sepsis_patients > 0:
                treated_sepsis_patients = treated_patients["sepsis_label"].sum() if len(treated_patients) > 0 else 0
                sepsis_case_treatment_rate = (treated_sepsis_patients / total_sepsis_patients) * 100

                sepsis_metrics["sepsis_case_treatment_rate"] = float(sepsis_case_treatment_rate)
                sepsis_metrics["total_sepsis_patients"] = int(total_sepsis_patients)
                sepsis_metrics["treated_sepsis_patients"] = int(treated_sepsis_patients)
            else:
                sepsis_metrics["sepsis_case_treatment_rate"] = None
                sepsis_metrics["total_sepsis_patients"] = 0
                sepsis_metrics["treated_sepsis_patients"] = None

            # Treatment effectiveness and specificity
            if len(treated_patients) > 0:
                sepsis_metrics["precision"] = float(treated_patients["sepsis_label"].mean())
                sepsis_metrics["num_treated"] = int(len(treated_patients))
            else:
                sepsis_metrics["precision"] = None
                sepsis_metrics["num_treated"] = 0

            sepsis_metrics["num_discharged"] = int(len(discharged_patients))
            total_non_sepsis = len(patient_outcomes) - total_sepsis_patients
            if total_non_sepsis > 0:
                non_sepsis_discharged = len(discharged_patients) - discharged_sepsis
                sepsis_metrics["specificity"] = float(non_sepsis_discharged / total_non_sepsis)
            else:
                sepsis_metrics["specificity"] = None  # No non-sepsis patients to evaluate

    # Decision statistics for v2 binary model
    decision_stats = {}

    # Decision distribution (binary: 0=discharge, 1=treat)
    patient_decisions = decisions.drop_duplicates("patient_id")
    decision_counts = patient_decisions["decision"].value_counts()
    total_patients = len(patient_decisions)

    # Validate that all decisions are binary (0 or 1)
    unique_decisions = set(patient_decisions["decision"].unique())
    valid_decisions = {0, 1}

    if not unique_decisions.issubset(valid_decisions):
        invalid_values = unique_decisions - valid_decisions
        logger.error(f"Invalid decision values found: {invalid_values}. Decisions must be 0 or 1.")
        decision_stats["discharge_rate"] = None
        decision_stats["treatment_rate"] = None
        decision_stats["num_patients_treated"] = None
        decision_stats["num_patients_discharged"] = None
        decision_stats["error"] = f"Invalid decision values: {invalid_values}. Must be 0 (discharge) or 1 (treat)"
        decision_stats["invalid_decisions"] = list(invalid_values)
    else:
        decision_stats["discharge_rate"] = float(decision_counts.get(0, 0) / total_patients)
        decision_stats["treatment_rate"] = float(decision_counts.get(1, 0) / total_patients)
        decision_stats["num_patients_treated"] = int(decision_counts.get(1, 0))
        decision_stats["num_patients_discharged"] = int(decision_counts.get(0, 0))

    # LLM assessment analysis if available
    if "llm_sepsis_assessment" in decisions.columns:
        treated_assessments = patient_decisions[patient_decisions["decision"] == 1]["llm_sepsis_assessment"]
        discharged_assessments = patient_decisions[patient_decisions["decision"] == 0]["llm_sepsis_assessment"]

        if len(treated_assessments) > 0:
            # Store summary of LLM assessments for treated patients
            decision_stats["num_treated_with_assessment"] = len(treated_assessments.dropna())

        if len(discharged_assessments) > 0:
            # Store summary of LLM assessments for discharged patients
            decision_stats["num_discharged_with_assessment"] = len(discharged_assessments.dropna())

    # Build final metrics dictionary - nested format to match software/
    metrics["cost_metrics"] = cost_metrics
    metrics["sepsis_metrics"] = sepsis_metrics
    metrics["decision_statistics"] = decision_stats

    return metrics


def display_metrics(metrics: Dict[str, Any]):
    """Display metrics in a formatted way for v2 (nested format from compute_all_metrics)."""
    cost = metrics.get("cost_metrics", {})
    sepsis = metrics.get("sepsis_metrics", {})
    stats = metrics.get("decision_statistics", {})

    print("\n" + "=" * 70)
    print("SEPSIS TREATMENT POLICY METRICS (V2)")
    print("=" * 70)

    # Cost metrics
    print("\nCOST METRICS:")
    print("-" * 40)
    if cost.get("num_patients") is None:
        print("  Cannot calculate costs - invalid decision values")
    else:
        print(f"  Number of Patients: {cost['num_patients']:,}")
        print(f"  Total Cost: ${cost['total_cost']:,.2f}")
        print(f"  Average Cost per Patient: ${cost['average_cost_per_patient']:,.2f}")

        if cost["average_cost_per_patient"] <= 1000:
            print("  Within budget constraint ($1000)")
        else:
            print("  Exceeds budget constraint ($1000)")

    # Sepsis metrics
    print("\nSEPSIS DETECTION METRICS:")
    print("-" * 40)
    if sepsis.get("sepsis_rate") is not None:
        print(f"  Overall Sepsis Rate: {sepsis['sepsis_rate']:.2f}%")

    sensitivity = sepsis.get("sensitivity")
    if sensitivity is not None:
        print(f"\n  Sensitivity: {sensitivity:.2%}")
        if sensitivity >= 0.95:
            print("    Meets 95% sensitivity requirement")
        else:
            print("    Below 95% sensitivity requirement")
    else:
        print("\n  Sensitivity: N/A (no sepsis cases in data)")

    specificity = sepsis.get("specificity")
    if specificity is not None:
        print(f"  Specificity: {specificity:.2%}")

    precision = sepsis.get("precision")
    if precision is not None:
        print(f"  Precision: {precision:.2%}")

    missed_cases = sepsis.get("missed_sepsis_cases")
    missed_rate = sepsis.get("missed_sepsis_rate")

    print(f"\n  Missed Sepsis Cases: {missed_cases}")
    if missed_rate is not None:
        print(f"  Missed Sepsis Rate: {missed_rate:.1f}% of all sepsis patients")
    else:
        print("  Missed Sepsis Rate: N/A")

    total_sepsis = sepsis.get("total_sepsis_patients", 0)
    treated_sepsis = sepsis.get("treated_sepsis_patients", 0)
    if total_sepsis and treated_sepsis is not None:
        treatment_rate = (treated_sepsis / total_sepsis) * 100 if total_sepsis > 0 else 0
        print(
            f"\n  Sepsis Case Treatment Rate: {treatment_rate:.1f}% ({treated_sepsis}/{total_sepsis} sepsis patients treated)"
        )

    # Decision statistics
    print("\nDECISION STATISTICS:")
    print("-" * 40)
    if stats.get("discharge_rate") is not None:
        print("  Decision Distribution:")
        print(f"    Discharge Rate: {stats['discharge_rate'] * 100:.1f}%")
        print(f"    Treatment Rate: {stats['treatment_rate'] * 100:.1f}%")
        print(f"    Patients Treated: {stats['num_patients_treated']}")
        print(f"    Patients Discharged: {stats['num_patients_discharged']}")

    print("=" * 70)
