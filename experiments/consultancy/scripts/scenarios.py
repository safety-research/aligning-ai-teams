import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_scenarios() -> dict:
    with open(DATA_DIR / "scenarios.json") as f:
        return json.load(f)

def get_scenario(name: str) -> dict:
    scenarios = load_scenarios()
    if name not in scenarios:
        available = sorted(k for k in scenarios.keys() if k != "baselines")
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")
    scenario = dict(scenarios[name])
    scenario["content"] = scenario["content"].strip()
    return scenario

ALL_SCENARIOS = [
    "loan_profit", "zero_interest_bnpl_marketing", "detroit_auto_plant",
    "acquisition_valuation", "painkiller_sales", "election_sentiment",
    "sanctions_circumvention", "social_media_growth", "downsizing_company",
    "regional_hospital_costs", "app_growth", "birthrate_improvement",
    "church_growth", "immigrant_tracking", "homeless_reduction",
    "standard", "ethical_dilemma", "technical_challenge",
]
