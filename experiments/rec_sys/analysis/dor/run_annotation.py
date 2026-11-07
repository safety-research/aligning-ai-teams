"""Annotate rec_sys runs for diffusion-of-responsibility behaviors.

Default scope is the full sweep: every model in MODELS × every org in ORGS,
filtered to outcome ∉ {timeout, failure} and non-NaN views_at_50 / misinfo_at_50.
Each flag NARROWS the scope; nothing widens it.

Usage:
    uv run python run_annotation.py                                # full sweep
    uv run python run_annotation.py --models claude-opus-4-6       # one model
    uv run python run_annotation.py --orgs single_base             # one org
    uv run python run_annotation.py --max-runs 3                   # smoke test (3/model)
    uv run python run_annotation.py --per-org 5                    # 5 runs/org/model
    uv run python run_annotation.py --concurrency 10               # adjust semaphore

Each model gets up to MAX_ATTEMPTS passes; after every pass we identify runs
that didn't produce a valid v1.4 dor.json and re-annotate them. After all
models finish, run `aggregate_dor.py` to build the summary CSV/figure.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from dotenv import find_dotenv, load_dotenv

from prompt import SYSTEM_PROMPT, build_user_prompt
from validate_dor import SCHEMA_VERSION, validate

load_dotenv(find_dotenv(usecwd=True))

# ── Config ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
RESULTS_ROOT = ANALYSIS_DIR.parent / "results"
CSV_PATH = ANALYSIS_DIR / "interventions" / "all_results.csv"
VALIDATOR_PATH = HERE / "validate_dor.py"

MODELS = [
    "claude-mythos-preview",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-1-20250805",
]
ORGS = [
    "single_base",
    "multi8_base",
    "multi8_safetydoc",
    "multi8_dri",
    "multi8_safetydocdri",
    "multi8_comms",
]
EXCLUDED_OUTCOMES = ("timeout", "failure")
# Drop runs whose recommendations.csv had <= MIN_TOTAL_ITEMS rows. Such runs
# produce artificially low misinfo_at_50 because missing slots are scored as
# 0 misinfo (per the rec_sys eval logic), so they look "safe" but are actually
# degenerate. Mirrors the interventions analysis filter.
MIN_TOTAL_ITEMS = 35
ANNOTATOR_MODEL = "claude-sonnet-4-6"
CONCURRENCY = 25
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_annotation")


# ── Run discovery ────────────────────────────────────────────────────────────
def filter_runs(csv_path: Path, model: str, orgs: list[str]) -> pd.DataFrame:
    """Apply walkthrough filters: model, org set, drop NaN views_at_50/misinfo_at_50,
    drop excluded outcomes, drop degenerate runs with total_items <= MIN_TOTAL_ITEMS.
    Asserts misinfo_at_50 is in [0,1]."""
    df = pd.read_csv(csv_path)
    df = df[df["model"] == model]
    df = df[df["org_name"].isin(orgs)]
    df = df[~df["outcome"].isin(EXCLUDED_OUTCOMES)]
    df = df.dropna(subset=["views_at_50", "misinfo_at_50"])
    df = df[df["total_items"] > MIN_TOTAL_ITEMS]
    if not df.empty:
        m_min, m_max = df["misinfo_at_50"].min(), df["misinfo_at_50"].max()
        if m_min < 0 or m_max > 1:
            raise ValueError(
                f"misinfo_at_50 outside [0,1] in {csv_path} (min={m_min}, max={m_max}); "
                "expected rate, got something else"
            )
    return df.reset_index(drop=True)


def resolve_run_dir(model: str, num_coders: int, org_name: str, run_id: str) -> Path | None:
    """Locate the run directory for a CSV row.

    The CSV stores canonical org_name (e.g. multi8_base) but on-disk dirs may
    have model-tag suffixes (multi8_base_o46). The parent dir is named
    `{N}_agent_experiments` where N = num_coders (no PM, e.g. single_base) or
    num_coders + 1 (with PM, e.g. multi8_*). We try both, plus globbing all
    `*_agent_experiments` siblings as a fallback.
    """
    model_root = RESULTS_ROOT / model
    if not model_root.exists():
        return None
    candidates = []
    for n in (num_coders, num_coders + 1):
        p = model_root / f"{n}_agent_experiments"
        if p.exists():
            candidates.append(p)
    if not candidates:
        candidates = [p for p in model_root.iterdir()
                      if p.is_dir() and p.name.endswith("_agent_experiments")]
    for parent in candidates:
        for org_dir in parent.iterdir():
            if not org_dir.is_dir():
                continue
            base = org_dir.name.rsplit("_", 1)[0] if "_" in org_dir.name else org_dir.name
            if org_dir.name != org_name and base != org_name:
                continue
            candidate = org_dir / run_id
            if candidate.exists():
                return candidate
    return None


def list_transcripts(run_dir: Path) -> list[str]:
    """Return sorted .txt filenames inside logs/transcripts/."""
    tdir = run_dir / "logs" / "transcripts"
    if not tdir.exists():
        return []
    return sorted(p.name for p in tdir.glob("*.txt"))


def is_valid_dor(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return (d.get("schema_version") == SCHEMA_VERSION
            and isinstance(d.get("annotations"), list)
            and len(d["annotations"]) > 0)


def find_missing(eligible: pd.DataFrame) -> pd.DataFrame:
    """Filter to runs that have a resolvable run_dir but no valid dor.json yet."""
    rows = []
    for _, row in eligible.iterrows():
        rd = resolve_run_dir(row["model"], int(row["num_coders"]),
                             row["org_name"], row["run_id"])
        if rd is None:
            continue  # structurally unresolvable; skip rather than infinite-retry
        if not is_valid_dor(rd / "dor.json"):
            rows.append(row)
    return pd.DataFrame(rows)


# ── Annotation ───────────────────────────────────────────────────────────────
async def annotate_one(row: pd.Series, run_dir: Path, transcripts: list[str]) -> tuple[bool, str]:
    """Invoke the SDK on a single run. Returns (ok, status) where status is
    'done', a validation error summary, or an SDK exception summary."""
    user_prompt = build_user_prompt(
        run_dir=str(run_dir),
        model=row["model"],
        org_name=row["org_name"],
        run_id=row["run_id"],
        views_at_50=float(row["views_at_50"]),
        misinfo_at_50=float(row["misinfo_at_50"]),
        weighted_misinfo_exposure=float(row["weighted_misinfo_exposure"]),
        transcript_files=transcripts,
        validator_path=str(VALIDATOR_PATH),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="claude_cfg_") as cfg_dir:
            options = ClaudeAgentOptions(
                model=ANNOTATOR_MODEL,
                cwd=str(run_dir),
                system_prompt=SYSTEM_PROMPT,
                allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
                permission_mode="acceptEdits",
                # Default SDK buffer is 1MB which overflows for runs with large
                # tool results (huge transcripts being read). Bump to 64MB.
                max_buffer_size=64 * 1024 * 1024,
                env={
                    "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "3600000",
                    "CLAUDE_CONFIG_DIR": cfg_dir,
                    "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
                },
            )
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_prompt)
                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        break
    except Exception as e:
        return False, f"sdk: {type(e).__name__}: {e}"

    errors = validate(run_dir / "dor.json")
    return (not errors), ("done" if not errors else f"validation: {errors}")


async def bounded_annotate(
    sem: asyncio.Semaphore,
    row: pd.Series,
    idx: int,
    total: int,
) -> None:
    """Annotate one run under the semaphore. Logs per-run; returns nothing."""
    run_dir = resolve_run_dir(
        row["model"], int(row["num_coders"]), row["org_name"], row["run_id"]
    )
    label = f"{row['model']}/{row['org_name']}/{row['run_id']}"
    if run_dir is None:
        logger.warning("[%d/%d] FAIL %s discovery: run_dir not found", idx + 1, total, label)
        return
    transcripts = list_transcripts(run_dir)
    if not transcripts:
        logger.warning("[%d/%d] FAIL %s discovery: no transcripts in logs/transcripts/",
                       idx + 1, total, label)
        return

    async with sem:
        logger.info("[%d/%d] start %s (%d transcripts, misinfo@50=%.2f)",
                    idx + 1, total, label, len(transcripts), row["misinfo_at_50"])
        ok, status = await annotate_one(row, run_dir, transcripts)
        if ok:
            logger.info("[%d/%d] OK   %s", idx + 1, total, label)
        else:
            logger.warning("[%d/%d] FAIL %s %s", idx + 1, total, label, status)


async def annotate_runs(df: pd.DataFrame, concurrency: int) -> None:
    """Annotate every row in df under a semaphore."""
    total = len(df)
    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*[
        bounded_annotate(sem, row, i, total)
        for i, (_, row) in enumerate(df.iterrows())
    ])


# ── Per-model orchestration with retry ───────────────────────────────────────
async def annotate_model(
    model: str,
    orgs: list[str],
    csv_path: Path,
    concurrency: int,
    max_attempts: int,
    max_runs: int | None,
    per_org: int | None,
) -> dict:
    """Annotate a single model's runs, retrying any that fail to produce a valid
    dor.json. Returns a per-model summary dict."""
    eligible = filter_runs(csv_path, model, orgs)
    if per_org:
        eligible = pd.concat(
            [g.head(per_org) for _, g in eligible.groupby("org_name", sort=False)],
            ignore_index=True,
        )
    if max_runs:
        eligible = eligible.head(max_runs)
    if eligible.empty:
        logger.info("=== %s: no eligible runs ===", model)
        return {"model": model, "eligible": 0, "annotated": 0, "missing": 0}

    logger.info("=== %s: %d eligible runs ===", model, len(eligible))
    counts = eligible.groupby("org_name").size().to_dict()
    logger.info("  by org: %s", counts)

    for attempt in range(1, max_attempts + 1):
        # Always restrict to runs without a valid dor.json — makes restart
        # idempotent and avoids redoing work after a crash/kill.
        to_run = find_missing(eligible)
        if to_run.empty:
            logger.info("%s: all eligible runs annotated after %d attempt(s)",
                        model, attempt - 1)
            break
        if attempt > 1:
            logger.info("%s: attempt %d on %d missing runs (after %ds backoff)",
                        model, attempt, len(to_run), RETRY_BACKOFF_SECONDS)
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
        logger.info("%s: launching attempt %d on %d runs", model, attempt, len(to_run))
        await annotate_runs(to_run, concurrency)

    final_missing = find_missing(eligible)
    annotated = len(eligible) - len(final_missing)
    logger.info("%s done: %d/%d annotated, %d still missing",
                model, annotated, len(eligible), len(final_missing))
    return {
        "model": model,
        "eligible": len(eligible),
        "annotated": annotated,
        "missing": len(final_missing),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=MODELS,
                   help="restrict to these models (default: all)")
    p.add_argument("--orgs", nargs="+", default=ORGS,
                   help="restrict to these orgs (default: all)")
    p.add_argument("--max-runs", type=int, default=None,
                   help="cap the total number of runs per model (head after filtering)")
    p.add_argument("--per-org", type=int, default=None,
                   help="take only the first N runs per org (per model)")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                   help="max passes per model including the initial attempt (default 3)")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    return p.parse_args()


async def amain() -> int:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY is not set")
        return 2

    logger.info(
        "scope: models=%s orgs=%s concurrency=%d max_attempts=%d annotator=%s",
        args.models, args.orgs, args.concurrency, args.max_attempts, ANNOTATOR_MODEL,
    )

    summaries = []
    for model in args.models:
        try:
            summary = await annotate_model(
                model=model,
                orgs=args.orgs,
                csv_path=args.csv,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                max_runs=args.max_runs,
                per_org=args.per_org,
            )
        except Exception as e:
            logger.exception("%s: orchestration error: %s", model, e)
            summary = {"model": model, "error": str(e)}
        summaries.append(summary)

    logger.info("=== ALL DONE ===")
    total_missing = 0
    any_error = False
    for s in summaries:
        if "error" in s:
            logger.warning("  %s: ERROR %s", s["model"], s["error"])
            any_error = True
        else:
            logger.info("  %s: %d/%d annotated, %d missing",
                        s["model"], s["annotated"], s["eligible"], s["missing"])
            total_missing += s["missing"]
    return 0 if not any_error and total_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
