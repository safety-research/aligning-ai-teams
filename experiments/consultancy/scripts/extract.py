#!/usr/bin/env python3
"""
Extract the final proposal from an experiment run.

Primary: find the highest-versioned proposal_v*.txt in workspace/
(written by comms_director each turn: proposal_v1.txt, proposal_v2.txt, ...).

Stub detection: the comms_director sometimes writes the real deliverable to a
separate .md file and leaves a short redirect in proposal_v*.txt. When this
happens, we recover the real content by parsing the stub for .md references
or falling back to the previous proposal version.

Fallback: scan wire transcripts for the longest <CLIENT_REPORT>...</CLIENT_REPORT>
block from the communications director.
"""

import re
from pathlib import Path
from typing import Optional

_STUB_THRESHOLD = 2000  # chars — real proposals are 5K+; stubs are < 2K


def extract_proposal(output_dir: Path) -> tuple[Optional[str], Optional[int]]:
    """Extract the final proposal from an experiment directory.

    Returns:
        (proposal_text, version_number) — version is the proposal_v{N}.txt number,
        or None if extracted from wire transcripts / not found.
    """
    output_dir = Path(output_dir)
    workspace = output_dir / "workspace"

    if workspace.exists():
        proposal_files = sorted(
            workspace.glob("proposal_v*.txt"),
            key=lambda p: _extract_version(p.name),
        )
        if proposal_files:
            result = _resolve_final(proposal_files, workspace)
            if result:
                return result

    # Fallback: extract longest <CLIENT_REPORT> from wire transcripts
    fallback = _extract_from_wire_transcripts(output_dir, "comms_director_0")
    return fallback, None


def extract_all_proposals(output_dir: Path) -> list[tuple[int, str]]:
    """Extract all proposal versions from an experiment directory.

    Returns:
        List of (version_number, proposal_text) sorted by version, ascending.
    """
    output_dir = Path(output_dir)
    workspace = output_dir / "workspace"
    if not workspace.exists():
        return []

    proposal_files = sorted(
        workspace.glob("proposal_v*.txt"),
        key=lambda p: _extract_version(p.name),
    )
    results = []
    for pf in proposal_files:
        text = pf.read_text().strip()
        if text:
            results.append((_extract_version(pf.name), text))

    # If the final version is a stub, replace it with recovered content
    if results:
        final_ver, final_text = results[-1]
        if _is_stub(final_text, proposal_files):
            resolved = _resolve_final(proposal_files, workspace)
            if resolved:
                text, ver = resolved
                results[-1] = (ver, text)
            else:
                results.pop()

    return results


# ─── Stub detection and recovery ─────────────────────────────────────────────


def _is_stub(text: str, all_proposal_files: list[Path]) -> bool:
    """Detect if a proposal_v*.txt is a redirect stub rather than real content.

    Two signals (both require < 2000 chars):
    1. Primary: previous version is 5x+ longer (content migrated elsewhere).
    2. Secondary: text references a .md file (proposals are always .txt).
    """
    if len(text) >= _STUB_THRESHOLD:
        return False
    # Primary: previous version is much longer
    if len(all_proposal_files) >= 2:
        prev_text = all_proposal_files[-2].read_text().strip()
        if len(prev_text) > len(text) * 5:
            return True
    # Secondary: references a .md file
    if _parse_md_refs(text):
        return True
    return False


def _resolve_final(
    proposal_files: list[Path], workspace: Path
) -> Optional[tuple[str, int]]:
    """Walk backward from the final proposal version to find real content.

    For each version (starting from the last):
    1. If not a stub → return it.
    2. If stub → check for a referenced .md file in workspace → return it.
    3. Otherwise → continue to the previous version.

    Returns (text, version_number) or None if nothing usable found.
    """
    for i in range(len(proposal_files) - 1, -1, -1):
        pf = proposal_files[i]
        text = pf.read_text().strip()
        version = _extract_version(pf.name)
        if not text:
            continue

        # Not a stub — use it
        files_up_to_here = proposal_files[: i + 1]
        if not _is_stub(text, files_up_to_here):
            return text, version

        # Stub — try to recover via referenced .md file
        recovered = _find_referenced_md(text, workspace)
        if recovered:
            return recovered, version

    return None


def _parse_md_refs(text: str) -> list[str]:
    """Extract .md filenames referenced in text.

    Matches patterns like /app/workspace/FINAL_REPORT.md or See CLIENT_PROPOSAL.md.
    Deduplicates, stripping the /app/workspace/ prefix.
    """
    refs = re.findall(r"/app/workspace/(\S+\.md)", text)
    refs += re.findall(r"[Ss]ee (\S+\.md)", text)
    seen = set()
    unique = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _find_referenced_md(stub_text: str, workspace: Path) -> Optional[str]:
    """Find the .md file referenced by a stub, if it exists in workspace."""
    for ref in _parse_md_refs(stub_text):
        fp = workspace / ref
        if fp.exists():
            text = fp.read_text().strip()
            if text and len(text) > len(stub_text):
                return text
    return None


# ─── Utilities ────────────────────────────────────────────────────────────────


def _extract_version(filename: str) -> int:
    """Extract version number from proposal_v{N}*.txt filenames.

    Handles suffixes like proposal_v7_FINAL.txt, proposal_v8_FINAL_APPROVED.txt.
    """
    match = re.search(r"proposal_v(\d+)", filename)
    return int(match.group(1)) if match else 0


def _extract_from_wire_transcripts(
    output_dir: Path, agent_id: str
) -> Optional[str]:
    """Scan wire transcript files for the longest CLIENT_REPORT from the given agent."""
    wires_dir = output_dir / "wires"
    if not wires_dir.exists():
        return None

    longest_report = None
    report_pattern = re.compile(
        r"<CLIENT_REPORT>(.*?)</CLIENT_REPORT>", re.DOTALL
    )

    for wire_file in sorted(wires_dir.glob("*.txt")):
        text = wire_file.read_text()
        messages = text.split("-" * 40)
        for msg in messages:
            msg = msg.strip()
            if not msg:
                continue
            lines = msg.split("\n", 1)
            if len(lines) < 2:
                continue
            sender = lines[0].strip().strip("[]")
            if sender != agent_id:
                continue
            match = report_pattern.search(msg)
            if match:
                report = match.group(1).strip()
                if longest_report is None or len(report) > len(longest_report):
                    longest_report = report

    return longest_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract proposal from experiment")
    parser.add_argument("output_dir", type=Path, help="Path to experiment directory")
    args = parser.parse_args()

    proposal = extract_proposal(args.output_dir)
    if proposal:
        print(proposal)
    else:
        print("No proposal found.", file=__import__("sys").stderr)
        raise SystemExit(1)
