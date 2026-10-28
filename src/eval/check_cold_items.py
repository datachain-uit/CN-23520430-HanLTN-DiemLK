"""Check that cold items are the 20% latest courses by first enrollment time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ids(payload: object) -> set[str]:
    if isinstance(payload, dict):
        payload = payload.get("cold_items", payload.get("course_id", payload))
    return {str(item) for item in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if the saved set is not an exact match.")
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be strictly between 0 and 1")

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "data/splits/cold_scenario"
    interactions = pd.read_parquet(data_dir / "data/full_interactions/interactions_full.parquet")
    interactions["course_id"] = interactions["course_id"].astype(str)
    first_seen = interactions.groupby("course_id", sort=False)["enroll_time"].min()
    course_count = len(first_seen)
    expected_count = int(course_count * args.fraction)
    ordered = sorted(first_seen.items(), key=lambda row: (-row[1], row[0]))
    expected = {course_id for course_id, _ in ordered[:expected_count]}

    with (split_dir / "cold_items_val.json").open(encoding="utf-8") as f:
        val_ids = _ids(json.load(f))
    with (split_dir / "cold_items_test.json").open(encoding="utf-8") as f:
        test_ids = _ids(json.load(f))
    actual = val_ids | test_ids
    overlap = expected & actual
    missing = expected - actual
    extra = actual - expected
    passed = not (missing or extra) and not (val_ids & test_ids)
    lines = [
        "Cold-item selection check (Appendix B.3)",
        f"Unique courses: {course_count}",
        f"Requested latest fraction: {args.fraction:.0%} ({expected_count} items)",
        f"Saved validation/test counts: {len(val_ids)}/{len(test_ids)} (union={len(actual)})",
        f"Expected/saved overlap: {len(overlap)} items ({len(overlap) / max(1, expected_count):.2%})",
        f"Missing from saved lists: {len(missing)}",
        f"Unexpected in saved lists: {len(extra)}",
        f"Val/test disjoint: {not bool(val_ids & test_ids)}",
        f"Exact match: {passed}",
    ]
    print("\n".join(lines))
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
