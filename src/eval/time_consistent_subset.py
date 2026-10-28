"""Summarize the cold-test subset whose items first appeared after the time cutoff."""

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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quantile", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.quantile < 1:
        parser.error("--quantile must be strictly between 0 and 1")

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "data/splits/cold_scenario"
    interactions = pd.read_parquet(data_dir / "data/full_interactions/interactions_full.parquet")
    test = pd.read_parquet(split_dir / "test_cold.parquet")
    with (split_dir / "cold_items_test.json").open(encoding="utf-8") as f:
        cold_test = _ids(json.load(f))

    interactions["course_id"] = interactions["course_id"].astype(str)
    test["course_id"] = test["course_id"].astype(str)
    cutoff = interactions["enroll_time"].quantile(args.quantile)
    first_seen = interactions.groupby("course_id", sort=False)["enroll_time"].min()
    first_seen_after_cutoff = set(first_seen[first_seen >= cutoff].index)
    eligible_items = cold_test & first_seen_after_cutoff
    selected = test[test["course_id"].isin(eligible_items)]
    users = selected["user_id"].nunique()
    test_users = test[test["course_id"].isin(cold_test)]["user_id"].nunique()
    interaction_share = len(selected) / max(1, int(test["course_id"].isin(cold_test).sum()))
    item_share = len(eligible_items) / max(1, len(cold_test))
    cold_test_rows = int(test["course_id"].isin(cold_test).sum())

    text = (
        "Time-consistent cold-test subset\n"
        f"Whole-dataset interaction-time quantile: {args.quantile:.2f}\n"
        f"Cutoff enroll_time: {cutoff}\n"
        f"Cold-test items first seen at/after cutoff: {len(eligible_items)}/{len(cold_test)} "
        f"({item_share:.2%})\n"
        f"Retained cold-test interactions: {len(selected)}/{cold_test_rows} "
        f"({interaction_share:.2%})\n"
        f"Users with a retained target: {users}/{test_users}\n"
        "Subset rule: retain a cold-test target only when the course's first "
        "recorded interaction in the full dataset is at/after the stated cutoff.\n"
    )
    output = args.output or data_dir / "results/time_consistent_subset.txt"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
