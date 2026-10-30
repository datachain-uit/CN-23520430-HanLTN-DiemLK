"""Thống kê lịch sử học viên không cắt theo thời gian (số 76,7% ở Mục 6.5 của bài)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "data/splits/cold_scenario"
    train = pd.read_parquet(split_dir / "train_cold.parquet")
    val = pd.read_parquet(split_dir / "val_cold.parquet")
    test = pd.read_parquet(split_dir / "test_cold.parquet")

    later_train = train.groupby("user_id")["enroll_time"].max()
    has_later_train = test["user_id"].map(later_train) > test["enroll_time"]

    last_history = pd.concat([train, val]).groupby("user_id")["enroll_time"].max()
    first_test = test.groupby("user_id")["enroll_time"].min()
    time_consistent = last_history.reindex(first_test.index) < first_test
    train_counts = train.groupby("user_id").size().reindex(first_test.index)

    text = (
        "Time-agnostic learner histories (cold-item split)\n"
        "Cold-test interactions with a later training interaction of the same learner: "
        f"{int(has_later_train.sum())}/{len(test)} ({has_later_train.mean():.1%})\n"
        "Learners whose whole train+val history precedes their first cold-test enrollment: "
        f"{int(time_consistent.sum())}/{len(first_test)} ({time_consistent.mean():.1%})\n"
        f"Median training interactions: {train_counts[time_consistent].median():.0f} "
        f"(these learners) vs {train_counts[~time_consistent].median():.0f} (other learners)\n"
    )
    output = args.output or data_dir / "results/time_agnostic_history.txt"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
