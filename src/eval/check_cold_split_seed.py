"""Kiểm tra quy tắc xáo trộn dùng khi chia danh sách khóa lạnh."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [str(course_id) for course_id in json.load(f)]


def shuffles(base: list[str], seed: int):
    shuffled = list(base)
    random.Random(seed).shuffle(shuffled)
    yield "random.Random(seed).shuffle", shuffled

    shuffled = list(base)
    np.random.RandomState(seed).shuffle(shuffled)
    yield "np.random.RandomState(seed).shuffle", shuffled

    order = np.random.default_rng(seed).permutation(len(base))
    yield "np.random.default_rng(seed).permutation", [base[index] for index in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "data/splits/cold_scenario"
    interactions = pd.read_parquet(data_dir / "data/full_interactions/interactions_full.parquet")
    interactions["course_id"] = interactions["course_id"].astype(str)
    first_seen = interactions.groupby("course_id", sort=False)["enroll_time"].min()
    ordered = sorted(first_seen.items(), key=lambda row: (row[1], row[0]))
    cold_count = int(len(ordered) * 0.20)
    by_time = [course_id for course_id, _ in ordered[-cold_count:]]
    bases = {"time order": by_time, "course_id order": sorted(by_time)}

    val = load_ids(split_dir / "cold_items_val.json")
    test = load_ids(split_dir / "cold_items_test.json")
    same_pool = set(by_time) == set(val) | set(test)
    print("Tập khóa lạnh khớp quy tắc 20% muộn nhất:", same_pool)

    found = False
    for seed in args.seeds:
        for base_name, base in bases.items():
            for shuffle_name, permutation in shuffles(base, seed):
                val_count = len(permutation) // 3
                same_val_set = set(permutation[:val_count]) == set(val)
                same_order = permutation[:val_count] == val and permutation[val_count:] == test
                found |= same_val_set
                print(
                    f"seed={seed:<5d} {base_name:16s} {shuffle_name:40s} "
                    f"val (tập): {same_val_set}  cả thứ tự: {same_order}"
                )
    if found:
        print("KẾT LUẬN: tái tạo được cách chia val/test")
    else:
        print("KẾT LUẬN: chưa tái tạo được; sửa câu ở Mục 4.1 (không nêu seed 42)")


if __name__ == "__main__":
    main()
