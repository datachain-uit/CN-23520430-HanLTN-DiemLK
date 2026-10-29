"""Kiểm tra khóa lạnh có thuộc 20% khóa xuất hiện muộn nhất hay không."""

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
    parser.add_argument(
        "--lists-dir",
        type=Path,
        default=None,
        help="Thư mục chứa cold_items_val.json và cold_items_test.json.",
    )
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Trả mã lỗi nếu tập khóa lạnh hoặc số lượng danh sách không khớp.",
    )
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction phải nằm trong khoảng mở từ 0 đến 1.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction phải nằm trong khoảng mở từ 0 đến 1.")

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = args.lists_dir or Path("data/splits/cold_scenario")
    if not split_dir.is_absolute():
        split_dir = data_dir / split_dir
    split_dir = split_dir.expanduser().resolve()
    interactions = pd.read_parquet(data_dir / "data/full_interactions/interactions_full.parquet")
    interactions["course_id"] = interactions["course_id"].astype(str)
    first_seen = interactions.groupby("course_id", sort=False)["enroll_time"].min()
    course_count = len(first_seen)
    expected_count = int(course_count * args.fraction)
    ordered = sorted(first_seen.items(), key=lambda row: (row[1], row[0]))
    expected = (
        {course_id for course_id, _ in ordered[-expected_count:]}
        if expected_count
        else set()
    )

    with (split_dir / "cold_items_val.json").open(encoding="utf-8") as f:
        val_ids = _ids(json.load(f))
    with (split_dir / "cold_items_test.json").open(encoding="utf-8") as f:
        test_ids = _ids(json.load(f))
    actual = val_ids | test_ids
    overlap = expected & actual
    missing = expected - actual
    extra = actual - expected
    expected_val_count = int(expected_count * args.validation_fraction)
    expected_test_count = expected_count - expected_val_count
    counts_match = len(val_ids) == expected_val_count and len(test_ids) == expected_test_count
    passed = not (missing or extra) and not (val_ids & test_ids) and counts_match
    lines = [
        "Kiểm tra quy tắc chọn khóa lạnh (Phụ lục B.3)",
        f"Số khóa học duy nhất: {course_count}",
        f"Tỷ lệ khóa xuất hiện muộn được chọn: {args.fraction:.0%} ({expected_count} khóa)",
        f"Số khóa validation/test đã lưu: {len(val_ids)}/{len(test_ids)} (hợp={len(actual)})",
        f"Số khóa validation/test kỳ vọng: {expected_val_count}/{expected_test_count}",
        f"Số lượng validation/test khớp: {counts_match}",
        f"Số khóa giao giữa tập kỳ vọng và tập đã lưu: {len(overlap)} ({len(overlap) / max(1, expected_count):.2%})",
        f"Số khóa còn thiếu: {len(missing)}",
        f"Số khóa không thuộc tập kỳ vọng: {len(extra)}",
        f"Hai tập validation/test không giao nhau: {not bool(val_ids & test_ids)}",
        f"Tập hợp khóa lạnh khớp chính xác: {passed}",
    ]
    print("\n".join(lines))
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
