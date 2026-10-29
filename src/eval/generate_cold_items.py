"""Sinh danh sách khóa lạnh validation/test theo quy tắc thời gian đã định."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.fraction < 1:
        parser.error("--fraction phải nằm trong khoảng mở từ 0 đến 1.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction phải nằm trong khoảng mở từ 0 đến 1.")

    DATA_DIR = args.data_dir.expanduser().resolve()
    interactions = pd.read_parquet(
        DATA_DIR / "data/full_interactions/interactions_full.parquet"
    )
    interactions["course_id"] = interactions["course_id"].astype(str)
    first_seen = interactions.groupby("course_id", sort=False)["enroll_time"].min()
    ordered = sorted(first_seen.items(), key=lambda row: (row[1], row[0]))
    cold_count = int(len(ordered) * args.fraction)
    cold_ids = [course_id for course_id, _ in ordered[-cold_count:]] if cold_count else []

    # Xáo trộn danh sách đã sắp thứ tự bằng seed cố định để chia val/test ổn định.
    rng = np.random.default_rng(args.seed)
    cold_ids = [cold_ids[index] for index in rng.permutation(len(cold_ids))]
    val_count = int(len(cold_ids) * args.validation_fraction)
    generated = {
        "cold_items_val.json": cold_ids[:val_count],
        "cold_items_test.json": cold_ids[val_count:],
    }

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("data/splits/cold_scenario/generated_cold_items")
    if not output_dir.is_absolute():
        output_dir = DATA_DIR / output_dir
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / name for name in generated]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        parser.error(f"Tệp đã tồn tại ({names}); dùng --overwrite nếu muốn ghi đè.")

    for path, ids in zip(output_paths, generated.values()):
        path.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Số khóa trong dữ liệu: {len(first_seen)}")
    print(f"Số khóa lạnh được chọn: {len(cold_ids)} ({args.fraction:.0%})")
    print(f"Số khóa validation/test: {len(generated['cold_items_val.json'])}/{len(generated['cold_items_test.json'])}")
    print(f"Seed chia tập: {args.seed}")
    print(f"Đã lưu danh sách mới tại: {output_dir}")
    print("Danh sách mới được lưu riêng; các tệp split dùng trong bài không bị thay đổi.")


if __name__ == "__main__":
    main()
