"""Tính kỳ vọng NDCG@20 macro chính xác cho thứ hạng ngẫu nhiên đều."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ids(payload: object) -> set[str]:
    if isinstance(payload, dict):
        payload = payload.get("cold_items", payload.get("course_id", payload))
    return {str(item) for item in payload}


def expected_ndcg(targets: dict[str, set[str]], candidate_count: int, k: int = 20) -> float:
    if candidate_count <= 0:
        raise ValueError("The candidate set must contain at least one item.")
    discounts = 1.0 / np.log2(np.arange(2, min(k, candidate_count) + 2))
    values = []
    for relevant in targets.values():
        n_relevant = len(relevant)
        if not n_relevant:
            continue
        ideal = discounts[: min(n_relevant, len(discounts))].sum()
        expected_dcg = (n_relevant / candidate_count) * discounts.sum()
        values.append(float(expected_dcg / ideal))
    if not values:
        raise ValueError("No users have a relevant item in the candidate set.")
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--catalog-count", type=int, default=None,
        help="Số khóa trong toàn bộ danh mục; mặc định lấy độ dài item_index.json đã lưu.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "data/splits/cold_scenario"
    with (split_dir / "cold_items_test.json").open(encoding="utf-8") as f:
        candidates = _ids(json.load(f))
    test = pd.read_parquet(split_dir / "test_cold.parquet")
    test["user_id"] = test["user_id"].astype(str)
    test["course_id"] = test["course_id"].astype(str)
    targets = {
        str(user): set(group["course_id"]) & candidates
        for user, group in test.groupby("user_id", sort=True)
    }
    targets = {user: items for user, items in targets.items() if items}
    value_cold = expected_ndcg(targets, len(candidates), args.k)
    catalog_count = args.catalog_count
    if catalog_count is None:
        index_path = data_dir / "data/embeddings/01_lightgcn_backbone_cold/item_index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as f:
                catalog_count = len(json.load(f))
    value_catalog = expected_ndcg(targets, catalog_count, args.k) if catalog_count else None
    relevant_counts = np.array([len(items) for items in targets.values()])
    text = (
        "Analytic uniform-random ranking baseline\n"
        f"Cold-only candidate set: {len(candidates)} items\n"
        f"Full-catalog candidate count: {catalog_count or 'not available'}\n"
        f"Users with at least one relevant test item: {len(targets)}\n"
        f"Relevant items per user: mean={relevant_counts.mean():.6f}, "
        f"median={np.median(relevant_counts):.0f}, max={relevant_counts.max()}\n"
        f"Expected macro NDCG@{args.k} (cold-only): {value_cold:.8f}\n"
        + (f"Expected macro NDCG@{args.k} (full catalog): {value_catalog:.8f}\n" if value_catalog is not None else "")
        + "Formula: for each rank r, P(relevant at r)=R/N; expected DCG is "
        + "normalized by that user's ideal DCG, then averaged over users.\n"
    )
    output = args.output or data_dir / "results/random_ndcg_analytic.txt"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
