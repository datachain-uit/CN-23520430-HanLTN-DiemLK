
import os
import json
import time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer

from text_utils import (
    get_clean_raw_course_text,
    clean_text_field,
    field_to_text,
    build_profile_text,
    compute_and_save_dup_groups,
    PREREQ_HINT_IN_ABOUT_REGEX
)


def get_sbert_dim(encoder):
    if hasattr(encoder, "get_embedding_dimension"):
        return encoder.get_embedding_dimension()
    return encoder.get_sentence_embedding_dimension()


def encode_all_raw_course_texts_only(df, encoder_name, output_dir, batch_size=32):
    gid_map, dup_summary = compute_and_save_dup_groups(df, output_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    course_ids = [str(row.get("course_id", row.get("id", f"course_{i}"))) for i, (_, row) in enumerate(df.iterrows())]
    raw_texts = [get_clean_raw_course_text(row) for _, row in df.iterrows()]

    raw_token_lens = [len(ids) for ids in encoder.tokenizer(raw_texts, add_special_tokens=True)["input_ids"]]
    raw_truncated = int(sum(l > encoder.max_seq_length for l in raw_token_lens))

    t0 = time.perf_counter()
    raw_emb = encoder.encode(
        raw_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=True
    ).astype(np.float32)
    raw_encode_sec = time.perf_counter() - t0

    emb_dim = get_sbert_dim(encoder)
    assert raw_emb.shape == (len(df), emb_dim), (
        f"Lỗi lệch kích thước raw embedding: {raw_emb.shape} != {(len(df), emb_dim)}"
    )

    raw_emb_path = os.path.join(output_dir, "raw_course_text_embeddings.npy")
    idx_path = os.path.join(output_dir, "raw_embeddings_course_index.json")

    np.save(raw_emb_path, raw_emb)
    meta = {
        "encoder_name": encoder_name,
        "num_courses": len(df),
        "embedding_dim": int(emb_dim),
        "max_seq_length": int(encoder.max_seq_length),
        "npy_path": raw_emb_path,
        "encode_time_sec": round(raw_encode_sec, 4),
        "avg_token_len": round(float(np.mean(raw_token_lens)), 2),
        "max_token_len": int(max(raw_token_lens)),
        "num_truncated_texts": raw_truncated,
        "dup_group_summary": dup_summary,
        "course_index": list(range(len(df))),
        "course_id": course_ids,
        "dup_group_id": [gid_map[cid] for cid in course_ids]
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu ma trận raw_course_text_embeddings ({raw_emb.shape}) tại: {raw_emb_path}")
    print(f"Đã lưu tệp thứ tự ánh xạ course_id và dup_group_id tại: {idx_path}")
    return meta


def encode_both_texts_with_sbert(results, encoder_name, output_dir, batch_size=32, dup_summary=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    raw_texts = [r["raw_course_text"] for r in results]
    llm_texts = [r["profile_text"] for r in results]
    # Biến thể Ablation bỏ 2 chiều tiên quyết và tiếp nối
    llm_3dim_texts = [
        build_profile_text(r.get("profile"), fallback_text=r["raw_course_text"], include_prereq_and_followup=False)
        for r in results
    ]
    name_texts = [r.get("course_name", "unknown") for r in results]

    raw_token_lens = [len(ids) for ids in encoder.tokenizer(raw_texts, add_special_tokens=True)["input_ids"]]
    llm_token_lens = [len(ids) for ids in encoder.tokenizer(llm_texts, add_special_tokens=True)["input_ids"]]

    raw_truncated = int(sum(l > encoder.max_seq_length for l in raw_token_lens))
    llm_truncated = int(sum(l > encoder.max_seq_length for l in llm_token_lens))

    t0 = time.perf_counter()
    raw_emb = encoder.encode(
        raw_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False
    ).astype(np.float32)
    raw_encode_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    llm_emb = encoder.encode(
        llm_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False
    ).astype(np.float32)
    llm_encode_sec = time.perf_counter() - t1

    llm_3dim_emb = encoder.encode(
        llm_3dim_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False
    ).astype(np.float32)

    name_emb = encoder.encode(
        name_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False
    ).astype(np.float32)

    emb_dim = get_sbert_dim(encoder)
    assert raw_emb.shape == llm_emb.shape == llm_3dim_emb.shape == (len(results), emb_dim), (
        f"Lỗi lệch kích thước embedding: raw={raw_emb.shape}, llm={llm_emb.shape}"
    )

    raw_norm = raw_emb / np.clip(np.linalg.norm(raw_emb, axis=1, keepdims=True), 1e-8, None)
    llm_norm = llm_emb / np.clip(np.linalg.norm(llm_emb, axis=1, keepdims=True), 1e-8, None)
    name_norm = name_emb / np.clip(np.linalg.norm(name_emb, axis=1, keepdims=True), 1e-8, None)

    cos_sims = np.sum(raw_norm * llm_norm, axis=1)
    name_llm_cos_sims = np.sum(name_norm * llm_norm, axis=1)
    for r, c_raw, c_name in zip(results, cos_sims, name_llm_cos_sims):
        r["cosine_raw_vs_llm"] = round(float(c_raw), 4)
        r["cosine_name_vs_llm"] = round(float(c_name), 4)

    k_lowest = max(1, int(np.ceil(0.01 * len(results))))
    sorted_indices = np.argsort(name_llm_cos_sims)[:k_lowest]
    lowest_1pct_courses = [
        {
            "course_id": results[i]["course_id"],
            "course_name": results[i]["course_name"],
            "dup_group_id": results[i].get("dup_group_id", results[i]["course_id"]),
            "cosine_name_vs_llm": round(float(name_llm_cos_sims[i]), 4),
            "cosine_raw_vs_llm": round(float(cos_sims[i]), 4),
            "overview": results[i]["profile"].get("overview", "") if results[i].get("profile") else ""
        }
        for i in sorted_indices
    ]
    lowest_1pct_path = os.path.join(output_dir, "lowest_1pct_name_profile_cosine.json")
    with open(lowest_1pct_path, "w", encoding="utf-8") as f:
        json.dump(lowest_1pct_courses, f, ensure_ascii=False, indent=2)

    raw_emb_path = os.path.join(output_dir, "raw_course_text_embeddings.npy")
    llm_emb_path = os.path.join(output_dir, "llm_profile_embeddings.npy")
    llm_3dim_emb_path = os.path.join(output_dir, "llm_profile_no_prereq_followup_embeddings.npy")
    idx_path = os.path.join(output_dir, "embeddings_course_index.json")

    np.save(raw_emb_path, raw_emb)
    np.save(llm_emb_path, llm_emb)
    np.save(llm_3dim_emb_path, llm_3dim_emb)

    index_meta = {
        "encoder_name": encoder_name,
        "num_courses": len(results),
        "embedding_dim": int(llm_emb.shape[1]),
        "max_seq_length": int(encoder.max_seq_length),
        "raw_course_text_stats": {
            "npy_path": raw_emb_path,
            "encode_time_sec": round(raw_encode_sec, 4),
            "avg_token_len": round(float(np.mean(raw_token_lens)), 2),
            "max_token_len": int(max(raw_token_lens)),
            "num_truncated_texts": raw_truncated
        },
        "llm_profile_text_stats": {
            "npy_path": llm_emb_path,
            "ablation_no_prereq_followup_npy_path": llm_3dim_emb_path,
            "encode_time_sec": round(llm_encode_sec, 4),
            "avg_token_len": round(float(np.mean(llm_token_lens)), 2),
            "max_token_len": int(max(llm_token_lens)),
            "num_truncated_texts": llm_truncated
        },
        "raw_vs_llm_cosine_similarity": {
            "mean": round(float(np.mean(cos_sims)), 4),
            "std": round(float(np.std(cos_sims)), 4),
            "min": round(float(np.min(cos_sims)), 4),
            "max": round(float(np.max(cos_sims)), 4)
        },
        "name_vs_llm_cosine_similarity": {
            "mean": round(float(np.mean(name_llm_cos_sims)), 4),
            "std": round(float(np.std(name_llm_cos_sims)), 4),
            "min": round(float(np.min(name_llm_cos_sims)), 4),
            "max": round(float(np.max(name_llm_cos_sims)), 4),
            "lowest_1pct_count": k_lowest,
            "lowest_1pct_path": lowest_1pct_path
        },
        "dup_group_summary": dup_summary,
        "course_index": [r["course_index"] for r in results],
        "course_id": [r["course_id"] for r in results],
        "dup_group_id": [r.get("dup_group_id", r["course_id"]) for r in results]
    }

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    return index_meta


# Đánh giá lượt ẩn tiên quyết với 2 mức nền mạnh hơn
def evaluate_masked_prereq_predictions(results, encoder_name, output_dir, df_courses=None, seed=2026):
    R = [
        r for r in results
        if r.get("profile") and field_to_text(r["profile"].get("prerequisite_knowledge"))
        and r.get("ground_truth_prereq")
    ]
    if not R:
        print("Không có bản ghi hợp lệ để đánh giá lượt ẩn tiên quyết.")
        return {}

    about_by_cid = {}
    if df_courses is not None:
        for _, row in df_courses.iterrows():
            cid = str(row.get("course_id", row.get("id", "")))
            about_by_cid[cid] = clean_text_field(row.get("course_about"), default_val="")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(encoder_name, device=device)
    enc.eval()

    E = lambda xs: enc.encode(xs, normalize_embeddings=True).astype(np.float32)
    gt = E([r["ground_truth_prereq"] for r in R])
    inf = E([field_to_text(r["profile"]["prerequisite_knowledge"]) for r in R])
    nm = E([r["course_name"] for r in R])
    raw = E([r["raw_course_text"] for r in R])  # tên + trường + GV + mô tả, không có trường tiên quyết
    cos = lambda a, b: (a * b).sum(1)

    paired_cos = cos(inf, gt)
    name_base_cos = cos(nm, gt)
    raw_base_cos = cos(raw, gt)

    rng = np.random.default_rng(seed)
    n = len(R)
    rand_indices = np.arange(n)
    if n > 1:
        for i in range(n):
            choices = [j for j in range(n) if j != i]
            rand_indices[i] = rng.choice(choices)
    random_base_cos = cos(inf, gt[rand_indices])

    clean_subset_mask = []
    for idx_r, r in enumerate(R):
        cid = str(r["course_id"])
        about_txt = about_by_cid.get(cid, r.get("raw_course_text", ""))
        gt_str = str(r["ground_truth_prereq"]).strip()
        has_regex_hint = bool(PREREQ_HINT_IN_ABOUT_REGEX.search(about_txt))
        has_verbatim_gt = bool(gt_str and len(gt_str) >= 4 and gt_str.lower() in about_txt.lower())
        has_leak = has_regex_hint or has_verbatim_gt
        clean_subset_mask.append(not has_leak)

        r["cosine_inferred_vs_gt"] = round(float(paired_cos[idx_r]), 4)
        r["cosine_name_vs_gt"] = round(float(name_base_cos[idx_r]), 4)
        r["cosine_raw_text_vs_gt"] = round(float(raw_base_cos[idx_r]), 4)
        r["cosine_inferred_vs_random_gt"] = round(float(random_base_cos[idx_r]), 4)
        r["about_contains_prereq_hint_or_verbatim"] = has_leak

    clean_subset_mask = np.array(clean_subset_mask, dtype=bool)

    def _compute_subset_stats(mask_arr, subset_label):
        sub_inf = paired_cos[mask_arr]
        sub_nm = name_base_cos[mask_arr]
        sub_raw = raw_base_cos[mask_arr]
        sub_rand = random_base_cos[mask_arr]
        m_len = int(mask_arr.sum())
        if m_len == 0:
            return {"num_courses": 0}

        def _wilcox(x, y):
            if m_len > 1 and not np.allclose(x, y):
                st, pv = wilcoxon(x, y, alternative="greater")
                return {"statistic": round(float(st), 4), "p_value_one_sided_greater": float(pv)}
            return {"statistic": 0.0, "p_value_one_sided_greater": 1.0}

        def _stat_dict(arr):
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4)
            }

        res = {
            "num_courses": m_len,
            "inferred_vs_gt_cosine": _stat_dict(sub_inf),
            "baselines": {
                "course_name_vs_gt": {
                    **_stat_dict(sub_nm),
                    "wilcoxon_inferred_greater": _wilcox(sub_inf, sub_nm)
                },
                "raw_course_text_vs_gt": {
                    **_stat_dict(sub_raw),
                    "wilcoxon_inferred_greater": _wilcox(sub_inf, sub_raw)
                },
                "random_paired_gt": {
                    **_stat_dict(sub_rand),
                    "wilcoxon_inferred_greater": _wilcox(sub_inf, sub_rand)
                }
            }
        }
        print(f"\nĐÁNH GIÁ LƯỢT ẨN TIÊN QUYẾT [{subset_label}: N = {m_len}]")
        for tag, base_arr in {
            "tên khóa": sub_nm,
            "văn bản gốc": sub_raw,
            "ghép ngẫu nhiên": sub_rand
        }.items():
            pv = wilcoxon(sub_inf, base_arr, alternative="greater").pvalue if (m_len > 1 and not np.allclose(sub_inf, base_arr)) else 1.0
            print(
                f"  + Mức nền [{tag:15s}]: cos(inf, gt) = {sub_inf.mean():.4f} ± {sub_inf.std():.4f} "
                f"vs. cos(base, gt) = {base_arr.mean():.4f} ± {base_arr.std():.4f} | Wilcoxon p = {pv:.4e}"
            )
        return res

    full_stats = _compute_subset_stats(np.ones(n, dtype=bool), "Toàn bộ 200 khóa")
    clean_stats = _compute_subset_stats(clean_subset_mask, "Tập con đã loại mô tả chứa gợi ý tiên quyết")

    eval_summary = {
        "num_evaluated_courses": n,
        "num_courses_with_prereq_hint_in_about": int((~clean_subset_mask).sum()),
        "num_clean_subset_courses": int(clean_subset_mask.sum()),
        "seed": seed,
        "encoder_name": encoder_name,
        "full_200_evaluation": full_stats,
        "clean_subset_no_prereq_in_about_evaluation": clean_stats,
        "paired_cosine": full_stats["inferred_vs_gt_cosine"],
        "course_name_baseline_cosine": full_stats["baselines"]["course_name_vs_gt"],
        "raw_course_text_baseline_cosine": full_stats["baselines"]["raw_course_text_vs_gt"],
        "random_baseline_cosine": full_stats["baselines"]["random_paired_gt"],
        "wilcoxon_signed_rank_test": full_stats["baselines"]["random_paired_gt"]["wilcoxon_inferred_greater"]
    }

    eval_report_path = os.path.join(output_dir, "masked_prereq_200_eval_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)

    print(f"-> Đã lưu báo cáo đánh giá lượt ẩn tiên quyết đầy đủ tại: {eval_report_path}")
    return eval_summary
