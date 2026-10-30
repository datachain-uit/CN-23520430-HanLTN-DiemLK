
import os
import json
import gc
import argparse
from pathlib import Path
import pandas as pd
import torch
import transformers
import sentence_transformers

from text_utils import (
    norm_prereq,
    align_dataframe_order,
    compute_and_save_dup_groups,
    check_name_mentions,
    scrub_name_mentions,
    build_profile_text
)
from sbert_encoder import (
    encode_all_raw_course_texts_only,
    encode_both_texts_with_sbert,
    evaluate_masked_prereq_predictions
)
from llm_generator import (
    get_model_revision,
    load_prompt_template,
    generate_one_course
)


def merge_shards_and_encode(df, args):
    df_aligned = align_dataframe_order(df, args.item2idx_path)
    gid_map, dup_summary = compute_and_save_dup_groups(df_aligned, args.output_dir)
    row_by_id = {str(row["course_id"]): row for _, row in df_aligned.iterrows()}

    records_by_id = {}
    for s in range(args.num_shards):
        shard_file = os.path.join(args.output_dir, f"profiles_shard{s}.jsonl")
        if not os.path.exists(shard_file):
            raise FileNotFoundError(f"Chưa tìm thấy file shard {s}: {shard_file}")
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    records_by_id[str(obj["course_id"])] = obj

    expected_ids = [str(cid) for cid in df_aligned["course_id"].tolist()]
    missing_ids = [cid for cid in expected_ids if cid not in records_by_id]
    assert len(missing_ids) == 0, f"Lỗi: Còn thiếu {len(missing_ids)} khóa học chưa sinh xong"
    assert len(records_by_id) == len(expected_ids), (
        f"Số lượng course_id không khớp: {len(records_by_id)} vs {len(expected_ids)}"
    )

    model_rev = get_model_revision(args.model_name, revision=args.model_revision)
    merged_results = []
    rechecked_mention_count = 0
    scrubbed_in_merge_count = 0

    for idx, cid in enumerate(expected_ids):
        rec = records_by_id[cid]
        rec["course_index"] = idx
        rec["dup_group_id"] = gid_map[cid]
        rec["model_revision"] = model_rev

        row_meta = row_by_id[cid]
        if rec.get("profile"):
            mentions = check_name_mentions(rec["profile"], row_meta)
            if mentions:
                scrubbed_in_merge_count += 1
                rec["profile"] = scrub_name_mentions(rec["profile"], row_meta)
                rec["profile_text"] = build_profile_text(rec["profile"], fallback_text=rec["raw_course_text"])
                mentions = check_name_mentions(rec["profile"], row_meta)
            rec["mentions_university_or_instructor"] = bool(mentions)
            rec["mentioned_entities"] = mentions
            if mentions:
                rechecked_mention_count += 1

        merged_results.append(rec)

    assert not any("(inferred)" in r["profile_text"].lower() for r in merged_results), (
        "Lỗi: Vẫn còn chuỗi '(inferred)' trong profile_text đưa vào SBERT!"
    )

    encode_info = encode_both_texts_with_sbert(
        merged_results, args.encoder_name, args.output_dir,
        batch_size=args.encode_batch_size, dup_summary=dup_summary
    )
    encode_info["model_name"] = args.model_name
    encode_info["model_revision"] = model_rev
    encode_info["rechecked_name_mentions_after_abbreviation_rules"] = {
        "scrubbed_during_merge_count": scrubbed_in_merge_count,
        "remaining_mention_count": rechecked_mention_count
    }

    full_json_path = os.path.join(args.output_dir, f"full_{len(merged_results)}_profiles.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(
        f"-> [Kiểm tra nhắc tên trường (gồm viết tắt/rút gọn)] Đã hậu xử lý: {scrubbed_in_merge_count} hồ sơ | "
        f"Còn nhắc tên: {rechecked_mention_count}/{len(merged_results)}"
    )
    print(f"-> [Model Revision] {args.model_name} SHA = {model_rev}")
    print(f"Đã gộp thành công {len(merged_results)} hồ sơ không trùng lặp tại: {full_json_path}")
    return encode_info


def main():
    parser = argparse.ArgumentParser(description="Sinh hồ sơ sư phạm 5 chiều bằng LLM và mã hóa văn bản SBERT")
    repo_root = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1])).expanduser().resolve()
    parser.add_argument("--data_path", type=str, default=str(repo_root / "data/course_llm/course_data.parquet"))
    parser.add_argument("--prompt_path", type=str, default=str(repo_root / "src/prompts/prompt_template.txt"))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--model_revision", type=str,
        default="a09a35458c702b33eeacc393d103063234e8bc28",
        help="Commit revision của mô hình Qwen dùng để sinh hồ sơ.",
    )
    parser.add_argument("--num_courses", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--single_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_inferred_hint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default=str(repo_root / "data/course_llm"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--item2idx_path", type=str, default=None)
    parser.add_argument("--encode_raw_only", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--masked_prereq_eval", action="store_true")
    parser.add_argument("--re_eval_masked_jsonl", type=str, default=None)
    parser.add_argument("--export_dup_groups_only", action="store_true")
    parser.add_argument("--eval_sample_size", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Đang đọc dữ liệu khóa học từ: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    df = df.loc[:, ~df.columns.duplicated()]
    total_dataset_courses = len(df)

    all_prereq_norm = [norm_prereq(row.get("course_prerequisites")) for _, row in df.iterrows()]
    stated_prereq_mask = [p not in {"none", "not stated"} for p in all_prereq_norm]
    num_stated_prereq = int(sum(stated_prereq_mask))
    num_none_prereq = int(sum(p == "none" for p in all_prereq_norm))
    num_unstated_prereq = int(sum(p == "not stated" for p in all_prereq_norm))
    print(f"-> Tổng số khóa học trong bộ dữ liệu : {total_dataset_courses}")
    print(
        f"-> Thống kê tiên quyết toàn cục      : Có ghi rõ = {num_stated_prereq} | "
        f"'none' = {num_none_prereq} | 'not stated' = {num_unstated_prereq}"
    )

    if args.export_dup_groups_only:
        df_ordered = align_dataframe_order(df, args.item2idx_path)
        compute_and_save_dup_groups(df_ordered, args.output_dir)
        return

    if args.re_eval_masked_jsonl:
        if not os.path.exists(args.re_eval_masked_jsonl):
            raise FileNotFoundError(f"Không tìm thấy tệp: {args.re_eval_masked_jsonl}")
        loaded_records = []
        with open(args.re_eval_masked_jsonl, "r", encoding="utf-8") as f:
            if args.re_eval_masked_jsonl.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        loaded_records.append(json.loads(line))
            else:
                loaded_records = json.load(f)
        evaluate_masked_prereq_predictions(
            loaded_records, args.encoder_name, args.output_dir, df_courses=df, seed=args.eval_seed
        )
        return

    if args.encode_raw_only:
        df_ordered = align_dataframe_order(df, args.item2idx_path)
        encode_all_raw_course_texts_only(
            df_ordered, args.encoder_name, args.output_dir, batch_size=args.encode_batch_size
        )
        return

    if args.merge_shards:
        merge_shards_and_encode(df, args)
        return

    # Chỉ import bitsandbytes và AutoModelForCausalLM khi thực sự cần chạy sinh LLM trên GPU
    import bitsandbytes as bnb
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    df_ordered_all = align_dataframe_order(df, args.item2idx_path)
    gid_map, dup_summary = compute_and_save_dup_groups(df_ordered_all, args.output_dir)

    if args.masked_prereq_eval:
        df_stated = df.loc[stated_prereq_mask].copy()
        df_stated = align_dataframe_order(df_stated, None)
        sample_n = min(args.eval_sample_size, len(df_stated))
        df_run = df_stated.sample(n=sample_n, random_state=args.eval_seed).reset_index(drop=True)
        jsonl_path = os.path.join(args.output_dir, f"masked_prereq_{sample_n}_profiles.jsonl")
    else:
        if args.num_shards > 1 or args.num_courses <= 0 or args.num_courses >= total_dataset_courses:
            df_run = df_ordered_all.iloc[args.shard::args.num_shards].copy()
            jsonl_path = os.path.join(args.output_dir, f"profiles_shard{args.shard}.jsonl")
        else:
            df_run = df_ordered_all.head(args.num_courses).copy()
            jsonl_path = os.path.join(args.output_dir, f"pilot_{len(df_run)}_profiles.jsonl")

    done_records = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec_loaded = json.loads(line)
                        done_records[str(rec_loaded["course_id"])] = rec_loaded
                    except Exception:
                        pass

    sys_prompt, user_template = load_prompt_template(args.prompt_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, revision=args.model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map_cfg = {"": 0} if (args.single_gpu and torch.cuda.is_available()) else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config,
        device_map=device_map_cfg, trust_remote_code=True, revision=args.model_revision
    )
    model.eval()
    model_revision = get_model_revision(
        args.model_name, model=model, revision=args.model_revision
    )

    if torch.cuda.is_available():
        for gi in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gi)

    results, total_input_tokens, total_output_tokens = [], 0, 0
    total_gen_time_sec, valid_json_count, retry_courses_count = 0.0, 0, 0

    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for idx, (orig_idx, row) in enumerate(df_run.iterrows(), start=1):
            cid = str(row.get("course_id", row.get("id", f"course_{idx}")))
            if cid in done_records:
                rec = done_records[cid]
                rec["dup_group_id"] = gid_map.get(cid, cid)
                rec["model_revision"] = model_revision
            else:
                rec = generate_one_course(
                    row=row,
                    course_index=idx - 1 if args.num_shards == 1 else int(orig_idx),
                    model=model, tokenizer=tokenizer,
                    sys_prompt=sys_prompt, user_template=user_template,
                    max_new_tokens=args.max_new_tokens, max_retries=args.max_retries,
                    add_inferred_hint=args.add_inferred_hint, force_mask_prereq=args.masked_prereq_eval
                )
                rec["model_name"] = args.model_name
                rec["model_revision"] = model_revision
                rec["dup_group_id"] = gid_map.get(cid, cid)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

            results.append(rec)
            retry_courses_count += int(rec["attempts"] > 1)
            valid_json_count += int(rec["is_valid_json"])
            total_input_tokens += rec["input_tokens"]
            total_output_tokens += rec["output_tokens"]
            total_gen_time_sec += rec["generation_time_sec"]

    n_tested = len(df_run)
    tokens_per_sec = total_output_tokens / max(total_gen_time_sec, 1e-6)
    sec_per_course = total_gen_time_sec / max(n_tested, 1)
    gpu_hours_per_course = sec_per_course / 3600.0

    peak_vram_per_gpu = {
        f"cuda:{gi}": round(torch.cuda.max_memory_allocated(gi) / (1024 ** 3), 2)
        for gi in range(torch.cuda.device_count())
    } if torch.cuda.is_available() else {}
    peak_vram_gb = max(peak_vram_per_gpu.values()) if peak_vram_per_gpu else 0.0

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encode_info, masked_eval_summary = None, None
    if args.masked_prereq_eval:
        masked_eval_summary = evaluate_masked_prereq_predictions(
            results, args.encoder_name, args.output_dir, df_courses=df, seed=args.eval_seed
        )
        profiles_path = os.path.join(args.output_dir, f"masked_prereq_{n_tested}_profiles.json")
        report_path = os.path.join(args.output_dir, f"masked_prereq_benchmark_{n_tested}.json")
    elif args.num_shards == 1:
        encode_info = encode_both_texts_with_sbert(
            results, args.encoder_name, args.output_dir, batch_size=args.encode_batch_size, dup_summary=dup_summary
        )
        profiles_path = os.path.join(args.output_dir, f"pilot_{n_tested}_profiles.json")
        report_path = os.path.join(args.output_dir, f"throughput_benchmark_{n_tested}.json")
    else:
        profiles_path = os.path.join(args.output_dir, f"profiles_shard{args.shard}.json")
        report_path = os.path.join(args.output_dir, f"throughput_benchmark_shard{args.shard}.json")

    benchmark_report = {
        "model_name": args.model_name,
        "model_revision": model_revision,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "bitsandbytes": getattr(bnb, "__version__", "unknown"),
            "sentence_transformers": sentence_transformers.__version__
        },
        "dup_group_summary": dup_summary,
        "num_courses_tested": n_tested,
        "valid_json_rate": round(valid_json_count / max(n_tested, 1), 4),
        "schema_compliant_rate": round(sum(1 for r in results if r["is_schema_compliant"]) / max(n_tested, 1), 4),
        "university_or_instructor_mention_count": sum(1 for r in results if r["mentions_university_or_instructor"]),
        "courses_needed_retry": retry_courses_count,
        "throughput_tokens_per_sec": round(tokens_per_sec, 2),
        "avg_sec_per_course": round(sec_per_course, 4),
        "gpu_hours_per_course": round(gpu_hours_per_course, 6),
        "peak_vram_gb_single_gpu": round(peak_vram_gb, 2),
        "sbert_encoding_summary": encode_info,
        "masked_prereq_evaluation": masked_eval_summary
    }

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
