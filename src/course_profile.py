
import os
import re
import json
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import transformers
import bitsandbytes as bnb
import sentence_transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

NEG_PREREQ_REGEX = re.compile(r"^(none|no|not required|not needed|nothing|n/a)\b", re.I)

REQUIRED_JSON_KEYS = [
    "overview",
    "target_learners",
    "learning_outcomes",
    "prerequisite_knowledge",
    "follow_up_topics"
]

# Giới hạn lược đồ 
MAX_OVERVIEW_WORDS = 60
MAX_LEARNER_WORDS = 30
MIN_OUTCOMES, MAX_OUTCOMES = 3, 5
MIN_LIST_ITEMS, MAX_LIST_ITEMS = 2, 5

INFERRED_HINT = (
    "\n\nNOTE: The prerequisite field of this course is 'not stated' in the source metadata. "
    "Do NOT answer 'none' or 'not stated' for prerequisite_knowledge. Infer the required knowledge/skills "
    "conservatively and begin the prerequisite_knowledge value with the tag '(inferred)'."
)

NEG_PREREQ_OUTPUT_REGEX = re.compile(r"^\W*(none|no|not|n/a|nothing|unknown|unspecified)\b", re.I)

# Biểu thức nhận diện nhãn (inferred)
TAG = re.compile(r"\s*\(\s*inferred\s*\)\s*:?\s*", re.I)

# Kiểm tra giá trị rỗng, NaN, None hoặc chuỗi 'nan'/'null'
def is_empty_val(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return len(x) == 0
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return (not s) or (s.lower() in {"nan", "none", "null", "[]"})

# Chuẩn hóa trường thành danh sách và bỏ nhãn (inferred) trước khi mã hóa
def as_list(x):
    if is_empty_val(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i).strip() for i in x if not is_empty_val(i)]
    return [s.strip() for s in re.split(r"[;,]", str(x)) if s.strip()]


def normalize_profile(p):
    p = dict(p)
    pre = as_list(p.get("prerequisite_knowledge"))
    # Dùng TAG.search thay vì TAG.match để bắt được (inferred) ở mọi vị trí
    inferred = bool(pre) and any(bool(TAG.search(x)) for x in pre)
    p["prerequisite_knowledge"] = [TAG.sub(" ", x).strip(" ,;:-") for x in pre if TAG.sub(" ", x).strip(" ,;:-")]
    p["follow_up_topics"] = [
        TAG.sub(" ", x).strip(" ,;:-") for x in as_list(p.get("follow_up_topics"))
        if TAG.sub(" ", x).strip(" ,;:-")
    ]
    p["learning_outcomes"] = [
        TAG.sub(" ", x).strip() for x in as_list(p.get("learning_outcomes"))
        if TAG.sub(" ", x).strip()
    ]
    p["overview"] = TAG.sub(" ", str(p.get("overview", ""))).strip()
    p["target_learners"] = TAG.sub(" ", str(p.get("target_learners", ""))).strip()
    return p, inferred

# Chuẩn hóa trường tiên quyết
def norm_prereq(x):
    if is_empty_val(x):
        return "not stated"
    if isinstance(x, (list, tuple, np.ndarray)):
        s = ", ".join(str(i).strip() for i in x if not is_empty_val(i)).strip()
    else:
        s = str(x).strip()
    if not s or s.lower() in {"nan", "null"}:
        return "not stated"
    return "none" if NEG_PREREQ_REGEX.match(s) else s


# Làm sạch các trường văn bản đơn hoặc danh sách
def clean_text_field(x, default_val="unknown"):
    if is_empty_val(x):
        return default_val
    if isinstance(x, (list, tuple, np.ndarray)):
        items = [str(i).strip() for i in x if not is_empty_val(i)]
        unique_items = list(dict.fromkeys(items))
        return ", ".join(unique_items) if unique_items else default_val
    s = str(x).strip()
    return s if s else default_val


# Trích xuất tối đa 15 khái niệm từ concept_name
def clean_concepts(x, max_items=15):
    if is_empty_val(x):
        return "not available"
    items = []
    if isinstance(x, (list, tuple, np.ndarray)):
        items = [str(i).strip() for i in x if not is_empty_val(i)]
    else:
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = [str(i).strip() for i in parsed if not is_empty_val(i)]
            except Exception:
                items = [i.strip(" '\"") for i in re.split(r"[,;|]", s[1:-1]) if i.strip(" '\"")]
        else:
            items = [i.strip() for i in re.split(r"[,;|]", s) if i.strip()]
    unique_items = list(dict.fromkeys(items))[:max_items]
    return ", ".join(unique_items) if unique_items else "not available"

# Lấy văn bản gốc course_text của khóa học để mã hóa bằng Sentence-BERT
# Nếu cột course_text bị trống, tự động ghép từ tên khóa, trường, giảng viên và mô tả
def get_clean_raw_course_text(row):
    raw_txt = clean_text_field(row.get("course_text"), default_val="")
    if raw_txt:
        return raw_txt
    name = clean_text_field(row.get("course_name"), default_val="unknown")
    school = clean_text_field(row.get("school_name"), default_val="unknown")
    teacher = clean_text_field(row.get("teacher_name"), default_val="unknown")
    about = clean_text_field(row.get("course_about"), default_val="not stated")
    return f"Course: {name}. University: {school}. Instructor: {teacher}. Description: {about}"

# Chuyển một trường (chuỗi hoặc danh sách) thành chuỗi phẳng
def field_to_text(x, sep="; "):
    if is_empty_val(x):
        return ""
    if isinstance(x, (list, tuple, np.ndarray)):
        return sep.join(str(i).strip() for i in x if not is_empty_val(i))
    return str(x).strip()


def word_count(x):
    return len(re.findall(r"\S+", field_to_text(x)))

# Kiểm tra hồ sơ có nhắc đến tên trường đại học hoặc tên giảng viên hay không
def check_name_mentions(profile, row):
    if not profile or row is None:
        return []
    combined_text = " ".join(field_to_text(profile.get(k)) for k in REQUIRED_JSON_KEYS).lower()
    mentions = []

    school_str = clean_text_field(row.get("school_name"), default_val="")
    if school_str and school_str.lower() != "unknown":
        for s in re.split(r"[,;|]", school_str):
            s_clean = s.strip()
            if len(s_clean) >= 3 and re.search(r"\b" + re.escape(s_clean.lower()) + r"\b", combined_text):
                mentions.append(f"school '{s_clean}'")
            if "massachusetts institute of technology" in s_clean.lower() and re.search(r"\bmit\b", combined_text):
                mentions.append("school 'MIT'")

    teacher_str = clean_text_field(row.get("teacher_name"), default_val="")
    if teacher_str and teacher_str.lower() != "unknown":
        for t in re.split(r"[,;|]", teacher_str):
            t_clean = t.strip()
            if len(t_clean) >= 4 and re.search(r"\b" + re.escape(t_clean.lower()) + r"\b", combined_text):
                mentions.append(f"instructor '{t_clean}'")

    return mentions

# Hàm hậu xử lý xóa tên trường/giảng viên nếu LLM vẫn cố tình nhắc sau khi đã retry
def scrub_name_mentions(profile, row):
    if not profile or row is None:
        return profile
    p = dict(profile)
    forbidden_schools = []
    school_str = clean_text_field(row.get("school_name"), default_val="")
    if school_str and school_str.lower() != "unknown":
        for s in re.split(r"[,;|]", school_str):
            s_clean = s.strip()
            if len(s_clean) >= 3:
                forbidden_schools.append(s_clean)
            if "massachusetts institute of technology" in s_clean.lower():
                forbidden_schools.append("MIT")

    forbidden_teachers = []
    teacher_str = clean_text_field(row.get("teacher_name"), default_val="")
    if teacher_str and teacher_str.lower() != "unknown":
        for t in re.split(r"[,;|]", teacher_str):
            t_clean = t.strip()
            if len(t_clean) >= 4:
                forbidden_teachers.append(t_clean)

    def _clean_str(text):
        if not isinstance(text, str):
            return text
        out = text
        for s in sorted(forbidden_schools, key=len, reverse=True):
            out = re.sub(r"\b(?:at|from|by|of|in)\s+(?:the\s+)?" + re.escape(s) + r"\b", "", out, flags=re.I)
            out = re.sub(r"\b" + re.escape(s) + r"\b", "the university", out, flags=re.I)
        for t in sorted(forbidden_teachers, key=len, reverse=True):
            out = re.sub(
                r"\b(?:taught\s+by|by|with|instructor|professor|prof\.?|dr\.?)\s+" + re.escape(t) + r"\b",
                "", out, flags=re.I
            )
            out = re.sub(r"\b" + re.escape(t) + r"\b", "the instructor", out, flags=re.I)
        out = re.sub(r"\s{2,}", " ", out).strip(" ,;-")
        return out

    for k in REQUIRED_JSON_KEYS:
        val = p.get(k)
        if isinstance(val, list):
            p[k] = [_clean_str(str(item)) for item in val if _clean_str(str(item))]
        elif isinstance(val, str):
            p[k] = _clean_str(val)
    return p

# Kiểm tra các giới hạn số từ và số ý theo lược đồ
def check_schema_constraints(profile, row=None):
    violations = []
    n_ov = word_count(profile.get("overview"))
    if n_ov == 0 or n_ov > MAX_OVERVIEW_WORDS:
        violations.append(f"overview has {n_ov} words (limit {MAX_OVERVIEW_WORDS})")

    n_tl = word_count(profile.get("target_learners"))
    if n_tl == 0 or n_tl > MAX_LEARNER_WORDS:
        violations.append(f"target_learners has {n_tl} words (limit {MAX_LEARNER_WORDS})")

    outcomes = profile.get("learning_outcomes")
    if not isinstance(outcomes, (list, tuple)):
        violations.append("learning_outcomes is not a list")
    else:
        k = len([o for o in outcomes if not is_empty_val(o)])
        if not (MIN_OUTCOMES <= k <= MAX_OUTCOMES):
            violations.append(f"learning_outcomes has {k} items (required {MIN_OUTCOMES}-{MAX_OUTCOMES})")
            
    # Kiểm tra prerequisite_knowledge và follow_up_topics có từ 2-5 mục sau khi chuẩn hóa thành danh sách
    pre_list = as_list(profile.get("prerequisite_knowledge"))
    if not (MIN_LIST_ITEMS <= len(pre_list) <= MAX_LIST_ITEMS):
        violations.append(f"prerequisite_knowledge has {len(pre_list)} items (required {MIN_LIST_ITEMS}-{MAX_LIST_ITEMS})")

    fol_list = as_list(profile.get("follow_up_topics"))
    if not (MIN_LIST_ITEMS <= len(fol_list) <= MAX_LIST_ITEMS):
        violations.append(f"follow_up_topics has {len(fol_list)} items (required {MIN_LIST_ITEMS}-{MAX_LIST_ITEMS})")

    # Kiểm tra không nhắc tên trường hoặc giảng viên trong bất kỳ trường nào
    name_mentions = check_name_mentions(profile, row)
    if name_mentions:
        violations.append(f"mentions forbidden entity ({', '.join(name_mentions)})")

    return violations


# Kiểm tra tính nhất quán của nhãn (inferred) khi tiên quyết gốc là 'not stated'
def classify_prereq_label(prereq_norm, profile):
    if prereq_norm != "not stated" or profile is None:
        return "not_applicable"
    txt = field_to_text(profile.get("prerequisite_knowledge"))
    # Kiểm tra (inferred) ở bất kỳ vị trí nào trong prerequisite_knowledge
    if "(inferred)" in txt.lower():
        return "inferred_labeled"
    if NEG_PREREQ_OUTPUT_REGEX.match(txt):
        return "declared_none_or_unstated"
    return "unlabeled_content"

# Nối kết 5 trường của hồ sơ sư phạm thành văn bản thống nhất để đưa vào Sentence-BERT
# Nếu profile bị lỗi (None), dùng fallback_text để đảm bảo không bị mất dòng khi tạo ma trận embedding
def build_profile_text(profile, fallback_text=""):
    if not profile:
        return fallback_text
    parts = [
        ("Overview", field_to_text(profile.get("overview"))),
        ("Target learners", field_to_text(profile.get("target_learners"))),
        ("Learning outcomes", field_to_text(profile.get("learning_outcomes"))),
        ("Prerequisite knowledge", field_to_text(profile.get("prerequisite_knowledge"))),
        ("Follow-up topics", field_to_text(profile.get("follow_up_topics"))),
    ]
    joined = " ".join(
        f"{k}: {v}." if not v.endswith((".", "!", "?")) else f"{k}: {v}"
        for k, v in parts if v
    ).strip()
    return joined if joined else fallback_text

# Lấy số chiều của SentenceTransformer
def get_sbert_dim(encoder):
    if hasattr(encoder, "get_embedding_dimension"):
        return encoder.get_embedding_dimension()
    return encoder.get_sentence_embedding_dimension()

# Sắp xếp DataFrame theo bảng ánh xạ  
def align_dataframe_order(df, item2idx_path=None):
    df = df.copy()
    if "course_id" not in df.columns and "id" in df.columns:
        df["course_id"] = df["id"].astype(str)
    else:
        df["course_id"] = df["course_id"].astype(str)

    if item2idx_path and os.path.exists(item2idx_path):
        print(f"Đang căn chỉnh thứ tự khóa học theo bảng ánh xạ: {item2idx_path}")
        if item2idx_path.endswith(".json"):
            with open(item2idx_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict):
                ordered_ids = [cid for cid, _ in sorted(mapping.items(), key=lambda x: int(x[1]))]
            elif isinstance(mapping, list):
                ordered_ids = [str(x) for x in mapping]
            else:
                ordered_ids = None
        elif item2idx_path.endswith(".parquet"):
            map_df = pd.read_parquet(item2idx_path)
            cid_col = "course_id" if "course_id" in map_df.columns else map_df.columns[0]
            ordered_ids = map_df[cid_col].astype(str).tolist()
        elif item2idx_path.endswith(".csv"):
            map_df = pd.read_csv(item2idx_path)
            cid_col = "course_id" if "course_id" in map_df.columns else map_df.columns[0]
            ordered_ids = map_df[cid_col].astype(str).tolist()
        else:
            ordered_ids = None

        if ordered_ids is not None:
            df = df.set_index("course_id").loc[ordered_ids].reset_index()
            return df

    return df.sort_values("course_id").reset_index(drop=True)

# Tách bước mã hóa course_text gốc cho toàn bộ 3.769 khóa 
def encode_all_raw_course_texts_only(df, encoder_name, output_dir, batch_size=32):
    from sentence_transformers import SentenceTransformer

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
        "course_index": list(range(len(df))),
        "course_id": course_ids
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu ma trận raw_course_text_embeddings ({raw_emb.shape}) tại: {raw_emb_path}")
    print(f"Đã lưu tệp thứ tự ánh xạ course_id tại: {idx_path}")
    return meta

# Mã hóa đồng thời cả course_text và hồ sơ sư phạm 5 chiều do LLM sinh
def encode_both_texts_with_sbert(results, encoder_name, output_dir, batch_size=32):
    from sentence_transformers import SentenceTransformer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)  

    raw_texts = [r["raw_course_text"] for r in results]
    llm_texts = [r["profile_text"] for r in results]
    name_texts = [r.get("course_name", "unknown") for r in results]

    # Thống kê số lượng token và số văn bản vượt quá max_seq_length của Sentence-BERT
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

    name_emb = encoder.encode(
        name_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False
    ).astype(np.float32)

    # Kiểm tra bảo toàn kích thước [N_courses, d_text]
    emb_dim = get_sbert_dim(encoder)
    assert raw_emb.shape == llm_emb.shape == (len(results), emb_dim), (
        f"Lỗi lệch kích thước embedding: raw={raw_emb.shape}, llm={llm_emb.shape}"
    )

    # Tính độ tương đồng cosine giữa vector văn bản gốc và vector hồ sơ LLM
    raw_norm = raw_emb / np.clip(np.linalg.norm(raw_emb, axis=1, keepdims=True), 1e-8, None)
    llm_norm = llm_emb / np.clip(np.linalg.norm(llm_emb, axis=1, keepdims=True), 1e-8, None)
    name_norm = name_emb / np.clip(np.linalg.norm(name_emb, axis=1, keepdims=True), 1e-8, None)

    cos_sims = np.sum(raw_norm * llm_norm, axis=1)
    # Đo tự động cosine giữa embedding tên khóa học và embedding hồ sơ để phát hiện lỗi dữ liệu nguồn
    name_llm_cos_sims = np.sum(name_norm * llm_norm, axis=1)
    for r, c_raw, c_name in zip(results, cos_sims, name_llm_cos_sims):
        r["cosine_raw_vs_llm"] = round(float(c_raw), 4)
        r["cosine_name_vs_llm"] = round(float(c_name), 4)

    # Liệt kê 1% khóa học (tối thiểu 1 khóa) có cosine(course_name, profile_text) thấp nhất để chấm tay
    k_lowest = max(1, int(np.ceil(0.01 * len(results))))
    sorted_indices = np.argsort(name_llm_cos_sims)[:k_lowest]
    lowest_1pct_courses = [
        {
            "course_id": results[i]["course_id"],
            "course_name": results[i]["course_name"],
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
    idx_path = os.path.join(output_dir, "embeddings_course_index.json")

    np.save(raw_emb_path, raw_emb)
    np.save(llm_emb_path, llm_emb)

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
        "course_index": [r["course_index"] for r in results],
        "course_id": [r["course_id"] for r in results]
    }

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    return index_meta

# Đo tự động chất lượng suy diễn tiên quyết trên 200 khóa học ẩn tiên quyết
def evaluate_masked_prereq_predictions(results, encoder_name, output_dir, seed=2026):
    from sentence_transformers import SentenceTransformer
    from scipy.stats import wilcoxon

    valid_records = [
        r for r in results
        if r.get("profile") and field_to_text(r["profile"].get("prerequisite_knowledge"))
        and r.get("ground_truth_prereq")
    ]
    if not valid_records:
        print("Không có bản ghi hợp lệ để đánh giá lượt ẩn tiên quyết.")
        return {}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    encoder.eval()

    inferred_texts = [field_to_text(r["profile"]["prerequisite_knowledge"]) for r in valid_records]
    gt_texts = [r["ground_truth_prereq"] for r in valid_records]

    inf_emb = encoder.encode(inferred_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    gt_emb = encoder.encode(gt_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # Cosine giữa tiên quyết suy diễn và tiên quyết gốc của cùng khóa học
    paired_cos = np.sum(inf_emb * gt_emb, axis=1)

    # Mức nền: ghép mỗi tiên quyết suy diễn với tiên quyết gốc của một khóa học khác chọn ngẫu nhiên
    rng = np.random.default_rng(seed)
    n = len(valid_records)
    rand_indices = np.arange(n)
    if n > 1:
        for i in range(n):
            choices = [j for j in range(n) if j != i]
            rand_indices[i] = rng.choice(choices)
    random_cos = np.sum(inf_emb * gt_emb[rand_indices], axis=1)

    for idx_r, r in enumerate(valid_records):
        r["cosine_inferred_vs_gt"] = round(float(paired_cos[idx_r]), 4)
        r["cosine_inferred_vs_random_gt"] = round(float(random_cos[idx_r]), 4)

    # Kiểm định Wilcoxon cặp
    if n > 1 and not np.allclose(paired_cos, random_cos):
        stat, p_val = wilcoxon(paired_cos, random_cos, alternative="greater")
        stat, p_val = float(stat), float(p_val)
    else:
        stat, p_val = 0.0, 1.0

    eval_summary = {
        "num_evaluated_courses": n,
        "seed": seed,
        "encoder_name": encoder_name,
        "paired_cosine": {
            "mean": round(float(np.mean(paired_cos)), 4),
            "std": round(float(np.std(paired_cos)), 4),
            "median": round(float(np.median(paired_cos)), 4)
        },
        "random_baseline_cosine": {
            "mean": round(float(np.mean(random_cos)), 4),
            "std": round(float(np.std(random_cos)), 4),
            "median": round(float(np.median(random_cos)), 4)
        },
        "wilcoxon_signed_rank_test": {
            "statistic": round(stat, 4),
            "p_value_one_sided_greater": p_val
        }
    }

    eval_report_path = os.path.join(output_dir, "masked_prereq_200_eval_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)

    print(
        f"Kết quả lượt ẩn tiên quyết ({n} khóa): "
        f"Cosine cặp đúng = {eval_summary['paired_cosine']['mean']} ± {eval_summary['paired_cosine']['std']} vs. "
        f"Ngẫu nhiên = {eval_summary['random_baseline_cosine']['mean']} ± {eval_summary['random_baseline_cosine']['std']} "
        f"(Wilcoxon p = {p_val:.4e})"
    )
    return eval_summary
    
def load_prompt_template(template_path):
    if not template_path or not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Không tìm thấy file prompt template: {template_path}. Hãy kiểm tra lại đường dẫn."
        )

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"File prompt template rỗng: {template_path}")

    if "USER:" in content:
        parts = content.split("USER:", 1)
        sys_prompt = parts[0].replace("SYSTEM:", "").strip()
        user_template = parts[1].strip()
    else:
        sys_prompt = (
            "You are an instructional designer. Use only the information given. "
            "If something is not stated, infer conservatively and mark it \"(inferred)\". Output valid JSON only."
        )
        user_template = content

    cleaned_lines = []
    for line in user_template.splitlines():
        if line.strip().startswith("#"):
            continue
        if "{prereq}" in line and "#" in line:
            line = line.split("#")[0].rstrip()
        cleaned_lines.append(line)
    user_template = "\n".join(cleaned_lines).strip()

    user_template = re.sub(r"\{\s*field\s+or\s+[^}]+\}", "{field}", user_template)
    user_template = re.sub(r"\{\s*concepts\s+or\s+[^}]+\}", "{concepts}", user_template)

    if not user_template:
        raise ValueError(f"Phần USER của template bị rỗng sau khi xử lý: {template_path}")

    return sys_prompt, user_template


# Tạo message từ siêu dữ liệu tĩnh của khóa học
def build_course_messages(
    row, sys_prompt, user_template, retry_idx=0, retry_reason="",
    inferred_hint=True, prev_output="", force_mask_prereq=False
):
    name = clean_text_field(row.get("course_name"), default_val="unknown")
    school = clean_text_field(row.get("school_name"), default_val="unknown")
    teacher = clean_text_field(row.get("teacher_name"), default_val="unknown")
    field = clean_text_field(row.get("course_field"), default_val="unknown")
    about = clean_text_field(row.get("course_about"), default_val="not stated")
    prereq = "not stated" if force_mask_prereq else norm_prereq(row.get("course_prerequisites"))
    concepts = clean_concepts(row.get("concept_name"), max_items=15)

    # Trích xuất danh sách tên trường/giảng viên cụ thể để dặn LLM tuyệt đối không đưa vào JSON
    forbidden_list = []
    if school.lower() != "unknown":
        forbidden_list.extend([s.strip() for s in re.split(r"[,;|]", school) if len(s.strip()) >= 3])
        if "massachusetts institute of technology" in school.lower():
            forbidden_list.append("MIT")
    if teacher.lower() != "unknown":
        forbidden_list.extend([t.strip() for t in re.split(r"[,;|]", teacher) if len(t.strip()) >= 4])

    user_content = user_template.format(
        name=name,
        school=school,
        teacher=teacher,
        field=field,
        about=about,
        prereq=prereq,
        concepts=concepts
    )

    if forbidden_list:
        user_content += (
            f"\nIMPORTANT: Keep university and instructor names for context only. "
            f"NEVER write any of these exact names in your JSON output: {', '.join(forbidden_list)}."
        )

    if inferred_hint and prereq == "not stated":
        user_content += INFERRED_HINT

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

    # Nếu phải sinh lại, dùng hội thoại nhiều lượt để LLM thấy rõ JSON lỗi vừa sinh và sửa trực tiếp
    if retry_idx > 0:
        fix_prompt = (
            f"Attempt {retry_idx} correction: Your previous output violated schema constraints ({retry_reason}). "
            "Rewrite and return ONLY the corrected valid JSON object with the exact 5 keys: "
            "\"overview\" (<= 60 words), \"target_learners\" (<= 30 words), "
            "\"learning_outcomes\" (list of 3-5 items), \"prerequisite_knowledge\" (list of 2-5 items), "
            "\"follow_up_topics\" (list of 2-5 items). "
            "Remove any mention of university or instructor names completely."
        )
        if prev_output:
            messages.append({"role": "assistant", "content": prev_output})
            messages.append({"role": "user", "content": fix_prompt})
        else:
            messages[-1]["content"] += "\n\n" + fix_prompt

    return messages, user_content


# Trích xuất và kiểm tra cú pháp JSON 5 chiều
def extract_and_validate_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None, "Không tìm thấy cặp dấu ngoặc nhọn JSON '{}'"

    json_str = text[start_idx:end_idx + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Lỗi cú pháp JSON: {str(e)}"

    if not isinstance(data, dict):
        return None, "Đầu ra JSON không phải là một Object"

    missing_keys = [k for k in REQUIRED_JSON_KEYS if k not in data]
    if missing_keys:
        return None, f"Thiếu các trường bắt buộc: {missing_keys}"

    return data, None

# Sinh hồ sơ cho 1 khóa học (tách hàm để dùng chung cho chạy thử, chạy shard và chạy ẩn tiên quyết 200 khóa)
def generate_one_course(
    row, course_index, model, tokenizer, sys_prompt, user_template,
    max_new_tokens=300, max_retries=2, add_inferred_hint=True, force_mask_prereq=False
):
    course_id = str(row.get("course_id", row.get("id", f"course_{course_index + 1}")))
    course_name = clean_text_field(row.get("course_name"), default_val="unknown")
    raw_course_text = get_clean_raw_course_text(row)
    original_prereq_norm = norm_prereq(row.get("course_prerequisites"))
    effective_prereq_norm = "not stated" if force_mask_prereq else original_prereq_norm

    parsed_json = None
    raw_output = ""
    error_msg = None
    course_gen_time = 0.0
    course_in_tokens = 0
    course_out_tokens = 0
    attempts_used = 0
    schema_violations = []
    retry_reason = ""
    fallback_json, fallback_violations, fallback_raw = None, [], ""

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1
        messages, user_prompt_str = build_course_messages(
            row, sys_prompt, user_template,
            retry_idx=attempt, retry_reason=retry_reason,
            inferred_hint=add_inferred_hint,
            prev_output=raw_output,
            force_mask_prereq=force_mask_prereq
        )
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        in_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Giải mã tham lam (temperature = 0)
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_ids = outputs[0][in_len:]
        out_len = len(gen_ids)
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

        course_gen_time += elapsed
        course_in_tokens += in_len
        course_out_tokens += out_len

        candidate, error_msg = extract_and_validate_json(raw_output)
        retry_reason = ""
        if candidate is not None:
            violations = check_schema_constraints(candidate, row=row)
            if not violations:
                parsed_json, schema_violations = candidate, []
                break
            fallback_json, fallback_violations, fallback_raw = candidate, violations, raw_output
            retry_reason = "; ".join(violations)
            error_msg = "Schema violation: " + retry_reason

    if parsed_json is None and fallback_json is not None:
        if check_name_mentions(fallback_json, row):
            fallback_json = scrub_name_mentions(fallback_json, row)
            fallback_violations = check_schema_constraints(fallback_json, row=row)
        parsed_json, schema_violations, raw_output = fallback_json, fallback_violations, fallback_raw

    prereq_label_status = classify_prereq_label(effective_prereq_norm, parsed_json)

    # Chuẩn hóa danh sách và loại bỏ nhãn (inferred) trước khi tạo profile_text
    if parsed_json is not None:
        prof, inferred = normalize_profile(parsed_json)
        name_mentions = check_name_mentions(prof, row)
    else:
        prof, inferred, name_mentions = None, False, []

    profile_text = build_profile_text(prof, fallback_text=raw_course_text)

    rec = {
        "course_index": int(course_index),
        "course_id": course_id,
        "course_name": course_name,
        "normalized_prereq": effective_prereq_norm,
        "prereq_inferred": inferred,
        "mentions_university_or_instructor": bool(name_mentions),
        "attempts": attempts_used,
        "is_valid_json": parsed_json is not None,
        "is_schema_compliant": parsed_json is not None and not schema_violations,
        "schema_violations": schema_violations,
        "prereq_label_status": prereq_label_status,
        "error_message": error_msg,
        "input_tokens": course_in_tokens,
        "output_tokens": course_out_tokens,
        "generation_time_sec": round(course_gen_time, 4),
        "prompt_user": user_prompt_str,
        "raw_output": raw_output,
        "profile": prof,
        "raw_course_text": raw_course_text,
        "profile_text": profile_text
    }
    if force_mask_prereq:
        rec["ground_truth_prereq"] = original_prereq_norm
    return rec

# Gộp kết quả từ các shard .jsonl, kiểm tra đủ 3.769 course_id không trùng và mã hóa SBERT
def merge_shards_and_encode(df, args):
    df_aligned = align_dataframe_order(df, args.item2idx_path)
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

    merged_results = []
    for idx, cid in enumerate(expected_ids):
        rec = records_by_id[cid]
        rec["course_index"] = idx
        merged_results.append(rec)

    assert not any("(inferred)" in r["profile_text"].lower() for r in merged_results), (
        "Lỗi: Vẫn còn chuỗi '(inferred)' trong profile_text đưa vào SBERT!"
    )

    encode_info = encode_both_texts_with_sbert(
        merged_results, args.encoder_name, args.output_dir, batch_size=args.encode_batch_size
    )

    full_json_path = os.path.join(args.output_dir, f"full_{len(merged_results)}_profiles.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"Đã gộp thành công {len(merged_results)} hồ sơ không trùng lặp tại: {full_json_path}")
    return encode_info

def main():
    parser = argparse.ArgumentParser(description="Sinh hồ sơ sư phạm 5 chiều bằng LLM và mã hóa văn bản SBERT")
    parser.add_argument("--data_path", type=str, default="data/course_llm/course_data.parquet")
    parser.add_argument("--prompt_path", type=str, default="src/prompts/prompt_template.txt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_courses", type=int, default=50, help="Đặt <= 0 hoặc >= 3769 để chạy toàn bộ")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--single_gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_inferred_hint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="data/course_llm")
    # Các tham số bổ sung
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--item2idx_path", type=str, default=None)
    parser.add_argument("--encode_raw_only", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--masked_prereq_eval", action="store_true")
    parser.add_argument("--eval_sample_size", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Đang đọc dữ liệu khóa học từ: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    df = df.loc[:, ~df.columns.duplicated()]
    total_dataset_courses = len(df)

    # Thống kê số khóa học có tiên quyết ghi rõ trên toàn bộ 3.769 khóa
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

    # Chỉ mã hóa course_text gốc cho đủ 3.769 khóa theo thứ tự item2idx dùng chung
    if args.encode_raw_only:
        df_ordered = align_dataframe_order(df, args.item2idx_path)
        encode_all_raw_course_texts_only(
            df_ordered, args.encoder_name, args.output_dir, batch_size=args.encode_batch_size
        )
        return

    # Gộp các file shard .jsonl sau khi chạy xong trên 2 tài khoản và mã hóa SBERT
    if args.merge_shards:
        merge_shards_and_encode(df, args)
        return

    # Chọn tập dữ liệu chạy 
    if args.masked_prereq_eval:
        df_stated = df.loc[stated_prereq_mask].copy()
        df_stated = align_dataframe_order(df_stated, None)
        sample_n = min(args.eval_sample_size, len(df_stated))
        df_run = df_stated.sample(n=sample_n, random_state=args.eval_seed).reset_index(drop=True)
        jsonl_path = os.path.join(args.output_dir, f"masked_prereq_{sample_n}_profiles.jsonl")
        print(f"-> Chạy lượt ẩn tiên quyết trên {len(df_run)}/{len(df_stated)} khóa có tiên quyết (seed={args.eval_seed})")
    else:
        if args.num_shards > 1 or args.num_courses <= 0 or args.num_courses >= total_dataset_courses:
            df_ordered = align_dataframe_order(df, args.item2idx_path)
            df_run = df_ordered.iloc[args.shard::args.num_shards].copy()
            jsonl_path = os.path.join(args.output_dir, f"profiles_shard{args.shard}.jsonl")
            print(f"-> [Shard {args.shard}/{args.num_shards}] Số khóa học phân công: {len(df_run)}")
        else:
            df_run = df.head(args.num_courses).copy()
            jsonl_path = os.path.join(args.output_dir, f"pilot_{len(df_run)}_profiles.jsonl")
            print(f"-> Số khóa học lấy ra chạy thử       : {len(df_run)}")

    # Kiểm tra các course_id đã hoàn thành trong file .jsonl để chạy tiếp nếu bị ngắt
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
        if done_records:
            print(f"-> Phát hiện {len(done_records)} khóa học đã sinh trước đó trong {jsonl_path}, sẽ bỏ qua và chạy tiếp.")

    sys_prompt, user_template = load_prompt_template(args.prompt_path)

    print(f"\nĐang tải mô hình {args.model_name} với lượng tử hóa 4-bit (NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map_cfg = {"": 0} if (args.single_gpu and torch.cuda.is_available()) else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map_cfg,
        trust_remote_code=True
    )
    model.eval()
    model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

    if torch.cuda.is_available():
        for gi in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gi)

    print(f"\nBắt đầu sinh hồ sơ sư phạm cho {len(df_run)} khóa học (temperature=0, max_new_tokens={args.max_new_tokens})")
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_gen_time_sec = 0.0
    valid_json_count = 0
    retry_courses_count = 0

    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for idx, (orig_idx, row) in enumerate(df_run.iterrows(), start=1):
            cid = str(row.get("course_id", row.get("id", f"course_{idx}")))
            if cid in done_records:
                rec = done_records[cid]
            else:
                rec = generate_one_course(
                    row=row,
                    course_index=idx - 1 if args.num_shards == 1 else int(orig_idx),
                    model=model,
                    tokenizer=tokenizer,
                    sys_prompt=sys_prompt,
                    user_template=user_template,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                    add_inferred_hint=args.add_inferred_hint,
                    force_mask_prereq=args.masked_prereq_eval
                )
                rec["model_name"] = args.model_name
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

            results.append(rec)
            if rec["attempts"] > 1:
                retry_courses_count += 1
            if rec["is_valid_json"]:
                valid_json_count += 1

            total_input_tokens += rec["input_tokens"]
            total_output_tokens += rec["output_tokens"]
            total_gen_time_sec += rec["generation_time_sec"]

            if idx % 5 == 0 or idx == len(df_run):
                curr_tps = total_output_tokens / max(total_gen_time_sec, 1e-6)
                print(
                    f"-> Đã xử lý {idx:4d}/{len(df_run)} khóa | "
                    f"JSON hợp lệ: {valid_json_count:4d}/{idx:4d} | "
                    f"Thông lượng: {curr_tps:.2f} token/giây | "
                    f"TB: {total_gen_time_sec / idx:.2f} giây/khóa"
                )

    # Kiểm tra không văn bản nào đưa vào SBERT còn chứa (inferred)
    assert not any("(inferred)" in r["profile_text"].lower() for r in results), (
        "Lỗi: Vẫn còn chuỗi '(inferred)' trong profile_text đưa vào SBERT!"
    )

    # Tính toán các chỉ số thông lượng và VRAM
    n_tested = len(df_run)
    tokens_per_sec = total_output_tokens / max(total_gen_time_sec, 1e-6)
    sec_per_course = total_gen_time_sec / max(n_tested, 1)
    gpu_hours_per_course = sec_per_course / 3600.0

    est_gpu_hours_3769 = 3769 * gpu_hours_per_course
    est_gpu_hours_with_200_eval = (3769 + 200) * gpu_hours_per_course

    peak_vram_per_gpu = {
        f"cuda:{gi}": round(torch.cuda.max_memory_allocated(gi) / (1024 ** 3), 2)
        for gi in range(torch.cuda.device_count())
    } if torch.cuda.is_available() else {}
    peak_vram_gb = max(peak_vram_per_gpu.values()) if peak_vram_per_gpu else 0.0
    peak_vram_sum_gb = round(sum(peak_vram_per_gpu.values()), 2)

    hf_map = getattr(model, "hf_device_map", None)
    model_devices = sorted({str(v) for v in hf_map.values()}) if hf_map else []

    n_schema_ok = sum(1 for r in results if r["is_schema_compliant"])
    n_mentions = sum(1 for r in results if r["mentions_university_or_instructor"])
    label_counts = {}
    for r in results:
        label_counts[r["prereq_label_status"]] = label_counts.get(r["prereq_label_status"], 0) + 1

    # Giải phóng bộ nhớ LLM trước khi nạp Sentence-BERT
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encode_info = None
    masked_eval_summary = None

    if args.masked_prereq_eval:
        print(f"\nĐang đánh giá tự động lượt ẩn tiên quyết bằng {args.encoder_name}")
        masked_eval_summary = evaluate_masked_prereq_predictions(
            results, args.encoder_name, args.output_dir, seed=args.eval_seed
        )
        profiles_path = os.path.join(args.output_dir, f"masked_prereq_{n_tested}_profiles.json")
        report_path = os.path.join(args.output_dir, f"masked_prereq_benchmark_{n_tested}.json")
    elif args.num_shards == 1:
        print(f"\nĐang mã hóa course_text và hồ sơ LLM bằng {args.encoder_name}")
        encode_info = encode_both_texts_with_sbert(
            results, args.encoder_name, args.output_dir, batch_size=args.encode_batch_size
        )
        profiles_path = os.path.join(args.output_dir, f"pilot_{n_tested}_profiles.json")
        report_path = os.path.join(args.output_dir, f"throughput_benchmark_{n_tested}.json")
    else:
        print(f"\nĐã hoàn tất sinh hồ sơ cho Shard {args.shard}/{args.num_shards}. Chạy --merge_shards sau khi đủ các shard để mã hóa SBERT")
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
        "dataset_prereq_counts": {
            "total_courses": total_dataset_courses,
            "stated_prereq_courses": num_stated_prereq,
            "none_prereq_courses": num_none_prereq,
            "not_stated_prereq_courses": num_unstated_prereq
        },
        "quantization": "4-bit NF4 (double quant, float16 compute)",
        "decoding": {"do_sample": False, "temperature": 0, "max_new_tokens": args.max_new_tokens},
        "shard_info": {"shard": args.shard, "num_shards": args.num_shards},
        "num_courses_tested": n_tested,
        "valid_json_rate": round(valid_json_count / max(n_tested, 1), 4),
        "schema_compliant_rate": round(n_schema_ok / max(n_tested, 1), 4),
        "university_or_instructor_mention_count": n_mentions,
        "university_or_instructor_mention_rate": round(n_mentions / max(n_tested, 1), 4),
        "courses_needed_retry": retry_courses_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens_per_course": round(total_output_tokens / max(n_tested, 1), 2),
        "total_generation_time_sec": round(total_gen_time_sec, 2),
        "avg_sec_per_course": round(sec_per_course, 4),
        "throughput_tokens_per_sec": round(tokens_per_sec, 2),
        "gpu_hours_per_course": round(gpu_hours_per_course, 6),
        "estimated_gpu_hours_3769_courses": round(est_gpu_hours_3769, 2),
        "estimated_gpu_hours_3969_total_runs": round(est_gpu_hours_with_200_eval, 2),
        "estimated_wall_hours_on_2_kaggle_accounts": round(est_gpu_hours_with_200_eval / 2.0, 2),
        "peak_vram_gb_single_gpu": round(peak_vram_gb, 2),
        "peak_vram_gb_sum_all_gpus": peak_vram_sum_gb,
        "peak_vram_gb_per_gpu": peak_vram_per_gpu,
        "model_devices_used": model_devices,
        "prereq_label_status_counts": label_counts,
        "sbert_encoding_summary": encode_info,
        "masked_prereq_evaluation": masked_eval_summary
    }

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

    print("\n=====================================================================")
    print(f"KẾT QUẢ ĐO THÔNG LƯỢNG LLM & MÃ HÓA VĂN BẢN ({n_tested} KHÓA HỌC)")
    print("=====================================================================")
    print(f" - Mô hình LLM                      : {benchmark_report['model_name']} (rev: {model_revision})")
    print(f" - Tỷ lệ JSON 5 chiều hợp lệ        : {valid_json_count}/{n_tested} ({benchmark_report['valid_json_rate']*100:.1f}%)")
    print(f" - Tỷ lệ đạt chuẩn giới hạn từ/ý    : {n_schema_ok}/{n_tested} ({benchmark_report['schema_compliant_rate']*100:.1f}%)")
    print(f" - Số hồ sơ nhắc tên trường/GV      : {n_mentions}/{n_tested} ({benchmark_report['university_or_instructor_mention_rate']*100:.1f}%)")
    print(f" - Số khóa học cần sinh lại (retry) : {retry_courses_count}")
    print(f" - Trung bình token sinh ra/khóa    : {benchmark_report['avg_output_tokens_per_course']} tokens")
    print(f" - Thông lượng (Throughput)         : {benchmark_report['throughput_tokens_per_sec']} token/giây")
    print(f" - Thời gian trung bình/khóa        : {benchmark_report['avg_sec_per_course']} giây/khóa")
    print(f" - Chi phí tính toán                : {benchmark_report['gpu_hours_per_course']} GPU-giờ/khóa")
    print(f" - Đỉnh VRAM tiêu thụ (trên 1 T4)   : {benchmark_report['peak_vram_gb_single_gpu']} GB (Tổng các GPU: {peak_vram_sum_gb} GB)")
    print(f" - Ước tính cho 3.769 khóa học      : {benchmark_report['estimated_gpu_hours_3769_courses']} GPU-giờ T4")
    print(f" - Ước tính 3.969 lượt (gồm +200)   : {benchmark_report['estimated_gpu_hours_3969_total_runs']} GPU-giờ T4")
    print(f" - Thời gian chạy song song 2 acc   : ~{benchmark_report['estimated_wall_hours_on_2_kaggle_accounts']} giờ/tài khoản")
    print(f" - Thống kê nhãn (inferred)         : {label_counts}")
    if encode_info is not None:
        print("---------------------------------------------------------------------")
        print(f" - Bộ mã hóa văn bản đóng băng      : {encode_info['encoder_name']} (dim={encode_info['embedding_dim']})")
        print(f" - File Embedding course_text gốc   : {encode_info['raw_course_text_stats']['npy_path']}")
        print(f" - File Embedding hồ sơ LLM 5 chiều : {encode_info['llm_profile_text_stats']['npy_path']}")
        print(f" - Cosine Sim TB (Gốc vs. LLM)      : {encode_info['raw_vs_llm_cosine_similarity']['mean']} ± {encode_info['raw_vs_llm_cosine_similarity']['std']}")
        print(f" - Cosine Sim TB (Tên khóa vs. LLM) : {encode_info['name_vs_llm_cosine_similarity']['mean']} ± {encode_info['name_vs_llm_cosine_similarity']['std']}")
        print(f" - File 1% khóa lệch tên thấp nhất  : {encode_info['name_vs_llm_cosine_similarity']['lowest_1pct_path']}")
    print(f" - File hồ sơ JSONL (resume)        : {jsonl_path}")
    print(f" - File hồ sơ JSON                  : {profiles_path}")
    print(f" - File báo cáo thông lượng         : {report_path}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
