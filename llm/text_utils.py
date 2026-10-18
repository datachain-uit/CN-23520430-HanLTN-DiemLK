
import os
import re
import json
import numpy as np
import pandas as pd

NEG_PREREQ_REGEX = re.compile(r"^(none|no|not required|not needed|nothing|n/a)\b", re.I)
NEG_PREREQ_OUTPUT_REGEX = re.compile(r"^\W*(none|no|not|n/a|nothing|unknown|unspecified)\b", re.I)
TAG = re.compile(r"\s*\(\s*inferred\s*\)\s*:?\s*", re.I)

# Biểu thức phát hiện mô tả chứa gợi ý tiên quyết 
PREREQ_HINT_IN_ABOUT_REGEX = re.compile(
    r"(prerequisit|required|basic knowledge of|should have)", re.I
)

# Biểu thức phát hiện tên khóa học dạng thử nghiệm/giữ chỗ 
PLACEHOLDER_NAME_REGEX = re.compile(
    r"(\btest\s+test\b|\bseries\s+course\s+test\b|\babandoned\b|\bclass\s+has\s+not\s+started\s+yet\b|^test\b)",
    re.I
)

REQUIRED_JSON_KEYS = [
    "overview",
    "target_learners",
    "learning_outcomes",
    "prerequisite_knowledge",
    "follow_up_topics"
]

MAX_OVERVIEW_WORDS = 60
MAX_LEARNER_WORDS = 30
MIN_OUTCOMES, MAX_OUTCOMES = 3, 5
MIN_LIST_ITEMS, MAX_LIST_ITEMS = 2, 5

# Bảng ánh xạ tên viết tắt và dạng rút gọn các trường đại học 
SCHOOL_ACRONYM_MAP = {
    "massachusetts institute of technology": ["MIT"],
    "tsinghua university": ["THU", "Tsinghua"],
    "peking university": ["PKU", "Peking"],
    "fudan university": ["FDU", "Fudan"],
    "zhejiang university": ["ZJU", "Zhejiang"],
    "shanghai jiao tong university": ["SJTU", "Jiao Tong"],
    "nanjing university": ["NJU", "Nanjing"],
    "university of science and technology of china": ["USTC"],
    "harbin institute of technology": ["HIT", "Harbin"],
    "wuhan university": ["WHU", "Wuhan"],
    "huazhong university of science and technology": ["HUST", "Huazhong"],
    "xi'an jiaotong university": ["XJTU", "Xi'an Jiaotong"],
    "beijing normal university": ["BNU"],
    "renmin university of china": ["RUC", "Renmin"],
    "beihang university": ["BUAA", "Beihang"],
    "beijing institute of technology": ["BIT"],
    "sichuan university": ["SCU", "Sichuan"],
    "sun yat-sen university": ["SYSU", "Sun Yat-sen"],
    "nankai university": ["NKU", "Nankai"],
    "tianjin university": ["TJU", "Tianjin"],
    "southeast university": ["SEU"],
    "east china normal university": ["ECNU"],
    "the chinese university of hong kong": ["CUHK"],
    "the hong kong university of science and technology": ["HKUST"],
    "national taiwan university": ["NTU"],
    "xiamen university": ["XMU", "Xiamen"],
    "shandong university": ["SDU", "Shandong"],
    "jilin university": ["JLU", "Jilin"],
    "central south university": ["CSU"],
    "dalian university of technology": ["DUT", "Dalian"],
    "chongqing university": ["CQU", "Chongqing"],
    "lanzhou university": ["LZU", "Lanzhou"],
    "tongji university": ["Tongji"],
}

GENERIC_SCHOOL_WORDS = {
    "university", "institute", "technology", "college", "school", "academy",
    "national", "international", "central", "normal", "china", "chinese",
    "science", "sciences", "engineering", "medical", "medicine", "foreign",
    "studies", "finance", "economics", "business", "political", "law",
    "agricultural", "agriculture", "forestry", "mining", "petroleum",
    "communications", "transportation", "aeronautics", "astronautics",
    "posts", "telecommunications", "electric", "power", "electronic",
    "information", "language", "languages", "culture", "media", "art", "arts",
    "music", "sport", "sports", "physical", "education", "teachers", "teacher",
    "union", "open", "polytechnic", "technical", "vocational", "public",
    "state", "federal", "royal", "imperial", "metropolitan", "city",
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest", "of", "the", "and", "for", "in", "at"
}


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


def as_list(x):
    if is_empty_val(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i).strip() for i in x if not is_empty_val(i)]
    return [s.strip() for s in re.split(r"[;,]", str(x)) if s.strip()]


def normalize_profile(p):
    p = dict(p)
    pre = as_list(p.get("prerequisite_knowledge"))
    inferred = bool(pre) and any(bool(TAG.search(x)) for x in pre)
    p["prerequisite_knowledge"] = [
        TAG.sub(" ", x).strip(" ,;:-") for x in pre if TAG.sub(" ", x).strip(" ,;:-")
    ]
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


def clean_text_field(x, default_val="unknown"):
    if is_empty_val(x):
        return default_val
    if isinstance(x, (list, tuple, np.ndarray)):
        items = [str(i).strip() for i in x if not is_empty_val(i)]
        unique_items = list(dict.fromkeys(items))
        return ", ".join(unique_items) if unique_items else default_val
    s = str(x).strip()
    return s if s else default_val


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


def get_clean_raw_course_text(row):
    raw_txt = clean_text_field(row.get("course_text"), default_val="")
    if raw_txt:
        return raw_txt
    name = clean_text_field(row.get("course_name"), default_val="unknown")
    school = clean_text_field(row.get("school_name"), default_val="unknown")
    teacher = clean_text_field(row.get("teacher_name"), default_val="unknown")
    about = clean_text_field(row.get("course_about"), default_val="not stated")
    return f"Course: {name}. University: {school}. Instructor: {teacher}. Description: {about}"


def field_to_text(x, sep="; "):
    if is_empty_val(x):
        return ""
    if isinstance(x, (list, tuple, np.ndarray)):
        return sep.join(str(i).strip() for i in x if not is_empty_val(i))
    return str(x).strip()


def word_count(x):
    return len(re.findall(r"\S+", field_to_text(x)))


def get_school_variants(school_str):
    if not school_str or school_str.lower() == "unknown":
        return []
    variants = []
    for s in re.split(r"[,;|]", school_str):
        s_clean = s.strip()
        if len(s_clean) < 3:
            continue
        variants.append((s_clean, False))
        s_lower = s_clean.lower()

        for full_k, abbr_list in SCHOOL_ACRONYM_MAP.items():
            if full_k in s_lower:
                for abbr in abbr_list:
                    variants.append((abbr, abbr.isupper()))

        tokens = [w for w in re.findall(r"[A-Za-z]+", s_clean) if w.lower() not in GENERIC_SCHOOL_WORDS]
        if len(tokens) == 1 and len(tokens[0]) >= 5:
            variants.append((tokens[0], False))

    seen = set()
    unique_variants = []
    for text_v, is_acro in variants:
        key = (text_v.lower() if not is_acro else text_v, is_acro)
        if key not in seen:
            seen.add(key)
            unique_variants.append((text_v, is_acro))
    return unique_variants


def check_name_mentions(profile, row):
    if not profile or row is None:
        return []
    combined_text_raw = " ".join(field_to_text(profile.get(k)) for k in REQUIRED_JSON_KEYS)
    combined_text_lower = combined_text_raw.lower()
    mentions = []

    school_str = clean_text_field(row.get("school_name"), default_val="")
    for v_text, is_acro in get_school_variants(school_str):
        if is_acro:
            if re.search(r"\b" + re.escape(v_text) + r"\b", combined_text_raw):
                mentions.append(f"school '{v_text}'")
        else:
            if re.search(r"\b" + re.escape(v_text.lower()) + r"\b", combined_text_lower):
                mentions.append(f"school '{v_text}'")

    teacher_str = clean_text_field(row.get("teacher_name"), default_val="")
    if teacher_str and teacher_str.lower() != "unknown":
        for t in re.split(r"[,;|]", teacher_str):
            t_clean = t.strip()
            if len(t_clean) >= 4 and re.search(r"\b" + re.escape(t_clean.lower()) + r"\b", combined_text_lower):
                mentions.append(f"instructor '{t_clean}'")

    return list(dict.fromkeys(mentions))


def scrub_name_mentions(profile, row):
    if not profile or row is None:
        return profile
    p = dict(profile)
    school_str = clean_text_field(row.get("school_name"), default_val="")
    school_variants = get_school_variants(school_str)

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
        for s_text, is_acro in sorted(school_variants, key=lambda x: len(x[0]), reverse=True):
            flags = 0 if is_acro else re.I
            out = re.sub(r"\b(?:at|from|by|of|in)\s+(?:the\s+)?" + re.escape(s_text) + r"\b", "", out, flags=flags)
            out = re.sub(r"\b" + re.escape(s_text) + r"\b", "the university", out, flags=flags)
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

    pre_list = as_list(profile.get("prerequisite_knowledge"))
    if not (MIN_LIST_ITEMS <= len(pre_list) <= MAX_LIST_ITEMS):
        violations.append(f"prerequisite_knowledge has {len(pre_list)} items (required {MIN_LIST_ITEMS}-{MAX_LIST_ITEMS})")

    fol_list = as_list(profile.get("follow_up_topics"))
    if not (MIN_LIST_ITEMS <= len(fol_list) <= MAX_LIST_ITEMS):
        violations.append(f"follow_up_topics has {len(fol_list)} items (required {MIN_LIST_ITEMS}-{MAX_LIST_ITEMS})")

    name_mentions = check_name_mentions(profile, row)
    if name_mentions:
        violations.append(f"mentions forbidden entity ({', '.join(name_mentions)})")

    return violations


def classify_prereq_label(prereq_norm, profile):
    if prereq_norm != "not stated" or profile is None:
        return "not_applicable"
    txt = field_to_text(profile.get("prerequisite_knowledge"))
    if "(inferred)" in txt.lower():
        return "inferred_labeled"
    if NEG_PREREQ_OUTPUT_REGEX.match(txt):
        return "declared_none_or_unstated"
    return "unlabeled_content"


def build_profile_text(profile, fallback_text="", include_prereq_and_followup=True):
    if not profile:
        return fallback_text
    parts = [
        ("Overview", field_to_text(profile.get("overview"))),
        ("Target learners", field_to_text(profile.get("target_learners"))),
        ("Learning outcomes", field_to_text(profile.get("learning_outcomes"))),
    ]
    if include_prereq_and_followup:
        parts.extend([
            ("Prerequisite knowledge", field_to_text(profile.get("prerequisite_knowledge"))),
            ("Follow-up topics", field_to_text(profile.get("follow_up_topics"))),
        ])
    joined = " ".join(
        f"{k}: {v}." if not v.endswith((".", "!", "?")) else f"{k}: {v}"
        for k, v in parts if v
    ).strip()
    return joined if joined else fallback_text


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


# Nhóm khóa trùng tên hoặc trùng mô tả 
def dup_group_id(df):
    parent = {c: c for c in df["course_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in ("course_name", "course_about"):
        key = df[col].fillna("").astype(str).str.strip().str.lower()
        for k, ids in df.groupby(key)["course_id"]:
            if not k:
                continue  # bỏ qua mô tả rỗng
            ids = list(ids)
            for x in ids[1:]:
                parent[find(x)] = find(ids[0])
    return {c: find(c) for c in df["course_id"]}


def compute_and_save_dup_groups(df, output_dir):
    df_work = df.copy()
    if "course_id" not in df_work.columns and "id" in df_work.columns:
        df_work["course_id"] = df_work["id"].astype(str)
    else:
        df_work["course_id"] = df_work["course_id"].astype(str)

    gid_map = dup_group_id(df_work)

    about_series = df_work["course_about"].fillna("").astype(str).str.strip().str.lower()
    non_empty_about = about_series[about_series != ""]
    about_vc = non_empty_about.value_counts()
    shared_about_count = int((about_vc > 1).sum())
    courses_sharing_about = int(about_vc[about_vc > 1].sum())

    name_series = df_work["course_name"].fillna("").astype(str).str.strip()
    placeholder_mask = name_series.apply(lambda x: bool(PLACEHOLDER_NAME_REGEX.search(x)))
    num_placeholder_courses = int(placeholder_mask.sum())

    dup_df = pd.DataFrame({
        "course_index": list(range(len(df_work))),
        "course_id": df_work["course_id"].tolist(),
        "dup_group_id": [gid_map[c] for c in df_work["course_id"]],
        "course_name_norm": name_series.str.lower().tolist(),
        "is_placeholder_course": placeholder_mask.tolist()
    })

    dup_parquet_path = os.path.join(output_dir, "course_dup_groups.parquet")
    dup_df.to_parquet(dup_parquet_path, index=False)

    num_unique_groups = int(dup_df["dup_group_id"].nunique())
    group_sizes = dup_df["dup_group_id"].value_counts()
    courses_in_dup_groups = int(group_sizes[group_sizes > 1].sum())

    summary = {
        "total_courses": len(df_work),
        "unique_dup_groups": num_unique_groups,
        "courses_in_multi_member_dup_groups": courses_in_dup_groups,
        "shared_non_empty_descriptions_count": shared_about_count,
        "courses_sharing_description": courses_sharing_about,
        "courses_sharing_description_pct": round(100.0 * courses_sharing_about / max(len(df_work), 1), 2),
        "placeholder_or_test_courses_count": num_placeholder_courses,
        "dup_groups_parquet_path": dup_parquet_path
    }
    print(
        f"-> [dup_group_id] Số nhóm phân biệt: {num_unique_groups}/{len(df_work)} | "
        f"Mô tả dùng chung: {shared_about_count} đoạn ({courses_sharing_about} khóa, "
        f"{summary['courses_sharing_description_pct']}%) | "
        f"Khóa tên giữ chỗ/test: {num_placeholder_courses} | Đã lưu: {dup_parquet_path}"
    )
    return gid_map, summary
