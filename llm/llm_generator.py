
import os
import re
import json
import time
import torch

from text_utils import (
    REQUIRED_JSON_KEYS,
    clean_text_field,
    clean_concepts,
    norm_prereq,
    get_clean_raw_course_text,
    get_school_variants,
    check_schema_constraints,
    check_name_mentions,
    scrub_name_mentions,
    classify_prereq_label,
    normalize_profile,
    build_profile_text
)

INFERRED_HINT = (
    "\n\nNOTE: The prerequisite field of this course is 'not stated' in the source metadata. "
    "Do NOT answer 'none' or 'not stated' for prerequisite_knowledge. Infer the required knowledge/skills "
    "conservatively and begin the prerequisite_knowledge value with the tag '(inferred)'."
)


# Lấy commit SHA (model_revision) phục vụ tái lập thực nghiệm
def get_model_revision(model_name, model=None, revision=None):
    try:
        from huggingface_hub import model_info
        info = model_info(model_name, revision=revision)
        if getattr(info, "sha", None):
            return info.sha
    except Exception:
        pass

    if model is not None:
        rev = getattr(getattr(model, "config", None), "_commit_hash", None)
        if rev and rev != "unknown":
            return rev
        name_or_path = getattr(getattr(model, "config", None), "_name_or_path", "")
        m = re.search(r"snapshots/([0-9a-f]{40})", str(name_or_path))
        if m:
            return m.group(1)

    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision

    try:
        cache_root = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")
        )
        folder_name = "models--" + model_name.replace("/", "--")
        snap_dir = os.path.join(cache_root, folder_name, "snapshots")
        if os.path.isdir(snap_dir):
            snaps = [d for d in os.listdir(snap_dir) if re.fullmatch(r"[0-9a-f]{40}", d)]
            if snaps:
                return sorted(snaps)[-1]
    except Exception:
        pass

    return "unknown"


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

    forbidden_list = [v_text for v_text, _ in get_school_variants(school)]
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
            f"NEVER write any of these exact names or abbreviations in your JSON output: {', '.join(forbidden_list)}."
        )

    if inferred_hint and prereq == "not stated":
        user_content += INFERRED_HINT

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

    if retry_idx > 0:
        fix_prompt = (
            f"Attempt {retry_idx} correction: Your previous output violated schema constraints ({retry_reason}). "
            "Rewrite and return ONLY the corrected valid JSON object with the exact 5 keys: "
            "\"overview\" (<= 60 words), \"target_learners\" (<= 30 words), "
            "\"learning_outcomes\" (list of 3-5 items), \"prerequisite_knowledge\" (list of 2-5 items), "
            "\"follow_up_topics\" (list of 2-5 items). "
            "Remove any mention of university or instructor names (including abbreviations) completely."
        )
        if prev_output:
            messages.append({"role": "assistant", "content": prev_output})
            messages.append({"role": "user", "content": fix_prompt})
        else:
            messages[-1]["content"] += "\n\n" + fix_prompt

    return messages, user_content


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
                do_sample=False,
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
        "mentioned_entities": name_mentions,
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
