import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT)).expanduser().resolve()

ITEM_IDX_PATH = str(DATA_DIR / "data/embeddings/01_lightgcn_backbone_cold/item_index.json")
USER_IDX_PATH = str(DATA_DIR / "data/embeddings/01_lightgcn_backbone_cold/user_index_cold.json")
GU_PATH = str(DATA_DIR / "data/embeddings/01_lightgcn_backbone_cold/g_u.npy")
GI_PATH = str(DATA_DIR / "data/embeddings/01_lightgcn_backbone_cold/g_i.npy")

RAW_IDX_PATH = str(DATA_DIR / "data/course_llm/1_embeddings/raw_embeddings_course_index.json")
LLM_IDX_PATH = str(DATA_DIR / "data/course_llm/1_embeddings/embeddings_course_index.json")
RAW_NPY = str(DATA_DIR / "data/course_llm/1_embeddings/raw_course_text_embeddings.npy")
LLM_NPY = str(DATA_DIR / "data/course_llm/1_embeddings/llm_profile_embeddings.npy")
LLM_3D_NPY = str(DATA_DIR / "data/course_llm/1_embeddings/llm_profile_no_prereq_followup_embeddings.npy")

COLD_VAL_JSON = str(DATA_DIR / "data/splits/cold_scenario/cold_items_val.json")
COLD_TEST_JSON = str(DATA_DIR / "data/splits/cold_scenario/cold_items_test.json")
TRAIN_PARQUET = str(DATA_DIR / "data/splits/cold_scenario/train_cold.parquet")
VAL_PARQUET = str(DATA_DIR / "data/splits/cold_scenario/val_cold.parquet")
TEST_PARQUET = str(DATA_DIR / "data/splits/cold_scenario/test_cold.parquet")

OUTPUT_DIR = str(DATA_DIR / "ColdRec_Data")


def find_file_in_repository(filename, default_path=None):
    """Tìm một tệp đầu vào theo tên trong bản sao repository cục bộ."""
    if default_path and os.path.exists(default_path):
        return default_path
    for base_dir in (str(DATA_DIR), "."):
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Không tìm thấy '{filename}' bên dưới DATA_DIR={DATA_DIR}."
    )


# Đường dẫn tệp phân nhóm trùng lặp P-RQ3 (tự động tìm nếu đặt ở thư mục khác)
DUP_GROUPS_PARQUET = find_file_in_repository(
    "course_dup_groups.parquet",
    default_path=str(DATA_DIR / "data/course_llm/course_dup_groups.parquet"),
)

# Nạp bộ đánh giá của repository.
sys.path.insert(0, str(DATA_DIR / "src/eval"))
try:
    from evaluator import evaluate, topk_random_ties
except FileNotFoundError:
    eval_file_path = find_file_in_repository("evaluate.py")
    sys.path.insert(0, os.path.dirname(eval_file_path))
    from evaluate import evaluate, topk_random_ties


def md5_hash(filepath):
    """Tính mã băm MD5 của tệp để truy vết phiên bản"""
    return hashlib.md5(open(filepath, "rb").read()).hexdigest()


def to_bool_mask(ids, c2i, n_items=3769):
    """Chuyển danh sách course_id sang mặt nạ bool dài 3.769"""
    m = torch.zeros(n_items, dtype=torch.bool)
    m[[c2i[str(c)] for c in ids]] = True
    return m


def build_hist(df, u2i, c2i):
    """Dựng từ điển lịch sử tương tác {u_idx: [i_idx, ...]}."""
    u_mapped = df["user_id"].astype(str).map(u2i)
    i_mapped = df["course_id"].astype(str).map(c2i)
    temp = pd.DataFrame({"u": u_mapped, "i": i_mapped}).dropna()
    temp["u"] = temp["u"].astype(int)
    temp["i"] = temp["i"].astype(int)
    return temp.groupby("u")["i"].apply(list).to_dict()


def _extract_id_list(json_path, key_name):
    data = json.load(open(json_path))
    items = data[key_name] if isinstance(data, dict) else data
    return [str(x) for x in items]


# Lớp bộ chuyển đổi dữ liệu và kiểm tra giao thức
class ColdRecDataAdapter:

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_and_verify_all()

    def _load_and_verify_all(self):
        print("=== KIỂM TRA MD5 VÀ BẢNG ÁNH XẠ CHỈ SỐ CHUNG ===")
        # Đọc và đối chiếu bảng ánh xạ chỉ số chung 
        self.course_ids = _extract_id_list(ITEM_IDX_PATH, "course_id")
        self.users = _extract_id_list(USER_IDX_PATH, "user_id")

        assert len(self.course_ids) == 3769 and self.course_ids == sorted(self.course_ids), "Lỗi bảng item_index.json!"
        assert len(self.users) == 98220 and self.users == sorted(self.users), "Lỗi bảng user_index_cold.json!"

        if os.path.exists(RAW_IDX_PATH):
            raw_cids = _extract_id_list(RAW_IDX_PATH, "course_id")
            assert self.course_ids == raw_cids, "item_index.json không khớp raw_embeddings_course_index.json!"
        if os.path.exists(LLM_IDX_PATH):
            llm_cids = _extract_id_list(LLM_IDX_PATH, "course_id")
            assert self.course_ids == llm_cids, "item_index.json không khớp embeddings_course_index.json!"

        self.item2idx = {c: i for i, c in enumerate(self.course_ids)}
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.n_users = len(self.users)
        self.n_items = len(self.course_ids)

        # Đọc danh sách khóa lạnh (251 cold-val, 502 cold-test)
        self.cold_val_ids = [str(c) for c in json.load(open(COLD_VAL_JSON))]
        self.cold_test_ids = [str(c) for c in json.load(open(COLD_TEST_JSON))]
        assert len(self.cold_val_ids) == 251 and len(self.cold_test_ids) == 502

        self.cold_val_idx = sorted([self.item2idx[c] for c in self.cold_val_ids])
        self.cold_test_idx = sorted([self.item2idx[c] for c in self.cold_test_ids])
        self.cold_idx = np.array(sorted(self.cold_val_idx + self.cold_test_idx), dtype=np.int64)
        assert len(self.cold_idx) == 753

        # Đọc các tệp tương tác và chạy assert chống rò rỉ
        self.tr_cold = pd.read_parquet(TRAIN_PARQUET)
        self.va_cold = pd.read_parquet(VAL_PARQUET)
        self.te_cold = pd.read_parquet(TEST_PARQUET)

        tr_items = set(self.tr_cold["course_id"].astype(str))
        cv, ct = set(self.cold_val_ids), set(self.cold_test_ids)
        assert not (cv & tr_items) and not (ct & tr_items) and not (cv & ct), "Rò rỉ khóa lạnh vào train hoặc trùng val/test!"
        assert set(self.va_cold["course_id"].astype(str)) <= cv
        assert set(self.te_cold["course_id"].astype(str)) <= ct
        assert set(self.va_cold["user_id"].astype(str)) <= set(self.tr_cold["user_id"].astype(str))
        assert set(self.te_cold["user_id"].astype(str)) <= set(self.tr_cold["user_id"].astype(str))

        # Xác định đúng 2.401 khóa ấm có tương tác trong train_cold
        self.warm_active_idx = np.unique(
            self.tr_cold["course_id"]
            .astype(str)
            .map(self.item2idx)
            .to_numpy(dtype=np.int64)
        )
        assert len(self.warm_active_idx) == 2401 and not (set(self.warm_active_idx.tolist()) & set(self.cold_idx.tolist()))

        # Đọc và kiểm tra Backbone LightGCN (g_u, g_i)
        self.g_u = np.load(GU_PATH).astype(np.float32)
        self.g_i = np.load(GI_PATH).astype(np.float32)

        gu_md5, gi_md5 = md5_hash(GU_PATH), md5_hash(GI_PATH)
        print(f"MD5 g_u.npy: {gu_md5} (Kỳ vọng: 7a170558...)")
        print(f"MD5 g_i.npy: {gi_md5} (Kỳ vọng: d17ad9df...)")

        assert self.g_u.shape == (98220, 64) and self.g_i.shape == (3769, 64)
        assert np.allclose(self.g_i[self.cold_idx], 0.0), "Khóa lạnh (753 khóa) phải có g_i = 0!"

        # Đảm bảo 615 khóa ấm không có tương tác trong train_cold cũng có g_i = 0
        inactive_mask = np.ones(self.n_items, dtype=bool)
        inactive_mask[self.warm_active_idx] = False
        assert np.allclose(self.g_i[inactive_mask], 0.0), "615 khóa ấm không tương tác phải có g_i = 0!"

        # Đọc và chuẩn hóa L2 cho các biến thể đặc trưng nội dung (raw, llm, fused, llm_3d, fused_3d)
        print(f"MD5 raw_emb: {md5_hash(RAW_NPY)} | MD5 llm_emb: {md5_hash(LLM_NPY)}")

        nz = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
        self.S_raw = nz(np.load(RAW_NPY).astype(np.float32))
        self.S_llm = nz(np.load(LLM_NPY).astype(np.float32))
        assert self.S_raw.shape == (3769, 384) and self.S_llm.shape == (3769,384,)
        self.S_fused = np.concatenate([self.S_raw, self.S_llm], axis=1).astype(np.float32)  # [3769, 768]

        self.contents = {
            "raw": self.S_raw,
            "llm": self.S_llm,
            "fused": self.S_fused,
        }

        if os.path.exists(LLM_3D_NPY):
            print(f"MD5 llm_3d_emb: {md5_hash(LLM_3D_NPY)}")
            self.S_llm_3d = nz(np.load(LLM_3D_NPY).astype(np.float32))
            assert self.S_llm_3d.shape == (3769, 384)
            self.S_fused_3d = np.concatenate(
                [self.S_raw, self.S_llm_3d], axis=1
            ).astype(np.float32)
            self.contents["llm_3d"] = self.S_llm_3d
            self.contents["fused_3d"] = self.S_fused_3d

        # Dựng từ điển mặt nạ lịch sử, đích đánh giá và mặt nạ ứng viên (a), (b)
        self.hist_tr = build_hist(self.tr_cold, self.user2idx, self.item2idx)
        self.hist_trva = build_hist(
            pd.concat([self.tr_cold, self.va_cold], ignore_index=True),
            self.user2idx,
            self.item2idx,
        )
        self.val_target_dict = build_hist(
            self.va_cold, self.user2idx, self.item2idx
        )
        self.test_target_dict = build_hist(
            self.te_cold, self.user2idx, self.item2idx
        )
        assert (
            len(self.val_target_dict) == 2092
            and len(self.test_target_dict) == 3340
        )

        self.cand_val_a = to_bool_mask(
            self.cold_val_ids, self.item2idx, self.n_items
        )
        self.cand_test_a = to_bool_mask(
            self.cold_test_ids, self.item2idx, self.n_items
        )
        self.cand_test_b = torch.ones(self.n_items, dtype=torch.bool)

        # Phân nhóm P-RQ3 ("trung" vs "moi") từ course_dup_groups.parquet
        dup_df = pd.read_parquet(DUP_GROUPS_PARQUET)
        c2dup = dict(zip(dup_df["course_id"].astype(str), dup_df["dup_group_id"]))
        active_warm_course_ids = [self.course_ids[i] for i in self.warm_active_idx]
        warm_active_dup_groups = {c2dup[c] for c in active_warm_course_ids}

        self.cold_test_dup_ids = [c for c in self.cold_test_ids if c2dup[c] in warm_active_dup_groups]
        self.cold_test_new_ids = [
            c
            for c in self.cold_test_ids
            if c2dup[c] not in warm_active_dup_groups
        ]
        assert (
            len(self.cold_test_dup_ids) == 124
            and len(self.cold_test_new_ids) == 378
        ), (
            f"Sai số lượng khóa trùng/mới: {len(self.cold_test_dup_ids)} /"
            f" {len(self.cold_test_new_ids)}"
        )

        self.groups_prq3 = {
            "trung": {self.item2idx[c] for c in self.cold_test_dup_ids},
            "moi": {self.item2idx[c] for c in self.cold_test_new_ids},
        }
        print("=== HOÀN TẤT KIỂM TRA: MỌI ASSERT ĐỀU ĐẠT CHUẨN! ===\n")

    def export_coldrec_files(self, dataset_name="MOOCCubeX", content_variant="raw"):
        """Xuất bộ dữ liệu sang đúng cấu trúc thư mục mà ColdRec (ALDI, CGRC, USIM)
        và mã chính thức zfnWong/ALDI yêu cầu, giữ nguyên chỉ số 0..98219 và
        0..3768.
        """
        # OUTPUT_DIR đã trỏ tới <repo>/ColdRec_Data. Tránh tạo thư mục lồng
        # <repo>/ColdRec_Data/ColdRec_Data/<dataset> khiến notebook ColdRec
        # phía sau không tìm thấy cold_item/.
        ds_dir = os.path.join(self.output_dir, dataset_name)
        cold_item_dir = os.path.join(ds_dir, "cold_item")
        os.makedirs(ds_dir, exist_ok=True)
        os.makedirs(cold_item_dir, exist_ok=True)

        # Ánh xạ các bảng tương tác sang chỉ số nguyên (user, item)
        def to_mapped_df(df):
            u = df["user_id"].astype(str).map(self.user2idx).astype(np.int64)
            i = df["course_id"].astype(str).map(self.item2idx).astype(np.int64)
            return pd.DataFrame({"user": u, "item": i})

        tr_df = to_mapped_df(self.tr_cold)
        va_df = to_mapped_df(self.va_cold)
        te_df = to_mapped_df(self.te_cold)
        cold_all_df = pd.concat([va_df, te_df], ignore_index=True)

        # Tạo warm_val, warm_test, overall_val, overall_test theo đúng cấu trúc ColdRec/data/convert.py
        # (lấy 1 tương tác ấm/học viên từ tập train để các hàm đánh giá warm/overall nội bộ của ColdRec không lỗi chia 0)
        warm_va_df = tr_df[tr_df["user"].isin(set(va_df["user"]))].drop_duplicates(
            subset=["user"], keep="first"
        )
        warm_te_df = tr_df[tr_df["user"].isin(set(te_df["user"]))].drop_duplicates(
            subset=["user"], keep="first"
        )
        overall_va_df = pd.concat([va_df, warm_va_df], ignore_index=True)
        overall_te_df = pd.concat([te_df, warm_te_df], ignore_index=True)

        # Lưu đủ 8 tệp CSV vào cả ds_dir và ds_dir/cold_item/
        csv_files = {
            "warm_train.csv": tr_df,
            "cold_item_val.csv": va_df,
            "cold_item_test.csv": te_df,
            "cold_item.csv": cold_all_df,
            "warm_val.csv": warm_va_df,
            "warm_test.csv": warm_te_df,
            "overall_val.csv": overall_va_df,
            "overall_test.csv": overall_te_df,
        }
        for fname, df_out in csv_files.items():
            df_out.to_csv(os.path.join(ds_dir, fname), index=False)
            df_out.to_csv(os.path.join(cold_item_dir, fname), index=False)

        # Lưu danh sách ID khóa ấm có tương tác (2.401 khóa) và khóa lạnh
        np.save(os.path.join(ds_dir, "warm_item_ids.npy"), self.warm_active_idx)
        np.save(os.path.join(ds_dir, "cold_val_item_ids.npy"),np.array(self.cold_val_idx, dtype=np.int64),)
        np.save(os.path.join(ds_dir, "cold_test_item_ids.npy"),np.array(self.cold_test_idx, dtype=np.int64),)
        np.save(os.path.join(ds_dir, "cold_item_ids.npy"), self.cold_idx)

        # Lưu Pre-trained Teacher Embeddings (LightGCN nền chung) dạng .npy và .pt
        warm_emb_concat = np.concatenate([self.g_u, self.g_i], axis=0)  # [98220 + 3769, 64]
        np.save(os.path.join(ds_dir, f"{dataset_name}_lightgcn.npy"), warm_emb_concat)
        np.save(os.path.join(ds_dir, "warm_emb.npy"), warm_emb_concat)
        np.save(os.path.join(ds_dir, "g_u.npy"), self.g_u)
        np.save(os.path.join(ds_dir, "g_i.npy"), self.g_i)

        torch.save(torch.from_numpy(self.g_u), os.path.join(ds_dir, "g_u_tensor.pt"))
        torch.save(torch.from_numpy(self.g_i), os.path.join(ds_dir, "g_i_tensor.pt"))
        torch.save({"user_emb": torch.from_numpy(self.g_u), "item_emb": torch.from_numpy(self.g_i)},os.path.join(ds_dir, "warm_emb_dict.pt"),)

        # Lưu ma trận đặc trưng nội dung khóa học theo biến thể được chọn ('raw', 'llm', 'fused', ...)
        content_mat = self.contents[content_variant]
        np.save(os.path.join(ds_dir, f"{dataset_name}_item_content.npy"), content_mat)
        np.save(os.path.join(ds_dir, "item_content.npy"), content_mat)

        # Chuẩn bị các mảng np.int32 theo đúng yêu cầu của ColdRec/main.py và ColdRec/data/convert.py
        warm_user_arr = np.array(sorted(set(tr_df["user"].tolist())), dtype=np.int32)
        warm_item_arr = np.array(self.warm_active_idx, dtype=np.int32)
        cold_user_arr = np.array(sorted(set(cold_all_df["user"].tolist())), dtype=np.int32)
        cold_item_arr = np.array(self.cold_idx, dtype=np.int32)

        info_dict = {
            # Các khóa bắt buộc cho ColdRec/main.py và ColdRec/data/convert.py
            "user_num": self.n_users,
            "item_num": self.n_items,
            "user_array": np.arange(self.n_users, dtype=np.int32),
            "item_array": np.arange(self.n_items, dtype=np.int32),
            "warm_user": warm_user_arr,
            "warm_item": warm_item_arr,
            "cold_user": cold_user_arr,
            "cold_item": cold_item_arr,
            "train_user": warm_user_arr,
            "warm_val_user": np.array(
                sorted(set(warm_va_df["user"].tolist())), dtype=np.int32
            ),
            "warm_test_user": np.array(
                sorted(set(warm_te_df["user"].tolist())), dtype=np.int32
            ),
            "cold_val_user": np.array(
                sorted(set(va_df["user"].tolist())), dtype=np.int32
            ),
            "cold_test_user": np.array(
                sorted(set(te_df["user"].tolist())), dtype=np.int32
            ),
            "overall_val_user": np.array(
                sorted(set(overall_va_df["user"].tolist())), dtype=np.int32
            ),
            "overall_test_user": np.array(
                sorted(set(overall_te_df["user"].tolist())), dtype=np.int32
            ),
            "train_item": warm_item_arr,
            "warm_val_item": np.array(
                sorted(set(warm_va_df["item"].tolist())), dtype=np.int32
            ),
            "warm_test_item": np.array(
                sorted(set(warm_te_df["item"].tolist())), dtype=np.int32
            ),
            "cold_val_item": np.array(self.cold_val_idx, dtype=np.int32),
            "cold_test_item": np.array(self.cold_test_idx, dtype=np.int32),
            "overall_val_item": np.array(
                sorted(set(overall_va_df["item"].tolist())), dtype=np.int32
            ),
            "overall_test_item": np.array(
                sorted(set(overall_te_df["item"].tolist())), dtype=np.int32
            ),
            # Các khóa bổ sung cho adapter nội bộ / ALDI
            "warm_items": set(self.warm_active_idx.tolist()),
            "cold_val_items": set(self.cold_val_idx),
            "cold_test_items": set(self.cold_test_idx),
            "cold_items": set(self.cold_idx.tolist()),
            "emb_dim": 64,
            "content_dim": int(content_mat.shape[1]),
            "user_his": self.hist_tr,
            "val_user_his": self.hist_trva,
        }

        # Lưu cả info_dict.pkl, para_dict.pkl và n_user_item.pkl ở cả 2 thư mục
        for target_d in (ds_dir, cold_item_dir):
            with open(os.path.join(target_d, "info_dict.pkl"), "wb") as f:
                pickle.dump(info_dict, f, protocol=4)
            with open(os.path.join(target_d, "para_dict.pkl"), "wb") as f:
                pickle.dump(info_dict, f, protocol=4)
            with open(os.path.join(target_d, "n_user_item.pkl"), "wb") as f:
                pickle.dump({"user": self.n_users, "item": self.n_items}, f, protocol=4)

        print(
            f"[OK] Đã xuất dữ liệu ColdRec ({content_variant},"
            f" dim={content_mat.shape[1]}) tại: {ds_dir}"
        )
        return ds_dir


# Các hàm đánh giá của nhóm (tập a, b, P-RQ3, P-RQ4)
@torch.no_grad()
def eval_subgroups(
    score_fn, targets, mask, cand, groups, ks=(5, 10, 20), tie_seed=0, bs=1024
):
    """Hàm tính P-RQ3 đúng chuẩn:

    Xếp hạng trên TOÀN BỘ 502 khóa của tập (a), sau đó chỉ tính chỉ số trên đích
    thuộc từng nhóm.
    """
    users, K = sorted(targets), max(ks)
    g = torch.Generator().manual_seed(tie_seed)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    cand = torch.as_tensor(cand)
    out = {
        n: {f"{m}@{k}": [] for m in ("Recall", "NDCG", "HR") for k in ks}
        for n in groups
    }
    for s in range(0, len(users), bs):
        ub = users[s : s + bs]
        sc = score_fn(torch.tensor(ub)).float().cpu().clone()
        sc[:, ~cand] = -float("inf")
        for r, u in enumerate(ub):
            if mask.get(u):
                sc[r, list(mask[u])] = -float("inf")
        top = topk_random_ties(sc, K, g)
        for r, u in enumerate(ub):
            for n, gset in groups.items():
                gt = [i for i in targets[u] if i in gset]
                if not gt:
                    continue
                hit = np.isin(top[r], gt).astype(float)
                for k in ks:
                    h = hit[:k]
                    idcg = disc[: min(len(gt), k)].sum()
                    out[n][f"Recall@{k}"].append(h.sum() / len(gt))
                    out[n][f"NDCG@{k}"].append((h * disc[:k]).sum() / idcg)
                    out[n][f"HR@{k}"].append(float(h.sum() > 0))
    return {n: {m: np.array(v) for m, v in d.items()} for n, d in out.items()}


@torch.no_grad()
def evaluate_coldrec_embeddings(
    adapter, u_emb, i_emb, model_name, variant_name, seed, device="cuda:0" if torch.cuda.is_available() else "cpu"
):
    """Hàm cầu nối bắt buộc:

    Nhận u_emb [98220, 64] và i_emb [3769, 64] xuất ra từ mô hình ColdRec (ALDI
    / CGRC / USIM), chấm điểm bằng evaluator.py và lưu đầy đủ file phục
    vụ Wilcoxon + P-RQ3.
    """
    u_t = torch.as_tensor(u_emb, dtype=torch.float32, device=device)
    i_t = torch.as_tensor(i_emb, dtype=torch.float32, device=device)
    assert u_t.shape == (98220, 64) and i_t.shape == (3769, 64)

    score_fn = lambda ub: (u_t[ub.to(device)] @ i_t.t()).cpu()

    # Đánh giá trên tập (a) (502 khóa cold-test) và tập (b) (3.769 khóa)
    kw = dict(mask=adapter.hist_trva, ks=(5, 10, 20), tie_seed=seed)
    res_a = evaluate(score_fn, adapter.test_target_dict, cand=adapter.cand_test_a, **kw)
    res_b = evaluate(score_fn, adapter.test_target_dict, cand=adapter.cand_test_b, **kw)

    # Đánh giá P-RQ3 theo nhóm khóa trùng / mới
    res_prq3 = eval_subgroups(
        score_fn,
        adapter.test_target_dict,
        adapter.hist_trva,
        adapter.cand_test_a,
        adapter.groups_prq3,
        ks=(5, 10, 20),
        tie_seed=seed,
    )
    assert (
        len(res_prq3["trung"]["Recall@20"]) == 1028
        and len(res_prq3["moi"]["Recall@20"]) == 2844
    )

    # Lưu biểu diễn người học và khóa học của seed này
    rep_dir = os.path.join(adapter.output_dir, f"results_{model_name}", "item_reps")
    os.makedirs(rep_dir, exist_ok=True)
    np.save(
        os.path.join(rep_dir, f"{model_name}_{variant_name}_seed{seed}.npy"),
        i_t.cpu().numpy(),
    )
    np.save(
        os.path.join(rep_dir, f"u_e_{variant_name}_seed{seed}.npy"),
        u_t.cpu().numpy(),
    )
    np.save(
        os.path.join(rep_dir, f"v_e_{variant_name}_seed{seed}.npy"),
        i_t.cpu().numpy(),
    )

    return res_a, res_b, res_prq3


if __name__ == "__main__":
    # Khởi tạo Adapter, chạy toàn bộ kiểm tra assert và xuất dữ liệu cho cả 5 biến thể (gồm 3D cho E-RQ3)
    adapter = ColdRecDataAdapter()
    for var in ["raw", "llm", "fused", "llm_3d", "fused_3d"]:
        if var in adapter.contents:
            adapter.export_coldrec_files(dataset_name=f"MOOCCubeX_{var}", content_variant=var)
