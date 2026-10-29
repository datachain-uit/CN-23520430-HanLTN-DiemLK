import copy
import itertools
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .coldrec_adapter import ColdRecDataAdapter, evaluate, eval_subgroups
except ImportError:
    from coldrec_adapter import ColdRecDataAdapter, evaluate, eval_subgroups

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 2023, 2024, 1007, 999]
ROBUST_VAL_SEEDS = [42, 2023, 2024]


# Kiến trúc ALDI chuyển nguyên bản từ mã TensorFlow chính thức 
class ALDI(nn.Module):
    def __init__(self, d_content, d_emb=64, h=200):
        super().__init__()
        self.f_I = nn.Sequential(
            nn.Linear(d_content, h),
            nn.BatchNorm1d(h),
            nn.Tanh(),
            nn.Linear(h, d_emb),
        )
        self.f_U = nn.Sequential(
            nn.Linear(d_emb, h),
            nn.BatchNorm1d(h),
            nn.Tanh(),
            nn.Linear(h, d_emb),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Khởi tạo chuẩn bị cắt với độ lệch chuẩn 0,01.
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.02, b=0.02)
                nn.init.zeros_(m.bias)

    def l2(self):  # l2_regularizer: sum(w^2)/2, cả weight và bias
        return (
            sum(
                (m.weight ** 2).sum() + (m.bias ** 2).sum()
                for m in self.modules()
                if isinstance(m, nn.Linear)
            )
            / 2
        )


def aldi_loss(model, u_t, c_pos, c_neg, e_pos, e_neg, w_pos, cfg):
    s_pos, s_neg = model.f_I(torch.cat([c_pos, c_neg])).chunk(2)  # BN trên lô dương + âm
    u_s = model.f_U(u_t)
    sp, sn = (u_s * s_pos).sum(-1), (u_s * s_neg).sum(-1)         # học sinh
    tp, tn = (u_t * e_pos).sum(-1), (u_t * e_neg).sum(-1)         # giáo viên (LightGCN)
    bce = F.binary_cross_entropy_with_logits
    sup = bce(sp - sn, torch.ones_like(sp))
    rank = (w_pos * bce(sp - sn, torch.sigmoid(tp - tn), reduction="none")).mean()
    s_id = (s_pos * s_pos).sum(-1) - (s_pos @ s_neg.t()).mean(1)
    t_id = (e_pos * e_pos).sum(-1) - (e_pos @ e_neg.t()).mean(1)
    iden = (w_pos * bce(s_id, torch.sigmoid(t_id), reduction="none")).mean()
    dist = ((tp - sp).abs() + (tn - sn).abs()).mean()
    return (
        sup
        + cfg["alpha"] * rank
        + cfg["beta"] * iden
        + cfg["gamma"] * dist
        + cfg["reg"] * model.l2()
    )


# Khởi tạo Adapter, trọng số tần suất khóa (tws) và hàm lấy mẫu 
ad = ColdRecDataAdapter()
g_u_t = torch.tensor(ad.g_u, dtype=torch.float32, device=DEV)
g_i_t = torch.tensor(ad.g_i, dtype=torch.float32, device=DEV)

u_tr = ad.tr_cold.user_id.astype(str).map(ad.user2idx).to_numpy()
i_tr = ad.tr_cold.course_id.astype(str).map(ad.item2idx).to_numpy()
deg_u = np.bincount(u_tr, minlength=ad.n_users)
f = np.zeros(ad.n_items)
np.add.at(f, i_tr, 1.0 / deg_u[u_tr])
freq = np.ones(ad.n_items, dtype=np.float32)
freq[ad.warm_active_idx] = f[ad.warm_active_idx]
M = 4.0
a = M / (ad.n_users / ad.n_items)
w_item = torch.tensor(
    np.clip(np.tanh(a * freq), 0, np.tanh(M)), dtype=torch.float32, device=DEV
)

warm_users = np.array(sorted(ad.hist_tr))


def sample_epoch(rng):  # như utils.bpr_neg_samp
    users = rng.choice(warm_users, size=len(u_tr), replace=True)
    pos = np.array([ad.hist_tr[u][rng.integers(len(ad.hist_tr[u]))] for u in users])
    neg = rng.choice(ad.warm_active_idx, size=len(u_tr), replace=True)
    return users, pos, neg


# 1.368 khóa không có tương tác trong train (753 khóa lạnh + 615 khóa ấm không hoạt động)
gen_cols = torch.tensor(
    np.setdiff1d(np.arange(ad.n_items), ad.warm_active_idx), device=DEV
)


def aldi_score_fn(model, content_t):
    model.eval()
    with torch.no_grad():
        u_map, gen = model.f_U(g_u_t), model.f_I(content_t)

    def fn(ub):
        ub = ub.to(DEV)
        sc = g_u_t[ub] @ g_i_t.t()                     # khóa ấm: điểm giáo viên
        sc[:, gen_cols] = u_map[ub] @ gen[gen_cols].t() # khóa còn lại: học sinh
        return sc

    return fn


def train_aldi(content_t, cfg, seed, max_epochs=200, patience=10, bs=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = ALDI(content_t.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])  # L2 nằm trong loss
    w_t = w_item if cfg["tws"] else torch.ones_like(w_item)
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        users, pos, neg = (torch.tensor(x, device=DEV) for x in sample_epoch(rng))
        for s in range(0, len(users) - bs, bs):  # như main.py: bỏ lô cuối thiếu
            u, p, n = users[s : s + bs], pos[s : s + bs], neg[s : s + bs]
            loss = aldi_loss(
                model,
                g_u_t[u],
                content_t[p],
                content_t[n],
                g_i_t[p],
                g_i_t[n],
                w_t[p],
                cfg,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = evaluate(
            aldi_score_fn(model, content_t),
            ad.val_target_dict,
            mask=ad.hist_tr,
            cand=ad.cand_val_a,
            ks=(20,),
            tie_seed=seed,
        )
        v = float(np.mean(r["NDCG@20"]))
        if v > best:
            best, best_state, bad, best_ep = (
                v,
                copy.deepcopy(model.state_dict()),
                0,
                ep,
            )
        else:
            bad += 1
            if bad > patience:
                break  # như main.py
    model.load_state_dict(best_state)
    return model, best, best_ep


# Không gian siêu tham số 20 cấu hình 
space = dict(
    lr=[1e-4, 5e-4, 1e-3, 5e-3],
    alpha=[0.1, 0.5, 0.9],
    beta=[0.01, 0.05, 0.1],
    gamma=[0.05, 0.1, 0.5],
    reg=[1e-5, 1e-4, 1e-3],
    tws=[0, 1],
)
grid = [dict(zip(space, v)) for v in itertools.product(*space.values())]
trials = random.Random(42).sample(grid, 20)
DEFAULT = dict(lr=1e-3, alpha=0.9, beta=0.05, gamma=0.1, reg=1e-4, tws=0)
if DEFAULT not in trials:
    trials[-1] = DEFAULT  # vẫn đúng 20 cấu hình


def evaluate_5_seeds(inp_name, cfg, tag_suffix=""):
    content_t = torch.tensor(ad.contents[inp_name], dtype=torch.float32, device=DEV)
    out_dir = os.path.join(ad.output_dir, "results_aldi")
    rep_dir = os.path.join(out_dir, "item_reps")
    score_dir = os.path.join(ad.output_dir, "scores")
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    metrics_a = {"Recall@20": [], "NDCG@20": []}
    metrics_b = {"Recall@20": [], "NDCG@20": []}
    sub_metrics = {
        "trung": {"Recall@20": [], "NDCG@20": []},
        "moi": {"Recall@20": [], "NDCG@20": []},
    }
    user_ndcg_a_seeds = []
    user_ndcg_sub_seeds = {"trung": [], "moi": []}

    n_params = sum(p.numel() for p in ALDI(content_t.shape[1]).parameters() if p.requires_grad)

    for s in SEEDS:
        t0 = time.time()
        model, val_ndcg, best_ep = train_aldi(content_t, cfg, seed=s)
        fn = aldi_score_fn(model, content_t)

        kw = dict(mask=ad.hist_trva, ks=(5, 10, 20), tie_seed=s)
        res_a = evaluate(fn, ad.test_target_dict, cand=ad.cand_test_a, **kw)
        res_b = evaluate(fn, ad.test_target_dict, cand=ad.cand_test_b, **kw)
        res_prq3 = eval_subgroups(
            fn,
            ad.test_target_dict,
            ad.hist_trva,
            ad.cand_test_a,
            ad.groups_prq3,
            ks=(5, 10, 20),
            tie_seed=s,
        )

        # Lưu vector biểu diễn theo từng seed (bọc trong torch.no_grad)
        model.eval()
        with torch.no_grad():
            u_map, gen = model.f_U(g_u_t), model.f_I(content_t)
        save_key = f"{inp_name}{tag_suffix}"
        np.save(os.path.join(rep_dir, f"u_e_{save_key}_seed{s}.npy"), u_map.detach().cpu().numpy())
        np.save(os.path.join(rep_dir, f"v_e_{save_key}_seed{s}.npy"), gen.detach().cpu().numpy())

        # Ghi nhận chỉ số
        metrics_a["Recall@20"].append(float(np.mean(res_a["Recall@20"])))
        metrics_a["NDCG@20"].append(float(np.mean(res_a["NDCG@20"])))
        metrics_b["Recall@20"].append(float(np.mean(res_b["Recall@20"])))
        metrics_b["NDCG@20"].append(float(np.mean(res_b["NDCG@20"])))

        user_ndcg_a_seeds.append(np.asarray(res_a["NDCG@20"], dtype=np.float32))
        for g in ("trung", "moi"):
            sub_metrics[g]["Recall@20"].append(float(np.mean(res_prq3[g]["Recall@20"])))
            sub_metrics[g]["NDCG@20"].append(float(np.mean(res_prq3[g]["NDCG@20"])))
            user_ndcg_sub_seeds[g].append(np.asarray(res_prq3[g]["NDCG@20"], dtype=np.float32))

        print(
            f"  [Seed {s}] ep={best_ep:02d} | val_NDCG@20={val_ndcg:.4f} | "
            f"(a) R@20={metrics_a['Recall@20'][-1]:.4f}, NDCG@20={metrics_a['NDCG@20'][-1]:.4f} | "
            f"(b) NDCG@20={metrics_b['NDCG@20'][-1]:.4f} | "
            f"trùng NDCG@20={sub_metrics['trung']['NDCG@20'][-1]:.4f}, "
            f"mới NDCG@20={sub_metrics['moi']['NDCG@20'][-1]:.4f} ({time.time()-t0:.1f}s)"
        )

    # Lưu tệp điểm từng học viên (trung bình qua 5 seed)
    save_key = f"{inp_name}{tag_suffix}"
    mean_user_a = np.mean(np.stack(user_ndcg_a_seeds, axis=0), axis=0)
    assert len(mean_user_a) == 3340
    np.save(os.path.join(score_dir, f"user_ndcg20_aldi_{save_key}.npy"), mean_user_a)

    for g, expected_len in (("trung", 1028), ("moi", 2844)):
        mean_user_g = np.mean(np.stack(user_ndcg_sub_seeds[g], axis=0), axis=0)
        assert len(mean_user_g) == expected_len
        np.save(os.path.join(score_dir, f"user_ndcg20_aldi_{save_key}_{g}.npy"), mean_user_g)

    summary = {
        "input": save_key,
        "cfg": cfg,
        "n_params": n_params,
        "a_R20_mean": float(np.mean(metrics_a["Recall@20"])),
        "a_R20_std": float(np.std(metrics_a["Recall@20"], ddof=1)),
        "a_NDCG20_mean": float(np.mean(metrics_a["NDCG@20"])),
        "a_NDCG20_std": float(np.std(metrics_a["NDCG@20"], ddof=1)),
        "b_R20_mean": float(np.mean(metrics_b["Recall@20"])),
        "b_NDCG20_mean": float(np.mean(metrics_b["NDCG@20"])),
        "trung_R20_mean": float(np.mean(sub_metrics["trung"]["Recall@20"])),
        "trung_NDCG20_mean": float(np.mean(sub_metrics["trung"]["NDCG@20"])),
        "moi_R20_mean": float(np.mean(sub_metrics["moi"]["Recall@20"])),
        "moi_NDCG20_mean": float(np.mean(sub_metrics["moi"]["NDCG@20"])),
    }
    print(
        f"=> TỔNG KẾT ALDI [{save_key}] (params={n_params:,}):\n"
        f"   Cold-test (a): NDCG@20 = {summary['a_NDCG20_mean']:.4f} ± {summary['a_NDCG20_std']:.4f} | "
        f"R@20 = {summary['a_R20_mean']:.4f} ± {summary['a_R20_std']:.4f}\n"
        f"   All-items (b): NDCG@20 = {summary['b_NDCG20_mean']:.4f} | R@20 = {summary['b_R20_mean']:.4f}\n"
        f"   Nhóm TRÙNG / MỚI: R@20 = {summary['trung_R20_mean']:.4f} / {summary['moi_R20_mean']:.4f} | "
        f"NDCG@20 = {summary['trung_NDCG20_mean']:.4f} / {summary['moi_NDCG20_mean']:.4f}\n"
    )
    return summary


# Cấu hình đã chốt ở lần chạy trước của 'raw' để kiểm tra tái lập (KHÔNG chọn lại nếu lệch nhẹ)
LOCKED_RAW_CFG = {
    "lr": 0.005,
    "alpha": 0.9,
    "beta": 0.01,
    "gamma": 0.1,
    "reg": 0.001,
    "tws": 0,
}


def tune_and_run_aldi(
    variants=("raw", "llm", "fused"),
    run_erq3=True,
    run_w6_robustness=True,
):
    out_dir = os.path.join(ad.output_dir, "results_aldi")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "aldi_summary.json")
    all_summaries = {}
    best_cfgs = {}
    top3_cfgs = {}

    # Chạy thực tế cấu hình DEFAULT (seed=42, raw) để kiểm tra tái lập (kỳ vọng 0.0773 tại epoch 5)
    print("=== KIỂM TRA CẤU HÌNH MẶC ĐỊNH (DEFAULT, seed=42, raw) ===")
    raw_content_t = torch.tensor(
        ad.contents["raw"], dtype=torch.float32, device=DEV
    )
    _, def_val_ndcg, def_best_ep = train_aldi(raw_content_t, DEFAULT, seed=42)
    print(
        f"ALDI chính thức cấu hình DEFAULT (raw, seed=42): "
        f"cold-val NDCG@20 = {def_val_ndcg:.4f} (tại epoch {def_best_ep}) "
        f"[Kỳ vọng: 0.0773 tại epoch 5]\n"
    )

    # Tinh chỉnh 20 cấu hình (seed=42) cho mọi đầu vào và LƯU LOG trước khi sắp xếp (Phụ lục A.3)
    for inp in variants:
        print(f"=== TINH CHỈNH 20 CẤU HÌNH CHO ALDI [{inp}] (seed=42) ===")
        content_t = torch.tensor(ad.contents[inp], dtype=torch.float32, device=DEV)
        trial_records = []
        for idx, cfg in enumerate(trials, 1):
            _, val_ndcg, best_ep = train_aldi(content_t, cfg, seed=42)
            trial_records.append((val_ndcg, best_ep, cfg))
            print(
                f"  Trial {idx:02d}/20 | val_NDCG@20={val_ndcg:.4f}"
                f" (ep={best_ep:02d}) | {cfg}"
            )

        # Lưu log 20 cấu hình trước khi sắp xếp theo đúng Phụ lục A.3
        json.dump(
            [
                dict(trial=i + 1, val_ndcg20=v, best_ep=e, cfg=c)
                for i, (v, e, c) in enumerate(trial_records)
            ],
            open(os.path.join(out_dir, f"trials_{inp}.json"), "w"),
            indent=1,
        )

        trial_records.sort(key=lambda x: x[0], reverse=True)
        best_val, best_ep, best_cfg = trial_records[0]

        if inp == "raw":
            print(
                f"[Kiểm tra tái lập raw] Best val_NDCG@20={best_val:.4f} tại"
                f" ep={best_ep} (Kỳ vọng: 0.1018 tại ep=25)"
            )
            if best_cfg != LOCKED_RAW_CFG:
                print(
                    f"[CẢNH BÁO X3] Cấu hình tốt nhất mới ({best_cfg}) khác lần trước."
                    f" Giữ nguyên cấu hình đã khóa: {LOCKED_RAW_CFG}"
                )
                best_cfg = LOCKED_RAW_CFG

        best_cfgs[inp] = best_cfg
        top3_cfgs[inp] = [r[2] for r in trial_records[:3]]
        print(
            f"-> Cấu hình tốt nhất [{inp}]: val_NDCG@20={best_val:.4f}"
            f" (ep={best_ep}) | {best_cfg}"
        )

        all_summaries[inp] = evaluate_5_seeds(inp, best_cfg)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Hồ sơ 3 chiều dùng cấu hình của 5 chiều tương ứng (không tune lại)
    if run_erq3:
        for inp_3d, base_5d in (("llm_3d", "llm"), ("fused_3d", "fused")):
            if inp_3d in ad.contents and base_5d in best_cfgs:
                print(f"=== CHẠY ALDI [{inp_3d}] DÙNG CẤU HÌNH CỦA [{base_5d}] ===")
                all_summaries[inp_3d] = evaluate_5_seeds(inp_3d, best_cfgs[base_5d])
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Phân tích độ bền chọn cấu hình bằng trung bình 3 seed trên top-3 trials và lưu log JSON (Phụ lục A.3)
    if run_w6_robustness:
        for inp in variants:
            print(f"=== PHÂN TÍCH ĐỘ BỀN CHỌN 3 SEED CHO ALDI [{inp}] ===")
            content_t = torch.tensor(
            ad.contents[inp], dtype=torch.float32, device=DEV
            )
            best_3s_val, best_3s_cfg = -1.0, None
            rob_log = []
            for cfg in top3_cfgs[inp]:
                vals = [train_aldi(content_t, cfg, seed=s)[1] for s in ROBUST_VAL_SEEDS]
                m_val = float(np.mean(vals))
                rob_log.append(dict(cfg=cfg, val_3seed=vals, mean=m_val))
                print(f"  Top-3 candidate {cfg} -> 3-seed cold-val mean = {m_val:.4f}")
                if m_val > best_3s_val:
                    best_3s_val, best_3s_cfg = m_val, cfg

            json.dump(
                rob_log,
                open(os.path.join(out_dir, f"robust3s_{inp}.json"), "w"),
                indent=1,
            )
            print(
                f"-> Cấu hình chọn theo 3 seed [{inp}]: {best_3s_cfg}"
                f" (val={best_3s_val:.4f})"
            )
            all_summaries[f"{inp}_robust3s"] = evaluate_5_seeds(
                inp, best_3s_cfg, tag_suffix="_robust3s"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    tune_and_run_aldi(
        variants=("raw", "llm", "fused"),
        run_erq3=True,
        run_w6_robustness=True,  # Đặt True khi chạy phân tích độ bền 
    )
