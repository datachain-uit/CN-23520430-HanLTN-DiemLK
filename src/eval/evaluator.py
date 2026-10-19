import torch
import numpy as np

def topk_random_ties(sc, K, g):
    """sc: [B, N] điểm, các khóa bị loại có giá trị -inf. Trả về chỉ số top-K [B, K]."""
    perm = torch.argsort(torch.rand(sc.shape, generator=g), dim=1)   # hoán vị riêng từng hàng
    sp = torch.gather(sc, 1, perm)
    order = torch.sort(sp, dim=1, descending=True, stable=True).indices[:, :K]
    return torch.gather(perm, 1, order).numpy()

# Chuyển danh sách course_id sang mặt nạ bool
def to_bool_mask(ids, c2i, n_items):
    m = torch.zeros(n_items, dtype=torch.bool)
    m[[c2i[str(c)] for c in ids]] = True
    return m

@torch.no_grad() 
def evaluate(scores_fn, target, mask, ks=(5, 10, 20), cand=None, bs=1024, tie_seed=0): 
    users = sorted(target); K = max(ks) 
    out = {f"{m}@{k}": [] for m in ("Recall", "NDCG", "HR") for k in ks} 
    g = torch.Generator().manual_seed(tie_seed) 
    disc = 1.0 / np.log2(np.arange(2, K + 2)) 

    first_batch = True
    for s in range(0, len(users), bs): 
        ub = users[s:s + bs]
        sc = scores_fn(torch.tensor(ub)).float().cpu() 
        if cand is not None: 
            cand = torch.as_tensor(cand)
            if first_batch:
                # assert kiểm tra kiểu dữ liệu và kích thước của cand
                assert cand.dtype == torch.bool and cand.numel() == sc.size(1), \
                    f"cand phải là mặt nạ bool dài {sc.size(1)}, nhận {cand.dtype}, {cand.numel()}"
            sc[:, ~cand] = -float("inf")
            #sc[:, ~torch.as_tensor(cand)] = -float("inf") 
        for r, u in enumerate(ub): 
            if mask.get(u): 
                m = mask.get(u, ())
                if first_batch:
                    # Assert kiểm tra chỉ số trong mask không bị vượt ngoài phạm vi
                    assert all(0 <= i < sc.size(1) for i in m), "mask chứa chỉ số ngoài phạm vi"
                sc[r, list(m)] = -float("inf")
                #sc[r, list(mask[u])] = -float("inf") 
        
        # Phá hòa ngẫu nhiên riêng biệt cho từng học viên
        top = topk_random_ties(sc, K, g)
        
        for r, u in enumerate(ub): 
            gt = target[u]; hit = np.isin(top[r], list(gt)).astype(float) 
            for k in ks: 
                h = hit[:k]; idcg = disc[:min(len(gt), k)].sum() 
                out[f"Recall@{k}"].append(h.sum() / len(gt)) 
                out[f"NDCG@{k}"].append((h * disc[:k]).sum() / idcg) 
                out[f"HR@{k}"].append(float(h.sum() > 0)) 
    return {m: np.array(v) for m, v in out.items()} 