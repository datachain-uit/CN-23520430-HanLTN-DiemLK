"""coldrec_fix.py - sửa ColdRec để chạy USIM đúng giao thức 

Chạy MỘT lần (idempotent) sau khi coldrec_adapter.py đã xuất dữ liệu:
    patch_coldrec_source()                       # (1)(2)(3) vá mã nguồn
    write_backbone_and_info_dict(ad)             # (4)(5) backbone LightGCN + info_dict.pkl
    verify_identity_mapping(ds)                  # kiểm tra mã nội bộ == mã gốc
"""
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COLDREC_ROOT = str(REPO_ROOT / "ColdRec")
COLDREC_DATA_ROOT = str(REPO_ROOT / "ColdRec_Data")
COLDREC_COMMIT = "1108192665aca2cb94b3aeb5bfed29850874eb1c"
DATASETS = ("MOOCCubeX_raw", "MOOCCubeX_fused", "MOOCCubeX_llm")
ID_MARK = "[patch-id]"
COLD_MARK = "[patch-cold-valid]"


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def patch_coldrec_source(root=COLDREC_ROOT, restore=True):
    """Vá model/__init__.py, util/databuilder.py, model/USIM.py. Khôi phục bản gốc từ git trước
    (xóa mọi patch/hook cũ, kể cả hook AUTO_EXPORT trong main.py của các notebook trước)."""
    if restore:
        subprocess.run(
            ["git", "-C", root, "checkout", "--", "main.py", "model/__init__.py",
             "util/databuilder.py", "model/USIM.py"],
            check=True,
        )
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head.startswith(COLDREC_COMMIT), f"ColdRec phải ở commit {COLDREC_COMMIT}, hiện là {head}"

    # (1) model/__init__.py: commit này import SimCSR nhưng không có tệp SimCSR.py
    p = os.path.join(root, "model", "__init__.py")
    s = _read(p)
    assert not os.path.exists(os.path.join(root, "model", "SimCSR.py")), "SimCSR.py tồn tại: không cần vá (1)"
    s = re.sub(r"^from \.SimCSR import SimCSR[ \t]*\n", "", s, flags=re.M)      # dòng import
    s = re.sub(r"^[ \t]*'SimCSR':[ \t]*SimCSR,[ \t]*\n", "", s, flags=re.M)    # mục trong AVAILABLE_MODELS
    assert "SimCSR" not in s, "model/__init__.py vẫn còn SimCSR"
    _write(p, s)
    compile(s, p, "exec")

    # (2) util/databuilder.py: mã nội bộ == mã gốc (ColdStartDataBuilder, lớp đầu tiên của tệp).
    #     Mặc định ColdRec đánh lại mã theo thứ tự xuất hiện trong csv => backbone bị gắn nhầm, và
    #     khóa không xuất hiện trong csv (340/753 khóa lạnh) không có mã nội bộ.
    p = os.path.join(root, "util", "databuilder.py")
    s = _read(p)
    assert ID_MARK not in s
    anchor = "    def generate_set(self):\n"
    pos = s.index(anchor)                                   # lần xuất hiện đầu tiên
    assert pos < s.index("class ColdStartMetaDataBuilder"), "anchor không nằm trong ColdStartDataBuilder"
    block = (
        f"        # {ID_MARK} mã nội bộ == mã gốc; gán đúng nội dung khóa/học viên theo mã gốc\n"
        "        for u in range(self.user_num):\n"
        "            self.user[u] = u\n"
        "            self.id2user[u] = u\n"
        "            if self.source_user_content is not None and u < self.source_user_content.shape[0]:\n"
        "                self.mapped_user_content[u] = self.source_user_content[u]\n"
        "        for i in range(self.item_num):\n"
        "            self.item[i] = i\n"
        "            self.id2item[i] = i\n"
        "            if self.source_item_content is not None and i < self.source_item_content.shape[0]:\n"
        "                self.mapped_item_content[i] = self.source_item_content[i]\n"
    )
    s = s[: pos + len(anchor)] + block + s[pos + len(anchor):]
    _write(p, s)
    compile(s, p, "exec")

    # (3) model/USIM.py: dừng sớm/chọn checkpoint theo khóa lạnh thay vì 'all'
    p = os.path.join(root, "model", "USIM.py")
    s = _read(p)
    n_all = s.count("valid_type='all')")
    assert n_all == 1, f"USIM.py: kỳ vọng đúng 1 chỗ valid_type='all', tìm thấy {n_all}"
    s = s.replace("valid_type='all')", f"valid_type='cold')  # {COLD_MARK}", 1)
    _write(p, s)
    compile(s, p, "exec")

    print("[OK] (1) __init__.py không còn SimCSR | (2) databuilder: ID nội bộ == ID gốc | (3) USIM: valid_type='cold'")


def write_backbone_and_info_dict(ad, datasets=DATASETS, root=COLDREC_ROOT, data_root=COLDREC_DATA_ROOT,
                                 expect_cold=753, expect_warm=2401):
    """(4) Lưu backbone LightGCN chung đúng tên ColdRec đọc khi --backbone LightGCN.
       (5) Đặt lại info_dict.pkl: đủ 753 khóa lạnh + 2.401 khóa ấm, giữ nguyên index gốc."""
    emb_dir = os.path.join(root, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    g_u = np.ascontiguousarray(ad.g_u, dtype=np.float32)
    g_i = np.ascontiguousarray(ad.g_i, dtype=np.float32)
    cold = np.asarray(ad.cold_idx, dtype=np.int32)
    warm = np.asarray(ad.warm_active_idx, dtype=np.int32)
    assert len(cold) == expect_cold and len(warm) == expect_warm, (len(cold), len(warm))
    assert not (set(cold.tolist()) & set(warm.tolist()))
    assert np.allclose(g_i[cold], 0.0), "g_i của khóa lạnh phải bằng 0"

    for ds in datasets:
        for side, arr in (("user", g_u), ("item", g_i)):
            out = os.path.join(emb_dir, f"{ds}_cold_item_LightGCN_{side}_emb.pt")
            if os.path.lexists(out):
                os.remove(out)                       # bỏ symlink cũ của notebook trước (tránh ghi xuyên symlink)
            torch.save(torch.from_numpy(arr.copy()), out)
            back = torch.load(out, map_location="cpu")
            assert tuple(back.shape) == arr.shape and np.array_equal(back.numpy(), arr), out
        print(f"[OK] backbone: emb/{ds}_cold_item_LightGCN_{{user,item}}_emb.pt  "
              f"user{tuple(g_u.shape)} item{tuple(g_i.shape)}")

        f = os.path.join(root, "data", ds, "cold_item", "info_dict.pkl")
        if not os.path.exists(f):
            f = os.path.join(data_root, ds, "cold_item", "info_dict.pkl")
        with open(f, "rb") as fh:
            info = pickle.load(fh)
        info["cold_item"] = cold.copy()
        info["warm_item"] = warm.copy()
        with open(f, "wb") as fh:
            pickle.dump(info, fh, protocol=4)
        with open(f, "rb") as fh:
            chk = pickle.load(fh)
        assert len(chk["cold_item"]) == expect_cold and len(chk["warm_item"]) == expect_warm
        assert np.array_equal(chk["cold_item"], cold) and np.array_equal(chk["warm_item"], warm)
        assert chk["user_num"] == g_u.shape[0] and chk["item_num"] == g_i.shape[0]
        print(f"[OK] info_dict.pkl [{ds}]: cold_item={len(chk['cold_item'])}, warm_item={len(chk['warm_item'])}")


_VERIFY_SCRIPT = r"""
import pickle, sys, types, numpy as np
sys.path.insert(0, '.')
from util.loader import DataLoader
from util.databuilder import ColdStartDataBuilder
import model  # kiểm tra model/__init__.py import được (đã bỏ SimCSR)
ds = sys.argv[1]
L = lambda n: DataLoader.load_data_set(f'./data/{ds}/cold_item/{n}.csv')
info = pickle.load(open(f'./data/{ds}/cold_item/info_dict.pkl', 'rb'))
content = np.load(f'./data/{ds}/{ds}_item_content.npy')
b = ColdStartDataBuilder(L('warm_train'), L('warm_val'), L('cold_item_val'), L('overall_val'),
                         L('warm_test'), L('cold_item_test'), L('overall_test'),
                         info['user_num'], info['item_num'], info['warm_user'], info['warm_item'],
                         info['cold_user'], info['cold_item'], None, content)
assert all(b.user[u] == u for u in range(info['user_num'])) and len(b.user) == info['user_num']
assert all(b.item[i] == i for i in range(info['item_num'])) and len(b.item) == info['item_num']
assert all(b.id2user[u] == u for u in range(info['user_num'])) and all(b.id2item[i] == i for i in range(info['item_num']))
assert np.array_equal(b.mapped_cold_item_idx, np.asarray(info['cold_item'])), 'khóa lạnh bị gắn nhầm mã'
assert np.array_equal(b.mapped_warm_item_idx, np.asarray(info['warm_item'])), 'khóa ấm bị gắn nhầm mã'
n = info['item_num']
assert np.allclose(b.mapped_item_content[:n], content[:n]), 'nội dung khóa không khớp mã gốc'
print('VERIFY_OK', ds, 'cold', len(b.mapped_cold_item_idx), 'warm', len(b.mapped_warm_item_idx))
"""


def verify_identity_mapping(ds, root=COLDREC_ROOT, python=sys.executable):
    """Dựng ColdStartDataBuilder y như main.py và kiểm tra mã nội bộ == mã gốc (chạy trong tiến trình riêng)."""
    r = subprocess.run([python, "-c", _VERIFY_SCRIPT, ds], cwd=root, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join(out[-3:]))
    assert r.returncode == 0 and any(l.startswith("VERIFY_OK") for l in out), f"verify_identity_mapping({ds}) thất bại"
