
import torch
import torch.nn as nn
import torch.nn.functional as F

class MVP(nn.Module):
    """
    Mô hình MVP kết hợp biểu diễn đồ thị LightGCN nền chung với nhánh chiếu ngữ nghĩa MLP 2 lớp,
    cơ chế ID-Dropout và căn chỉnh tương phản InfoNCE.
    """
    def __init__(
        self,
        g_u_init: torch.Tensor,
        g_i_init: torch.Tensor,
        sem: torch.Tensor,
        d: int = 64,
        p_drop: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.nu, self.ni = g_u_init.size(0), g_i_init.size(0)
        self.p = p_drop
        self.freeze_backbone = freeze_backbone

        # Đóng băng hoặc tinh chỉnh tiếp g_u, g_i từ LightGCN nền chung
        if freeze_backbone:
            self.register_buffer("g_u", g_u_init.clone().detach())
            self.register_buffer("g_i", g_i_init.clone().detach())
        else:
            self.g_u = nn.Parameter(g_u_init.clone().detach())
            self.g_i = nn.Parameter(g_i_init.clone().detach())

        # Vector văn bản đóng băng: [3769, 384] hoặc [3769, 768]
        self.register_buffer("sem", sem.clone().detach())

        # Mạng chiếu phi tuyến 2 lớp với GELU: d_t -> d -> d
        d_text = sem.size(1)
        self.proj = nn.Sequential(
            nn.Linear(d_text, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def item_rep(self, cold_mask: torch.Tensor = None):
        """
        Trả về:
            h_item: [n_items, d] - Biểu diễn khóa học (dùng cho BPR loss khi train hoặc chấm điểm khi eval)
            s_proj: [n_items, d] - Biểu diễn ngữ nghĩa f(s_i) (dùng cho InfoNCE align_loss)
            g_raw:  [n_items, d] - Biểu diễn cộng tác g_i gốc chưa bị ID-dropout (dùng cho InfoNCE align_loss)
        """
        s_proj = self.proj(self.sem)
        g_raw = self.g_i
        g_used = g_raw

        if self.training and self.p > 0.0:
            keep = (torch.rand(g_used.size(0), 1, device=g_used.device) > self.p).float()
            g_used = g_used * keep

        if cold_mask is not None:
            g_used = g_used.masked_fill(cold_mask[:, None], 0.0)

        h_item = g_used + s_proj
        return h_item, s_proj, g_raw


def sample_neg(n: int, warm_items: torch.Tensor) -> torch.Tensor:
    """Lấy mẫu âm nghiêm ngặt trong tập khóa học ấm (I_w)."""
    idx = torch.randint(len(warm_items), (n,), device=warm_items.device)
    return warm_items[idx]


def bpr_loss(u_emb: torch.Tensor, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
    """Hàm mất mát xếp hạng BPR."""
    pos_scores = (u_emb * h_pos).sum(dim=-1)
    neg_scores = (u_emb * h_neg).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()
