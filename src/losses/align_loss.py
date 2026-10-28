
import torch
import torch.nn.functional as F

def align_loss(
    g: torch.Tensor,
    s: torch.Tensor,
    dup_group_id: torch.Tensor,
    tau: float = 0.2,
    use_false_neg_mask: bool = True,
) -> torch.Tensor:
    """
    Hàm mất mát căn chỉnh tương phản InfoNCE có che cặp âm giả theo dup_group_id.
    
    Tham số:
        g: [B_I, d] - Vector cộng tác g_i (CHƯA bị ID-dropout) của các khóa ấm DUY NHẤT trong batch.
        s: [B_I, d] - Vector ngữ nghĩa sau chiếu f(s_i) của các khóa ấm DUY NHẤT trong batch.
        dup_group_id: [B_I] - Mã nhóm trùng lặp (gộp bắc cầu trùng tên chuẩn hóa hoặc trùng mô tả).
        tau: Nhiệt độ InfoNCE (mặc định 0.2).
        use_false_neg_mask: Đặt True cho SemProj chính; đặt False khi chạy Ablation "bỏ mặt nạ âm giả".
    """
    g_norm = F.normalize(g, dim=-1)
    s_norm = F.normalize(s, dim=-1)
    logits = (g_norm @ s_norm.t()) / tau

    if use_false_neg_mask:
        same_group = dup_group_id[:, None] == dup_group_id[None, :]
        eye = torch.eye(len(g), dtype=torch.bool, device=g.device)
        # Che các cặp khác mã khóa học (ngoài đường chéo) nhưng cùng dup_group_id
        logits = logits.masked_fill(same_group & ~eye, float("-inf"))

    targets = torch.arange(len(g), device=g.device)
    return F.cross_entropy(logits, targets)
