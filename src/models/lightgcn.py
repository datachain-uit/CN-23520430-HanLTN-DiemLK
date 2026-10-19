import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
            
        lightgcn_emb = torch.stack(embs, dim=1).mean(dim=1)
        users_emb_final, items_emb_final = torch.split(lightgcn_emb, [self.num_users, self.num_items])
        return users_emb_final, items_emb_final

def bpr_loss(users, pos_items, neg_items, u_emb, i_emb):
    pos_scores = torch.sum(u_emb[users] * i_emb[pos_items], dim=1)
    neg_scores = torch.sum(u_emb[users] * i_emb[neg_items], dim=1)
    return -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))


def build_lightgcn_adj_matrix(users_tensor, items_tensor, num_users, num_items, device):
    """
    Nhận vào danh sách user và item tương tác, trả về ma trận kề thưa chuẩn hóa (adj_matrix) trên thiết bị chỉ định.
    """
    row = torch.cat([users_tensor, items_tensor + num_users])
    col = torch.cat([items_tensor + num_users, users_tensor])
    val = torch.ones(row.size(0), dtype=torch.float32)
    
    total_nodes = num_users + num_items
    assoc = torch.sparse_coo_tensor(torch.stack([row, col]), val, (total_nodes, total_nodes)).to(device)
    
    deg = torch.sparse.sum(assoc, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    
    i_idx = torch.arange(total_nodes, device=device)
    diag_indices = torch.stack([i_idx, i_idx], dim=0)
    deg_mat_inv_sqrt = torch.sparse_coo_tensor(diag_indices, deg_inv_sqrt.to(device), (total_nodes, total_nodes))
    
    adj_matrix = torch.sparse.mm(deg_mat_inv_sqrt, torch.sparse.mm(assoc, deg_mat_inv_sqrt))
    return adj_matrix