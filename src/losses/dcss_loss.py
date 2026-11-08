"""
Hàm mục tiêu cho DCSS (Vardakas et al., Machine Learning 2026, 115:81)
và cải tiến đề xuất IM-DCSS / Ada-IM-DCSS (Information Maximization & Scheduled Entropy Warmup):

1. DCSS gốc (Eq. 13 - 22, Trang 8 - 11):
    L_{DCSS} = w_rec * L_{rec} + lambda_1 * (1 - Sf(h_r(X))) + lambda_2 * H(Q | X)

2. Cải tiến IM-DCSS & Ada-IM-DCSS (Ours):
   Khắc phục hiện tượng thành phần H(Q | X) ở Eq. (20) ép xác suất của các mẫu biên nhập nhằng
   hội tụ quá sớm về one-hot theo lỗi khởi tạo K-Means:
    L_{IM-DCSS} = w_rec * L_{rec} + lambda_1 * (1 - Sf(h_r(X)))
                  + lambda_bal * KL(p_bar || Uniform_K) + lambda_2 * omega(t) * H(Q | X)
   Trong đó:
   - KL(p_bar || Uniform_K) = sum_j p_bar_j * log(K * p_bar_j) cực đại hóa Entropy biên H(p_bar),
     ngăn gộp cụm/sụp đổ cụm ngay từ epoch đầu mà không khóa cứng từng mẫu riêng lẻ.
   - omega(t) in [0, 1]: Lịch trình Cosine Warmup trì hoãn H(Q | X) trong giai đoạn đầu (t < warmup_ratio)
     để Soft Silhouette tự do tách biên cụm, sau đó mới làm sắc nét phân phối ở giai đoạn hội tụ cuối.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .dec_idec_loss import reconstruction_mse_loss


def compute_soft_silhouette(z: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Tính điểm Soft Silhouette khả vi Sf(X) in [-1, 1] trên mini-batch theo đúng Eq. (13)-(17) (Trang 8):
    - Eq. (13): a_{C_I}(x_i) = (sum_{j=1, j != i}^N P_{C_I}(x_j) d(z_i, z_j)) / (sum_{j=1, j != i}^N P_{C_I}(x_j) + eps)
    - Eq. (14): b_{C_I}(x_i) = min_{J != I} a_{C_J}(x_i)
    - Eq. (15): s_{C_I}(x_i) = (b_{C_I}(x_i) - a_{C_I}(x_i)) / max{a_{C_I}(x_i), b_{C_I}(x_i)}
    - Eq. (16): sf(x_i)      = sum_{I=1}^K P_{C_I}(x_i) s_{C_I}(x_i)
    - Eq. (17): Sf(X)        = (1 / N) * sum_{i=1}^N sf(x_i)
    """
    batch_size, n_clusters = q.shape
    if batch_size <= 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    pairwise_dist = torch.cdist(z, z, p=2.0)
    eye_mask = 1.0 - torch.eye(batch_size, device=z.device, dtype=z.dtype)
    masked_dist = pairwise_dist * eye_mask  # [B, B]

    cluster_mass_total = q.sum(dim=0, keepdim=True)  # [1, K]
    denom = (cluster_mass_total - q).clamp(min=eps)  # [B, K]

    a_ic = (masked_dist @ q) / denom  # [B, K]

    large_val = 1e6
    diag_k = torch.eye(n_clusters, device=z.device, dtype=z.dtype).unsqueeze(0)  # [1, K, K]
    a_expanded = a_ic.unsqueeze(1).expand(batch_size, n_clusters, n_clusters)  # [B, K, K]
    b_ic = torch.min(a_expanded + diag_k * large_val, dim=2).values  # [B, K]

    max_ab = torch.maximum(a_ic, b_ic).clamp(min=eps)
    s_ic = (b_ic - a_ic) / max_ab  # [B, K]

    sf_i = torch.sum(q * s_ic, dim=1)  # [B]
    return torch.mean(sf_i)


def compute_conditional_entropy(q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Entropy có điều kiện H(Q | X) theo đúng Eq. (20) & (22) bài báo DCSS (Trang 10-11):
        H(Q | X) = - (1 / N) * sum_{i=1}^N sum_{j=1}^K h_{rj}(x_i) log h_{rj}(x_i) >= 0
    """
    q_safe = q.clamp(min=eps)
    return -torch.mean(torch.sum(q_safe * torch.log(q_safe), dim=1))


def compute_marginal_balance_kl(q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Thành phần cực đại hóa Entropy biên trong Information Maximization (IM):
        KL(p_bar || Uniform_K) = sum_{j=1}^K p_bar_j * log(K * p_bar_j) = log(K) - H(p_bar) >= 0
    Đạt cực tiểu (= 0) khi phân bố kích thước các cụm cân bằng, chống gộp cụm / cụm rỗng.
    """
    n_clusters = q.shape[1]
    p_bar = q.mean(dim=0).clamp(min=eps)
    return torch.sum(p_bar * torch.log(p_bar * n_clusters))


def compute_entropy_regularization(
    q: torch.Tensor,
    balance_weight: float = 0.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Hàm tương thích ngược cho DCSS gốc (khi balance_weight == 0 trả về đúng Eq. 20)."""
    cond_ent = compute_conditional_entropy(q, eps=eps)
    if balance_weight > 0.0:
        return cond_ent + balance_weight * compute_marginal_balance_kl(q, eps=eps)
    return cond_ent


def compute_cosine_warmup_factor(progress: float, warmup_ratio: float) -> float:
    """
    Tính hệ số điều lịch omega(t) in [0, 1] cho IM-DCSS:
    - Khi progress <= warmup_ratio: omega(t) = 0.0 (giai đoạn tự do tách cụm bằng Soft Silhouette).
    - Khi progress > warmup_ratio: tăng mượt theo hàm Cosine từ 0.0 -> 1.0.
    """
    if warmup_ratio <= 0.0:
        return 1.0
    if progress <= warmup_ratio:
        return 0.0
    if warmup_ratio >= 1.0:
        return 0.0
    rel = min(max((progress - warmup_ratio) / (1.0 - warmup_ratio), 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * rel))


class DCSSLoss(nn.Module):
    """
    Lớp tính Loss tổng hợp cho:
    - DCSS gốc (sigma=1, sigma=RMS với lambda_1 = 0.01, lambda_2 = 0.01, use_infomax=False)
    - Ada-DCSS (Đóng góp 1: Per-Cluster EMA-RMS Sigma)
    - IM-DCSS (Đóng góp 2: Information Maximization + Cosine Entropy Warmup)
    - Ada-IM-DCSS (Đóng góp tổng hợp Full Proposed: Ada + IM)
    - Các cấu hình Ablation.
    """

    def __init__(
        self,
        w_rec: float = 1.0,
        lambda_sil: float = 0.01,
        beta_ent: float = 0.01,
        use_infomax: bool = False,
        lambda_bal: float = 0.005,
        entropy_warmup_ratio: float = 0.60,
        eta_cons: float = 0.0,
        cons_on_assignments: bool = True,
        balance_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_rec = w_rec
        self.lambda_sil = lambda_sil
        self.beta_ent = beta_ent
        self.use_infomax = use_infomax
        self.lambda_bal = lambda_bal
        self.entropy_warmup_ratio = entropy_warmup_ratio
        self.eta_cons = eta_cons
        self.cons_on_assignments = cons_on_assignments
        self.balance_weight = balance_weight

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        q: torch.Tensor,
        progress: float = 1.0,
        x_hat_tilde: Optional[torch.Tensor] = None,
        q_tilde: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        e_rec = reconstruction_mse_loss(x, x_hat)
        s_soft = compute_soft_silhouette(z, q)
        sil_term = 1.0 - s_soft

        cond_ent = compute_conditional_entropy(q)
        bal_kl = compute_marginal_balance_kl(q)

        if self.use_infomax:
            omega = compute_cosine_warmup_factor(progress, self.entropy_warmup_ratio)
            reg_term = self.lambda_bal * bal_kl + (self.beta_ent * omega) * cond_ent
        else:
            reg_term = self.beta_ent * (cond_ent + self.balance_weight * bal_kl)

        total = self.w_rec * e_rec + self.lambda_sil * sil_term + reg_term

        e_cons = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.eta_cons > 0.0 and x_hat_tilde is not None:
            rec_cons = reconstruction_mse_loss(x_hat_tilde, x_hat)
            if self.cons_on_assignments and q_tilde is not None:
                assign_cons = torch.mean((q - q_tilde) ** 2)
                e_cons = rec_cons + 0.1 * assign_cons
            else:
                e_cons = rec_cons
            total = total + self.eta_cons * e_cons

        return {
            "total": total,
            "rec": e_rec.detach(),
            "s_soft": s_soft.detach(),
            "sil_loss": sil_term.detach(),
            "ent": cond_ent.detach(),
            "bal_kl": bal_kl.detach(),
            "cons": e_cons.detach(),
        }
