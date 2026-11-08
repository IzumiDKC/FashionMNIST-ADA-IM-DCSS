"""
Hàm mục tiêu cho DEC (2016) và IDEC (2017) theo đúng chuẩn hóa tại Mục 3.1 - 3.3 của Final.docx:
- Eq. (3.1): E = (1 / (N * D)) * sum_{i, d} (x_{i,d} - x_hat_{i,d})^2
- Eq. (3.4): K_n = (1 / N) * sum_{i, j} p_{ij} * ln(p_{ij} / q_{ij})
- Eq. (3.5): L_{IDEC} = w_rec * E + gamma * K_n
"""

from typing import Dict

import torch
import torch.nn as nn


def reconstruction_mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Sai số tái tạo trung bình trên cả N mẫu và D đặc trưng theo Eq. (3.1)."""
    return torch.mean((x - x_hat) ** 2)


def kl_divergence_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Phân kỳ Kullback-Leibler lấy trung bình theo số mẫu N (Eq. 3.4):
    K_n = (1/N) * sum_i sum_j p_ij * ln(p_ij / q_ij)
    """
    q_safe = q.clamp(min=1e-10)
    p_safe = p.clamp(min=1e-10)
    kl_per_sample = torch.sum(p_safe * torch.log(p_safe / q_safe), dim=1)
    return torch.mean(kl_per_sample)


class DECIDECLoss(nn.Module):
    """
    Lớp tính Loss tổng hợp cho DEC (w_rec=0.0, gamma_kl=1.0)
    và IDEC (w_rec=1.0, gamma_kl=0.1).
    """

    def __init__(self, w_rec: float = 1.0, gamma_kl: float = 0.1) -> None:
        super().__init__()
        self.w_rec = w_rec
        self.gamma_kl = gamma_kl

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        e_rec = reconstruction_mse_loss(x, x_hat)
        k_n = kl_divergence_loss(q, p)
        total = self.w_rec * e_rec + self.gamma_kl * k_n
        return {
            "total": total,
            "rec": e_rec.detach(),
            "kl": k_n.detach(),
        }
