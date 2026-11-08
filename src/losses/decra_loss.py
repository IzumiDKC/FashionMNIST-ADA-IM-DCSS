"""
Hàm mục tiêu cho DECRA (2024) theo Mục 3.4 (Eq. 3.6 & 3.7) của Final.docx:
- z_tilde = A*z + b
- E_cons = MSE(g_psi(z_tilde), g_psi(z))
- L_{DECRA, demo} = 2 * E + 0.6 * E_cons + 0.1 * (K_n / K)
"""

from typing import Dict

import torch
import torch.nn as nn

from .dec_idec_loss import kl_divergence_loss, reconstruction_mse_loss


class DECRALoss(nn.Module):
    """Lớp tính Loss cho DECRA bản chuyển thể dùng chung Autoencoder."""

    def __init__(
        self,
        n_clusters: int = 10,
        w_rec: float = 2.0,
        eta_cons: float = 0.6,
        gamma_kl: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_clusters = max(int(n_clusters), 1)
        self.w_rec = w_rec
        self.eta_cons = eta_cons
        self.gamma_kl = gamma_kl

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        x_hat_tilde: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        e_rec = reconstruction_mse_loss(x, x_hat)
        e_cons = reconstruction_mse_loss(x_hat_tilde, x_hat)
        k_n = kl_divergence_loss(q, p)
        total = self.w_rec * e_rec + self.eta_cons * e_cons + (self.gamma_kl / self.n_clusters) * k_n
        return {
            "total": total,
            "rec": e_rec.detach(),
            "cons": e_cons.detach(),
            "kl": k_n.detach(),
        }
