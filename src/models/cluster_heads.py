"""
Các lớp đầu phân cụm (Clustering Heads) bám sát bài báo gốc DCSS (Vardakas et al., Machine Learning 2026):
1. StudentTClusterHead: Dùng cho DEC (2016), IDEC (2017), DECRA (2024), IDEC+BRB-inspired.
2. RBFClusterHead: Dùng cho DCSS (2026) gốc (Mục 4 & 5.3, Trang 9-16: RBF + Softmax với hệ số sắc nét T = 20,
   tức chia cho temperature = 1/20 = 0.05) và Đóng góp mới Ada-DCSS (Cluster-wise Adaptive EMA-RMS Bandwidth
   kết hợp Unsupervised Kernel Saturation Guard).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTClusterHead(nn.Module):
    """
    Đầu gán cụm mềm bằng phân phối Student-t (alpha = 1) theo Eq. (3) Trang 5 bài báo DCSS.
    """

    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def set_centroids(self, centroids: torch.Tensor) -> None:
        with torch.no_grad():
            self.centroids.copy_(centroids.to(self.centroids.device))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(z, self.centroids, p=2.0).pow(2)
        q = (1.0 + dist_sq / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-12)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        f_j = q.sum(dim=0, keepdim=True) + 1e-12
        weight = (q ** 2) / f_j
        p = weight / (weight.sum(dim=1, keepdim=True) + 1e-12)
        return p.detach()

    @torch.no_grad()
    def brb_reset_centroids(self, z_all: torch.Tensor, perturb_scale: float = 0.03) -> None:
        """
        Cơ chế Breaking the Reclustering Barrier (Miklautz et al., 2024 - được trích dẫn tại Mục 6 Trang 21 bài báo DCSS):
        Tái ước lượng tâm cụm theo phân hoạch hiện tại kèm nhiễu nhỏ để thoát điểm kẹt cục bộ.
        """
        q = self.forward(z_all)
        labels = q.argmax(dim=1)
        global_std = z_all.std(dim=0, keepdim=True).clamp(min=1e-4)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 2:
                new_c = z_all[mask].mean(dim=0)
                noise = torch.randn_like(new_c) * (global_std.squeeze(0) * perturb_scale)
                self.centroids[k].copy_(new_c + noise)


class RBFClusterHead(nn.Module):
    """
    Mạng phân cụm h_r(z) gồm tầng RBF nối tiếp Softmax theo Mục 4 & 5.3 (Trang 9, 10, 15) của bài báo DCSS:
        phi_ij = exp(-||z_i - mu_j||^2 / sigma_j^2)
        h_rj   = softmax_j(20 * phi_ij)   (tương đương temperature = 1/20 = 0.05)

    Hỗ trợ 3 chế độ băng thông `sigma_mode`:
    - "fixed": Giữ cố định sigma_j = sigma_init (kiểm chứng độ nhạy khi đặt hằng số cứng).
    - "static_rms": Khởi tạo bằng RMS khoảng cách trên không gian nhúng pretrained và cập nhật
      cùng tham số mạng phân cụm r theo Algorithm 1 (Trang 11).
    - "adaptive_ema_rms" (Đóng góp 1 - Ada-DCSS):
      Tự động điều chỉnh băng thông riêng cho từng cụm sigma_j^(t) theo EMA của khoảng cách nội cụm mềm,
      kết hợp cơ chế tự bảo vệ chống bão hòa nhân RBF hoàn toàn không giám sát (Unsupervised Saturation Guard).
    """

    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        temperature: float = 0.05,  # 1 / 20 -> tương ứng T = 20 tại Trang 15 bài báo DCSS
        sigma_mode: str = "static_rms",
        sigma_init: float = 1.0,
        ema_alpha: float = 0.15,
        normalize_latent_scale: bool = False,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.sigma_mode = sigma_mode
        self.sigma_init = sigma_init
        self.ema_alpha = ema_alpha
        self.normalize_latent_scale = normalize_latent_scale

        self.centroids = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.register_buffer("sigma", torch.full((n_clusters,), float(sigma_init)))

        # Trong chế độ static_rms (bám sát Algorithm 1 bài báo DCSS), cho phép tinh chỉnh tham số log_sigma trong r
        if sigma_mode == "static_rms":
            self.log_sigma_param = nn.Parameter(torch.zeros(n_clusters))
        else:
            self.register_parameter("log_sigma_param", None)

    def set_centroids(self, centroids: torch.Tensor) -> None:
        with torch.no_grad():
            self.centroids.copy_(centroids.to(self.centroids.device))

    @torch.no_grad()
    def initialize_sigma_from_embeddings(self, z_all: torch.Tensor) -> None:
        """Khởi tạo băng thông sigma_j theo chế độ được cấu hình."""
        if self.sigma_mode == "fixed":
            self.sigma.fill_(float(self.sigma_init))
            return

        dist_sq = torch.cdist(z_all, self.centroids, p=2.0).pow(2)
        min_dist_sq, nearest_cluster = dist_sq.min(dim=1)
        global_rms = torch.sqrt(dist_sq.mean().clamp(min=1e-6))

        if self.sigma_mode == "static_rms":
            self.sigma.fill_(float(global_rms.item()))
            if self.log_sigma_param is not None:
                self.log_sigma_param.copy_(torch.log(self.sigma.clamp(min=1e-4)))
        elif self.sigma_mode == "adaptive_ema_rms":
            # Nếu truyền sigma_init != 1.0 (kịch bản Sensitivity), khởi tạo đúng tại sigma_init
            # để kiểm chứng khả năng tự hiệu chỉnh không giám sát của Ada-DCSS / Ada-IM-DCSS
            if abs(self.sigma_init - 1.0) > 1e-5:
                self.sigma.fill_(float(self.sigma_init))
            else:
                cluster_rms_vec = torch.full((self.n_clusters,), float(global_rms.item()), device=z_all.device, dtype=z_all.dtype)
                for j in range(self.n_clusters):
                    mask = nearest_cluster == j
                    if mask.sum() > 2:
                        cluster_rms_vec[j] = torch.sqrt(min_dist_sq[mask].mean().clamp(min=1e-6))
                rel_disp = cluster_rms_vec / cluster_rms_vec.mean().clamp(min=1e-6)
                init_sigma = global_rms * (0.65 + 0.35 * rel_disp)
                self.sigma.copy_(torch.clamp(init_sigma, min=0.55 * global_rms, max=1.80 * global_rms))

    @torch.no_grad()
    def _compute_adaptive_target_sigma(self, dist_sq: torch.Tensor, q_weights: torch.Tensor) -> torch.Tensor:
        """
        Tính băng thông mục tiêu riêng cho từng cụm theo tỷ lệ phân tán nội cụm tương đối (bất biến thang đo Entropy)
        kết hợp neo theo global_rms và cân bằng nhẹ khối lượng cụm.
        """
        global_rms = torch.sqrt(dist_sq.mean().clamp(min=1e-6))
        cluster_mass = q_weights.sum(dim=0).clamp(min=1e-6)  # [K]
        weighted_intra_d2 = (q_weights * dist_sq).sum(dim=0) / cluster_mass  # [K]
        soft_intra_rms = torch.sqrt(weighted_intra_d2.clamp(min=1e-6))

        # Tỷ lệ phân tán tương đối r_j có trung bình bằng 1.0 (không bị phóng đại khi q còn mềm ở pha Warmup)
        rel_disp = soft_intra_rms / soft_intra_rms.mean().clamp(min=1e-6)

        p_bar = cluster_mass / q_weights.shape[0]
        uniform_p = 1.0 / float(self.n_clusters)
        mass_factor = (uniform_p / p_bar.clamp(min=1e-4)).pow(0.05).clamp(0.90, 1.12)

        target_sigma = global_rms * (0.65 + 0.35 * rel_disp) * mass_factor
        return torch.clamp(target_sigma, min=0.55 * global_rms, max=1.85 * global_rms)

    @torch.no_grad()
    def rescue_sigma_with_rms(self, z_batch: torch.Tensor) -> None:
        dist_sq = torch.cdist(z_batch, self.centroids, p=2.0).pow(2)
        rms_val = torch.sqrt(dist_sq.mean().clamp(min=1e-6))
        self.sigma.fill_(float(rms_val.item()))
        if self.log_sigma_param is not None:
            self.log_sigma_param.fill_(float(torch.log(rms_val).item()))

    def forward(self, z: torch.Tensor, update_sigma: bool = True) -> torch.Tensor:
        dist_sq = torch.cdist(z, self.centroids, p=2.0).pow(2)  # [B, K]

        if self.sigma_mode == "static_rms" and self.log_sigma_param is not None:
            active_sigma = torch.exp(self.log_sigma_param).clamp(min=0.15, max=4.5)
            with torch.no_grad():
                self.sigma.copy_(active_sigma.detach())
        elif self.sigma_mode == "adaptive_ema_rms":
            # Unsupervised Saturation Guard: kiểm tra vùng hoạt động của nhân RBF trước khi lan truyền ngược
            if self.training and update_sigma:
                with torch.no_grad():
                    cur_s2 = self.sigma.pow(2).unsqueeze(0).clamp(min=1e-8)
                    phi_probe = torch.exp(-(dist_sq.detach() / cur_s2).clamp(0.0, 60.0))
                    phi_mean = float(phi_probe.mean().item())
                    if phi_mean < 0.08 or phi_mean > 0.78:
                        # Khởi tạo gán mềm tạm bằng khoảng cách gần nhất để ước lượng RMS cục bộ không cần nhãn
                        global_s2 = dist_sq.detach().mean().clamp(min=1e-6)
                        q_probe = F.softmax(torch.exp(-dist_sq.detach() / global_s2) / max(self.temperature, 1e-4), dim=1)
                        rescued = self._compute_adaptive_target_sigma(dist_sq.detach(), q_probe)
                        self.sigma.copy_(rescued)
            active_sigma = self.sigma
        else:
            active_sigma = self.sigma

        sigma_sq = active_sigma.pow(2).unsqueeze(0).clamp(min=1e-8)
        phi = torch.exp(-dist_sq / sigma_sq)
        logits = phi / max(self.temperature, 1e-4)
        q = F.softmax(logits, dim=1)

        if self.training and update_sigma and self.sigma_mode == "adaptive_ema_rms":
            with torch.no_grad():
                target_s = self._compute_adaptive_target_sigma(dist_sq.detach(), q.detach())
                self.sigma.mul_(1.0 - self.ema_alpha).add_(target_s * self.ema_alpha)

        return q
