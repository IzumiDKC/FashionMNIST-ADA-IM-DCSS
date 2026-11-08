"""
Module quản lý cấu hình thực nghiệm (Hyperparameters, Protocol & Method Registry).
Đồng bộ chuẩn xác với bài báo gốc DCSS (Vardakas et al., Machine Learning 2026, 115:81, Mục 5.3 & 5.5)
và hệ thống cải tiến mới kết hợp:
1. Ada-DCSS (Per-Cluster Adaptive EMA-RMS Bandwidth + Unsupervised Saturation Guard)
2. IM-DCSS (Information Maximization: Marginal Balance + Scheduled Cosine Entropy Warmup)
3. Ada-IM-DCSS (Full Proposed: Hợp nhất Ada-DCSS và IM-DCSS, giữ nguyên tốc độ 1.0x).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """Cấu hình chung cho toàn bộ chiến dịch thực nghiệm (Shared Protocol theo bài báo DCSS 2026)."""

    dataset_name: str = "fashion-mnist"
    data_path: Optional[str] = None
    labels_path: Optional[str] = None
    output_dir: str = "outputs_fashion_mnist"

    # Kích thước dữ liệu & kiến trúc Autoencoder dùng chung
    input_dim: int = 784
    n_clusters: int = 10
    hidden_dims: Tuple[int, ...] = (500, 500, 2000)
    latent_dim: int = 10

    # Giao thức huấn luyện theo Mục 5.5 (Trang 16) bài báo DCSS (2026)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    pretrain_epochs: int = 80
    finetune_epochs: int = 40
    batch_size: int = 256
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    finetune_lr: float = 5e-4
    device: str = "auto"

    # Tùy chọn đánh giá mở rộng (Inductive Generalization)
    inductive_test_ratio: float = 0.0


@dataclass
class MethodConfig:
    """Cấu hình chi tiết cho từng phương pháp / biến thể phân cụm."""

    name: str
    family: str  # "kmeans" | "dec" | "idec" | "decra" | "brb" | "dcss"

    # Hệ số Reconstruction loss (IDEC, DECRA, DCSS)
    w_rec: float = 1.0

    # Hệ số KL clustering loss (DEC, IDEC, DECRA, BRB)
    gamma_kl: float = 0.1
    kl_update_interval: int = 1

    # Hệ số Consistency loss & nhánh biến đổi Residual (dùng cho đối chứng DECRA 2024)
    use_affine_branch: bool = False
    eta_cons: float = 0.0
    cons_on_assignments: bool = False

    # Cấu hình BRB-inspired (Miklautz et al., 2024)
    brb_reset_interval: int = 10
    brb_perturb_scale: float = 0.025

    # Cấu hình chuẩn bài báo DCSS (Trang 15 - 16): lambda_1 = 0.01, lambda_2 = 0.01, T = 20 (temperature = 0.05)
    lambda_sil: float = 0.01
    beta_ent: float = 0.01
    balance_weight: float = 0.0
    temperature: float = 0.05
    sigma_mode: str = "static_rms"   # "fixed" | "static_rms" | "adaptive_ema_rms"
    sigma_init: float = 1.0
    ema_alpha: float = 0.15
    normalize_latent_scale: bool = False

    # Cấu hình Cải tiến IM-DCSS (Information Maximization & Scheduled Entropy Warmup)
    use_infomax: bool = False
    lambda_bal: float = 0.004
    entropy_warmup_ratio: float = 0.65

    # Bộ giám sát không giám sát & tự phục hồi khi sụp đổ cụm
    use_unsup_monitor: bool = True
    auto_rescue_collapse: bool = False


def get_standard_benchmark_configs() -> List[MethodConfig]:
    """
    Trả về 10 cấu hình đối chứng chuẩn:
    - 7 cấu hình đối chứng gốc (AE+K-means, DEC, IDEC, DECRA, IDEC+BRB, DCSS sigma=1, DCSS RMS)
    - 3 cấu hình Đóng góp cải tiến mới (Ada-DCSS, IM-DCSS, Ada-IM-DCSS).
    """
    return [
        # --- 7 Cấu hình đối chứng chuẩn ---
        MethodConfig(
            name="AE + K-means",
            family="kmeans",
            w_rec=0.0,
            gamma_kl=0.0,
        ),
        MethodConfig(
            name="DEC (2016)",
            family="dec",
            w_rec=0.0,
            gamma_kl=1.0,
        ),
        MethodConfig(
            name="IDEC (2017)",
            family="idec",
            w_rec=1.0,
            gamma_kl=0.1,
        ),
        MethodConfig(
            name="DECRA (2024)",
            family="decra",
            w_rec=1.0,
            gamma_kl=0.5,
            use_affine_branch=True,
            eta_cons=0.5,
        ),
        MethodConfig(
            name="IDEC + BRB-inspired",
            family="brb",
            w_rec=1.0,
            gamma_kl=0.05,
            brb_reset_interval=10,
            brb_perturb_scale=0.025,
        ),
        MethodConfig(
            name="DCSS (sigma=1)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.01,
            beta_ent=0.01,
            temperature=0.05,
            sigma_mode="fixed",
            sigma_init=1.0,
        ),
        MethodConfig(
            name="DCSS (RMS)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.01,
            beta_ent=0.01,
            temperature=0.05,
            sigma_mode="static_rms",
        ),
        # --- 3 Cấu hình Đóng góp Cải tiến Mới (Ada-DCSS, IM-DCSS, Ada-IM-DCSS) ---
        MethodConfig(
            name="Ada-DCSS (Ours - EMA Per-Cluster Sigma)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.012,
            beta_ent=0.010,
            balance_weight=0.15,
            temperature=0.05,
            sigma_mode="adaptive_ema_rms",
            ema_alpha=0.15,
            normalize_latent_scale=True,
            auto_rescue_collapse=True,
        ),
        MethodConfig(
            name="IM-DCSS (Ours - InfoMax & Entropy Warmup)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.012,
            beta_ent=0.0025,
            temperature=0.05,
            sigma_mode="static_rms",
            use_infomax=True,
            lambda_bal=0.004,
            entropy_warmup_ratio=0.70,
        ),
        MethodConfig(
            name="Ada-IM-DCSS (Ours - Full Proposed)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.012,
            beta_ent=0.0025,
            temperature=0.05,
            sigma_mode="adaptive_ema_rms",
            ema_alpha=0.15,
            normalize_latent_scale=True,
            use_infomax=True,
            lambda_bal=0.004,
            entropy_warmup_ratio=0.70,
            auto_rescue_collapse=True,
        ),
    ]


def get_dcss_ablation_configs() -> List[MethodConfig]:
    """Bảng Ablation 5 cấp độ chứng minh trực tiếp vai trò của từng thành phần từ DCSS gốc tới Ada-IM-DCSS."""
    return [
        MethodConfig(
            name="Ablation 1: Only Soft-Silhouette (No Rec, No Ent)",
            family="dcss",
            w_rec=0.0,
            lambda_sil=0.01,
            beta_ent=0.0,
            temperature=0.05,
            sigma_mode="static_rms",
        ),
        MethodConfig(
            name="Ablation 2: Rec + Soft-Silhouette (No Entropy)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.01,
            beta_ent=0.0,
            temperature=0.05,
            sigma_mode="static_rms",
        ),
        MethodConfig(
            name="Ablation 3: Full DCSS (Static RMS + Eq.20 Entropy)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.01,
            beta_ent=0.01,
            temperature=0.05,
            sigma_mode="static_rms",
        ),
        MethodConfig(
            name="Ablation 4: IM-DCSS (Soft-Silhouette + InfoMax Warmup)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.012,
            beta_ent=0.0025,
            temperature=0.05,
            sigma_mode="static_rms",
            use_infomax=True,
            lambda_bal=0.004,
            entropy_warmup_ratio=0.70,
        ),
        MethodConfig(
            name="Ablation 5: Ada-IM-DCSS (Per-Cluster EMA + InfoMax - Full Ours)",
            family="dcss",
            w_rec=1.0,
            lambda_sil=0.012,
            beta_ent=0.0025,
            temperature=0.05,
            sigma_mode="adaptive_ema_rms",
            ema_alpha=0.15,
            normalize_latent_scale=True,
            use_infomax=True,
            lambda_bal=0.004,
            entropy_warmup_ratio=0.70,
            auto_rescue_collapse=True,
        ),
    ]


def get_sigma_sensitivity_configs(
    sigma_values: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)
) -> List[MethodConfig]:
    """
    Kịch bản kiểm tra độ bền siêu tham số (Sensitivity Matrix):
    So sánh DCSS gốc (Fixed sigma) với Ada-IM-DCSS (Ours) qua 5 mức khởi tạo sigma_init lệch thang đo.
    """
    configs: List[MethodConfig] = []
    for s in sigma_values:
        configs.append(
            MethodConfig(
                name=f"DCSS Fixed (sigma_init={s})",
                family="dcss",
                w_rec=1.0,
                lambda_sil=0.01,
                beta_ent=0.01,
                temperature=0.05,
                sigma_mode="fixed",
                sigma_init=s,
            )
        )
        configs.append(
            MethodConfig(
                name=f"Ada-IM-DCSS Ours (sigma_init={s})",
                family="dcss",
                w_rec=1.0,
                lambda_sil=0.012,
                beta_ent=0.0025,
                temperature=0.05,
                sigma_mode="adaptive_ema_rms",
                sigma_init=s,
                ema_alpha=0.15,
                normalize_latent_scale=True,
                use_infomax=True,
                lambda_bal=0.004,
                entropy_warmup_ratio=0.70,
                auto_rescue_collapse=True,
            )
        )
    return configs
