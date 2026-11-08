"""
Engine tính toán Vector hóa thuần NumPy (Pure Vectorized NumPy Autodiff & Adam Engine).
Cho phép chạy toàn bộ 10 cấu hình thực nghiệm (AE+KMeans, DEC, IDEC, DECRA, IDEC+BRB,
DCSS sigma=1, DCSS RMS, và 3 cấu hình đóng góp mới: Ada-DCSS, Consist-DCSS, Ada-Consist-DCSS)
trực tiếp trên môi trường NumPy/SciPy/Scikit-learn mà không bắt buộc cài đặt PyTorch.
"""

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ExperimentConfig, MethodConfig
from ..evaluation.metrics import compute_all_supervised_metrics, compute_unsupervised_metrics
from .unsup_monitor import UnsupervisedCollapseMonitor


class AdamOptimizerNumPy:
    """Bộ tối ưu hóa Adam chuẩn (Kingma & Ba, 2015) viết thuần NumPy."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray], max_grad_norm: float = 5.0) -> None:
        self.t += 1
        # Clip gradient norm toàn cục để đảm bảo ổn định số học
        total_norm_sq = sum(float(np.sum(g ** 2)) for g in grads.values() if g is not None)
        total_norm = np.sqrt(total_norm_sq)
        clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-12))

        b1_corr = 1.0 - self.beta1 ** self.t
        b2_corr = 1.0 - self.beta2 ** self.t

        for name, param in self.params.items():
            g = grads.get(name)
            if g is None:
                continue
            g = g * clip_coef
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[name] / b1_corr
            v_hat = self.v[name] / b2_corr
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumPySharedAutoencoder:
    """
    Mạng Autoencoder dùng chung + Nhánh biến đổi Affine (z_tilde = z @ A^T + b)
    hỗ trợ lan truyền thuận và lan truyền ngược (backpropagation) giải tích chính xác.
    Kiến trúc:
    Input (D) -> Linear+ReLU (H1) -> Linear (latent_dim) -> Linear+ReLU (H1) -> Linear+Sigmoid (D)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 10,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Khởi tạo Kaiming / Xavier chuẩn
        self.params: Dict[str, np.ndarray] = {
            "W_enc1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32),
            "b_enc1": np.zeros(hidden_dim, dtype=np.float32),
            "W_enc2": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, latent_dim)).astype(np.float32),
            "b_enc2": np.zeros(latent_dim, dtype=np.float32),
            "W_dec1": rng.normal(0.0, np.sqrt(2.0 / latent_dim), size=(latent_dim, hidden_dim)).astype(np.float32),
            "b_dec1": np.zeros(hidden_dim, dtype=np.float32),
            "W_dec2": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, input_dim)).astype(np.float32),
            "b_dec2": np.zeros(input_dim, dtype=np.float32),
            # Nhánh biến đổi Affine (cho DECRA và Consist-DCSS): z_tilde = z @ W_aff + b_aff
            "W_aff": (np.eye(latent_dim) + rng.normal(0.0, 0.02, size=(latent_dim, latent_dim))).astype(np.float32),
            "b_aff": np.zeros(latent_dim, dtype=np.float32),
        }

    def reset_affine_identity(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.params["W_aff"] = (
            np.eye(self.latent_dim) + rng.normal(0.0, 0.005, size=(self.latent_dim, self.latent_dim))
        ).astype(np.float32)
        self.params["b_aff"] = np.zeros(self.latent_dim, dtype=np.float32)

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            self.params[k] = v.copy()

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        h1_pre = x @ self.params["W_enc1"] + self.params["b_enc1"]
        h1 = np.maximum(0.0, h1_pre)
        z = h1 @ self.params["W_enc2"] + self.params["b_enc2"]
        cache = {"x": x, "h1_pre": h1_pre, "h1": h1, "z": z}
        return z, cache

    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        g1_pre = z @ self.params["W_dec1"] + self.params["b_dec1"]
        g1 = np.maximum(0.0, g1_pre)
        logits = g1 @ self.params["W_dec2"] + self.params["b_dec2"]
        logits_clip = np.clip(logits, -15.0, 15.0)
        x_hat = 1.0 / (1.0 + np.exp(-logits_clip))
        cache = {"z_in": z, "g1_pre": g1_pre, "g1": g1, "x_hat": x_hat}
        return x_hat, cache

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
        z, enc_cache = self.encode(x)
        x_hat, dec_cache = self.decode(z)
        return z, x_hat, {"enc": enc_cache, "dec": dec_cache}

    def forward_affine(self, z: np.ndarray) -> np.ndarray:
        return z @ self.params["W_aff"] + self.params["b_aff"]

    def backward_decoder(
        self, grad_x_hat: np.ndarray, dec_cache: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Lan truyền ngược qua Decoder: trả về (grad_z, grads_decoder)."""
        x_hat = dec_cache["x_hat"]
        g1 = dec_cache["g1"]
        g1_pre = dec_cache["g1_pre"]
        z_in = dec_cache["z_in"]

        d_logits = grad_x_hat * x_hat * (1.0 - x_hat)  # [B, D]
        dW_dec2 = g1.T @ d_logits
        db_dec2 = np.sum(d_logits, axis=0)

        dg1 = d_logits @ self.params["W_dec2"].T
        dg1_pre = dg1 * (g1_pre > 0.0)
        dW_dec1 = z_in.T @ dg1_pre
        db_dec1 = np.sum(dg1_pre, axis=0)

        grad_z = dg1_pre @ self.params["W_dec1"].T
        grads = {
            "W_dec2": dW_dec2,
            "b_dec2": db_dec2,
            "W_dec1": dW_dec1,
            "b_dec1": db_dec1,
        }
        return grad_z, grads

    def backward_encoder(
        self, grad_z: np.ndarray, enc_cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Lan truyền ngược qua Encoder từ grad_z."""
        x = enc_cache["x"]
        h1 = enc_cache["h1"]
        h1_pre = enc_cache["h1_pre"]

        dW_enc2 = h1.T @ grad_z
        db_enc2 = np.sum(grad_z, axis=0)

        dh1 = grad_z @ self.params["W_enc2"].T
        dh1_pre = dh1 * (h1_pre > 0.0)
        dW_enc1 = x.T @ dh1_pre
        db_enc1 = np.sum(dh1_pre, axis=0)

        return {
            "W_enc2": dW_enc2,
            "b_enc2": db_enc2,
            "W_enc1": dW_enc1,
            "b_enc1": db_enc1,
        }


# =====================================================================
# Các hàm tính Loss & Đạo hàm giải tích cho DEC / IDEC / DECRA / DCSS
# =====================================================================

def student_t_forward(z: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. (3.2) Final.docx: q_ij = (1 + ||z_i - mu_j||^2)^(-1) / sum_k (...)"""
    diff = z[:, None, :] - centroids[None, :, :]  # [B, K, d]
    dist_sq = np.sum(diff ** 2, axis=-1)  # [B, K]
    inv = 1.0 / (1.0 + dist_sq)
    q = inv / (np.sum(inv, axis=1, keepdims=True) + 1e-12)
    return q, inv


def student_t_target_p(q: np.ndarray) -> np.ndarray:
    """Eq. (3.3) Final.docx: p_ij = (q_ij^2 / f_j) / sum_k (q_ik^2 / f_k)"""
    f_j = np.sum(q, axis=0, keepdims=True) + 1e-12
    w = (q ** 2) / f_j
    return w / (np.sum(w, axis=1, keepdims=True) + 1e-12)


def student_t_kl_backward(
    z: np.ndarray,
    centroids: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    inv_dist: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Tính giá trị phân kỳ KL K_n (Eq. 3.4) và đạo hàm giải tích chính xác dK_n/dz, dK_n/d(centroids)
    theo Xie et al. (ICML 2016, Eq. 4 & 5).
    """
    batch_size = float(z.shape[0])
    q_safe = np.clip(q, 1e-10, 1.0)
    p_safe = np.clip(p, 1e-10, 1.0)
    kl_val = float(np.mean(np.sum(p_safe * np.log(p_safe / q_safe), axis=1)))

    # (p_ij - q_ij) * (1 + ||z_i - mu_j||^2)^(-1)
    coeff = (p - q) * inv_dist * (2.0 / batch_size)  # [B, K]
    diff = z[:, None, :] - centroids[None, :, :]  # [B, K, d]

    grad_z = np.sum(coeff[:, :, None] * diff, axis=1)  # [B, d]
    grad_centroids = -np.sum(coeff[:, :, None] * diff, axis=0)  # [K, d]
    return kl_val, grad_z.astype(np.float32), grad_centroids.astype(np.float32)


def rbf_forward(
    z: np.ndarray,
    centroids: np.ndarray,
    sigma: np.ndarray,
    temperature: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Eq. (3.8) Final.docx:
    phi_ij = exp(-||z_i - mu_j||^2 / sigma_j^2)
    q_ij   = softmax_j(phi_ij / T)
    """
    diff = z[:, None, :] - centroids[None, :, :]  # [B, K, d]
    dist_sq = np.sum(diff ** 2, axis=-1)  # [B, K]
    sigma_sq = np.maximum(sigma ** 2, 1e-8)[None, :]  # [1, K]
    phi = np.exp(-np.clip(dist_sq / sigma_sq, 0.0, 60.0))  # [B, K]

    logits = phi / max(temperature, 1e-4)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(logits)
    q = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-12)

    cache = {
        "diff": diff,
        "phi": phi,
        "sigma_sq": sigma_sq,
        "q": q,
        "temperature": temperature,
    }
    return q.astype(np.float32), cache


def rbf_backward(
    grad_q: np.ndarray, rbf_cache: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Lan truyền ngược từ grad_q qua Softmax và nhân Gaussian RBF về z và centroids."""
    q = rbf_cache["q"]
    phi = rbf_cache["phi"]
    sigma_sq = rbf_cache["sigma_sq"]
    diff = rbf_cache["diff"]
    temp = rbf_cache["temperature"]

    # Đạo hàm qua softmax(phi / T)
    sum_q_dq = np.sum(q * grad_q, axis=1, keepdims=True)
    d_phi = q * (grad_q - sum_q_dq) / max(temp, 1e-4)  # [B, K]

    # d(phi)/d(dist_sq) = -phi / sigma_j^2
    g_ij = d_phi * (-2.0 * phi / sigma_sq)  # [B, K]

    grad_z = np.sum(g_ij[:, :, None] * diff, axis=1)  # [B, d]
    grad_centroids = -np.sum(g_ij[:, :, None] * diff, axis=0)  # [K, d]
    return grad_z.astype(np.float32), grad_centroids.astype(np.float32)


def soft_silhouette_forward_backward(
    z: np.ndarray, q: np.ndarray, eps: float = 1e-7
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Tính điểm Soft Silhouette khả vi S_soft (Mục 3.5 Final.docx) và đạo hàm giải tích
    d(1 - S_soft)/dz, d(1 - S_soft)/dq trên mini-batch.
    """
    B, K = q.shape
    if B <= 2:
        return 0.0, np.zeros_like(z), np.zeros_like(q)

    # 1. Ma trận khoảng cách cặp Euclid D_{ik} = ||z_i - z_k||_2
    diff_zz = z[:, None, :] - z[None, :, :]  # [B, B, d]
    dist_sq = np.sum(diff_zz ** 2, axis=-1)
    eye_b = np.eye(B, dtype=np.float32)
    dist = np.sqrt(dist_sq + eye_b * 1e-12) * (1.0 - eye_b)  # [B, B]

    # 2. Khoảng cách trung bình từ mẫu i tới từng cụm c: a_{ic} = d(i, c) = N_{ic} / M_{ic}
    q_sum = np.sum(q, axis=0, keepdims=True)  # [1, K]
    M_ic = np.maximum(q_sum - q, eps)  # [B, K]
    N_ic = dist @ q  # [B, K]
    d_ic = N_ic / M_ic  # [B, K]

    # 3. Khoảng cách liên cụm gần nhất mềm b_{ic} = softmin_{c' != c} d(i, c')
    tau_b = 0.1
    diag_k = np.eye(K, dtype=np.float32)[None, :, :]  # [1, K, K]
    d_exp = np.broadcast_to(d_ic[:, None, :], (B, K, K))  # [B, K, K]
    logits_other = -(d_exp + diag_k * 1e4) / tau_b
    logits_other = logits_other - np.max(logits_other, axis=2, keepdims=True)
    exp_other = np.exp(logits_other) * (1.0 - diag_k)
    w_other = exp_other / (np.sum(exp_other, axis=2, keepdims=True) + 1e-12)  # [B, K, K]
    b_ic = np.sum(w_other * d_exp, axis=2)  # [B, K]

    a_ic = d_ic
    denom_ab = np.maximum(np.maximum(a_ic, b_ic), eps)  # [B, K]
    s_ic = (b_ic - a_ic) / denom_ab  # [B, K]

    s_i = np.sum(q * s_ic, axis=1)  # [B]
    s_soft = float(np.mean(s_i))

    # --- Lan truyền ngược cho L_sil = 1 - S_soft (tức là dL_sil = -dS_soft) ---
    scale = -1.0 / float(B)
    grad_q_direct = scale * s_ic  # [B, K]
    ds_ic = scale * q  # [B, K]

    # s_ic = (b - a) / max(a, b)
    a_gt_b = (a_ic >= b_ic).astype(np.float32)
    b_gt_a = 1.0 - a_gt_b
    # Nếu a >= b: s = b/a - 1 -> ds/da = -b/a^2, ds/db = 1/a
    # Nếu b > a : s = 1 - a/b -> ds/da = -1/b, ds/db = a/b^2
    da_ic = ds_ic * (a_gt_b * (-b_ic / (denom_ab ** 2)) + b_gt_a * (-1.0 / denom_ab))
    db_ic = ds_ic * (a_gt_b * (1.0 / denom_ab) + b_gt_a * (a_ic / (denom_ab ** 2)))

    # Lan truyền ngược từ b_ic về d_ic qua softmin
    # b_{ic} = sum_{c'} w_{i,c,c'} d_{i,c'}
    # dw/dd_{i,c'} = -(1/tau_b) * w * (d_{i,c'} - b_{ic})
    softmin_factor = w_other * (1.0 - (d_exp - b_ic[:, :, None]) / tau_b)  # [B, K, K]
    dd_from_b = np.sum(db_ic[:, :, None] * softmin_factor, axis=1)  # [B, K]

    dd_ic = da_ic + dd_from_b  # [B, K]

    # d_ic = N_ic / M_ic, với N_ic = dist @ q và M_ic = sum_{k != i} q_{kc}
    dN_ic = dd_ic / M_ic  # [B, K]
    dM_ic = -dd_ic * d_ic / M_ic  # [B, K]

    # Đạo hàm theo q từ N_ic và M_ic
    grad_q_from_N = dist.T @ dN_ic  # [B, K]
    grad_q_from_M = np.sum(dM_ic, axis=0, keepdims=True) - dM_ic  # [B, K]
    grad_q = grad_q_direct + grad_q_from_N + grad_q_from_M

    # Đạo hàm theo ma trận khoảng cách dist [B, B] -> theo z [B, d]
    d_dist = dN_ic @ q.T  # [B, B]
    d_dist_sym = (d_dist + d_dist.T) * (1.0 - eye_b)
    inv_dist = 1.0 / (dist + eye_b + 1e-8) * (1.0 - eye_b)
    coef_zz = d_dist_sym * inv_dist  # [B, B]
    grad_z = 2.0 * np.sum(coef_zz[:, :, None] * diff_zz, axis=1)  # [B, d]

    return s_soft, grad_z.astype(np.float32), grad_q.astype(np.float32)


def entropy_reg_forward_backward(q: np.ndarray, eps: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    Tính điều chuẩn Entropy H_2(Q) (Eq. 3.9 Final.docx) và đạo hàm dH_2(Q)/dq:
    Kết hợp Conditional Entropy (làm sắc nét gán cụm) và Marginal Cluster Balance KL
    (ngăn mọi mẫu sụp đổ vào 1 cụm hoặc nuốt cụm nhỏ).
    """
    B, K = q.shape
    q_safe = np.clip(q, eps, 1.0)

    cond_ent = -float(np.mean(np.sum(q_safe * np.log(q_safe), axis=1)))
    grad_cond = -(1.0 + np.log(q_safe)) / float(B)

    p_bar = np.clip(np.mean(q_safe, axis=0), eps, 1.0)
    bal_kl = float(np.sum(p_bar * np.log(p_bar * K)))
    grad_bal = np.broadcast_to((1.0 + np.log(p_bar * K))[None, :] / float(B), (B, K))

    # Trọng số 2.5 cho thành phần cân bằng cụm (Cluster Balance Prior) để giữ đủ K cụm hoạt động
    return cond_ent + 2.5 * bal_kl, (0.4 * grad_cond + 2.5 * grad_bal).astype(np.float32)


@dataclass
class NumPyPretrainState:
    """Trạng thái sau tiền huấn luyện Shared Autoencoder + K-Means (thuần NumPy)."""

    seed: int
    ae_state_dict: Dict[str, np.ndarray]
    initial_centroids: np.ndarray
    kmeans_pred: np.ndarray
    kmeans_unseen_pred: np.ndarray
    pretrain_time_sec: float
    initial_rec_loss: float


@dataclass
class RunResult:
    """Kết quả chi tiết của 1 lượt chạy (1 cấu hình x 1 seed)."""

    method_name: str
    family: str
    seed: int
    acc: float
    nmi: float
    ari: float
    unsup_silhouette: float
    active_clusters: float
    assignment_entropy: float
    finetune_time_sec: float
    pretrain_time_sec: float
    final_sigma_mean: float
    final_sigma_std: float
    collapse_rescued: int
    predictions: np.ndarray
    inductive_acc: float = float("nan")
    inductive_nmi: float = float("nan")
    inductive_ari: float = float("nan")


def _run_kmeans_init(z: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=150, random_state=seed)
    labels = km.fit_predict(z)
    return km.cluster_centers_.astype(np.float32), labels.astype(np.int64)


def pretrain_shared_autoencoder_numpy(
    X_train: np.ndarray,
    exp_cfg: ExperimentConfig,
    seed: int,
    X_unseen: Optional[np.ndarray] = None,
) -> NumPyPretrainState:
    """Tiền huấn luyện Shared Autoencoder chung và khởi tạo K-Means đúng 1 lần trên mỗi seed."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    hidden_dim = exp_cfg.hidden_dims[0] if exp_cfg.hidden_dims else 256
    ae = NumPySharedAutoencoder(
        input_dim=exp_cfg.input_dim,
        hidden_dim=hidden_dim,
        latent_dim=exp_cfg.latent_dim,
        seed=seed,
    )
    opt_params = {k: ae.params[k] for k in ["W_enc1", "b_enc1", "W_enc2", "b_enc2", "W_dec1", "b_dec1", "W_dec2", "b_dec2"]}
    optimizer = AdamOptimizerNumPy(opt_params, lr=exp_cfg.pretrain_lr)

    n_samples, input_dim = X_train.shape
    batch_size = exp_cfg.batch_size
    last_mse = 0.0

    for _ in range(exp_cfg.pretrain_epochs):
        perm = rng.permutation(n_samples)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            batch_x = X_train[perm[start : start + batch_size]]
            b_len = float(batch_x.shape[0])
            _, x_hat, cache = ae.forward(batch_x)
            diff = x_hat - batch_x
            mse = float(np.mean(diff ** 2))
            grad_x_hat = (2.0 / (b_len * input_dim)) * diff

            grad_z, grads_dec = ae.backward_decoder(grad_x_hat, cache["dec"])
            grads_enc = ae.backward_encoder(grad_z, cache["enc"])
            grads = {**grads_dec, **grads_enc}
            optimizer.step(grads)

            ep_loss += mse
            n_batches += 1
        last_mse = ep_loss / max(n_batches, 1)

    z_all, _ = ae.encode(X_train)
    centroids, kmeans_pred = _run_kmeans_init(z_all, exp_cfg.n_clusters, seed=seed)

    kmeans_unseen_pred = np.array([], dtype=np.int64)
    if X_unseen is not None and len(X_unseen) > 0:
        z_unseen, _ = ae.encode(X_unseen)
        d_unseen = np.sum((z_unseen[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        kmeans_unseen_pred = np.argmin(d_unseen, axis=1).astype(np.int64)

    elapsed = time.perf_counter() - t0
    return NumPyPretrainState(
        seed=seed,
        ae_state_dict=ae.get_state_dict(),
        initial_centroids=centroids,
        kmeans_pred=kmeans_pred,
        kmeans_unseen_pred=kmeans_unseen_pred,
        pretrain_time_sec=elapsed,
        initial_rec_loss=last_mse,
    )


def _initialize_rbf_sigma(
    z_all: np.ndarray,
    centroids: np.ndarray,
    method_cfg: MethodConfig,
    n_clusters: int,
) -> np.ndarray:
    """Khởi tạo vector băng thông sigma_j cho DCSS / Ada-DCSS."""
    if method_cfg.sigma_mode == "fixed":
        return np.full(n_clusters, float(method_cfg.sigma_init), dtype=np.float32)

    dist_sq = np.sum((z_all[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    min_dist_sq = np.min(dist_sq, axis=1)
    nearest_c = np.argmin(dist_sq, axis=1)
    global_rms = float(np.sqrt(max(np.mean(dist_sq), 1e-6)))

    if method_cfg.sigma_mode == "static_rms":
        return np.full(n_clusters, global_rms, dtype=np.float32)

    # Chế độ adaptive_ema_rms (Ada-DCSS)
    if abs(method_cfg.sigma_init - 1.0) > 1e-5:
        # Trong kịch bản kiểm tra độ nhạy sigma_init != 1.0: khởi tạo từ đúng sigma_init
        return np.full(n_clusters, float(method_cfg.sigma_init), dtype=np.float32)

    c_rms_arr = np.full(n_clusters, global_rms, dtype=np.float32)
    for j in range(n_clusters):
        mask = nearest_c == j
        if np.sum(mask) > 2:
            c_rms_arr[j] = float(np.sqrt(max(np.mean(min_dist_sq[mask]), 1e-6)))
    rel_scale = c_rms_arr / max(float(np.mean(c_rms_arr)), 1e-6)
    sigmas = global_rms * (0.75 + 0.25 * rel_scale)
    return sigmas.astype(np.float32)


def _update_ema_sigma_numpy(
    sigma: np.ndarray,
    z: np.ndarray,
    centroids: np.ndarray,
    q: np.ndarray,
    ema_alpha: float,
    auto_rescue: bool = True,
) -> np.ndarray:
    """
    Đóng góp 1 (Ada-DCSS): Cập nhật băng thông từng cụm sigma_j^(t) theo EMA của khoảng cách nội cụm mềm
    được neo theo thang đo toàn cục (Scale-Anchored Per-Cluster EMA-RMS Bandwidth),
    kèm Điều kiện Tự hiệu chỉnh Nhân RBF Không giám sát (Unsupervised Kernel Saturation Guard).
    """
    dist_sq = np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # [B, K]
    sigma_sq = np.maximum(sigma ** 2, 1e-8)[None, :]
    phi = np.exp(-np.clip(dist_sq / sigma_sq, 0.0, 60.0))
    phi_mean = float(np.mean(phi))

    cluster_mass = np.maximum(np.sum(q, axis=0), 1e-6)  # [K]
    p_bar = cluster_mass / float(z.shape[0])
    K_val = float(centroids.shape[0])
    weighted_dist_sq = np.sum(q * dist_sq, axis=0) / cluster_mass  # [K]

    global_rms = float(np.sqrt(max(np.mean(dist_sq), 1e-6)))
    soft_cluster_rms = np.sqrt(np.maximum(weighted_dist_sq, 1e-6))
    rel_cluster_scale = soft_cluster_rms / max(float(np.mean(soft_cluster_rms)), 1e-6)

    # Hệ số cân bằng khối lượng cụm (Mass-Balanced Bandwidth Factor):
    # Cụm đang thiếu mẫu (p_bar < 1/K) được mở rộng nhẹ băng thông để thu hút mẫu biên
    mass_factor = np.clip(((1.0 / K_val) / np.maximum(p_bar, 1e-4)) ** 0.18, 0.85, 1.22)
    target_sigma = (0.65 * global_rms + 0.35 * soft_cluster_rms) * mass_factor
    target_sigma = np.clip(target_sigma, 0.35 * global_rms, 2.50 * global_rms)

    # Nếu nhân RBF bị bão hòa hoàn toàn (sigma quá nhỏ -> phi ~ 0, hoặc quá lớn -> phi ~ 1)
    if auto_rescue and (phi_mean < 0.08 or phi_mean > 0.78):
        return target_sigma.astype(np.float32)

    return ((1.0 - ema_alpha) * sigma + ema_alpha * target_sigma).astype(np.float32)


def train_clustering_method_numpy(
    method_cfg: MethodConfig,
    exp_cfg: ExperimentConfig,
    pretrain_state: NumPyPretrainState,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray],
    X_unseen: Optional[np.ndarray] = None,
    y_unseen: Optional[np.ndarray] = None,
) -> RunResult:
    """Huấn luyện 1 cấu hình phân cụm sâu bằng Engine Vector hóa NumPy."""
    seed = pretrain_state.seed
    rng = np.random.default_rng(seed)

    hidden_dim = exp_cfg.hidden_dims[0] if exp_cfg.hidden_dims else 256
    ae = NumPySharedAutoencoder(
        input_dim=exp_cfg.input_dim,
        hidden_dim=hidden_dim,
        latent_dim=exp_cfg.latent_dim,
        seed=seed,
    )
    ae.load_state_dict(pretrain_state.ae_state_dict)
    ae.reset_affine_identity(seed=seed)

    # 1. Cấu hình đối chứng AE + K-means
    if method_cfg.family == "kmeans":
        z_all, _ = ae.encode(X_train)
        q_km = np.eye(exp_cfg.n_clusters, dtype=np.float32)[pretrain_state.kmeans_pred]
        unsup = compute_unsupervised_metrics(z_all, q_km, seed=seed)
        sup = compute_all_supervised_metrics(y_train, pretrain_state.kmeans_pred)
        ind_sup = (
            compute_all_supervised_metrics(y_unseen, pretrain_state.kmeans_unseen_pred)
            if y_unseen is not None and len(pretrain_state.kmeans_unseen_pred) > 0
            else {"ACC": float("nan"), "NMI": float("nan"), "ARI": float("nan")}
        )
        return RunResult(
            method_name=method_cfg.name,
            family=method_cfg.family,
            seed=seed,
            acc=sup["ACC"],
            nmi=sup["NMI"],
            ari=sup["ARI"],
            unsup_silhouette=unsup["unsup_silhouette"],
            active_clusters=unsup["active_clusters"],
            assignment_entropy=unsup["assignment_entropy"],
            finetune_time_sec=0.0,
            pretrain_time_sec=pretrain_state.pretrain_time_sec,
            final_sigma_mean=0.0,
            final_sigma_std=0.0,
            collapse_rescued=0,
            predictions=pretrain_state.kmeans_pred,
            inductive_acc=ind_sup["ACC"],
            inductive_nmi=ind_sup["NMI"],
            inductive_ari=ind_sup["ARI"],
        )

    # 2. Các cấu hình phân cụm sâu
    centroids = pretrain_state.initial_centroids.copy()
    is_rbf = method_cfg.family == "dcss"
    z_init, _ = ae.encode(X_train)
    sigma = (
        _initialize_rbf_sigma(z_init, centroids, method_cfg, exp_cfg.n_clusters)
        if is_rbf
        else np.ones(exp_cfg.n_clusters, dtype=np.float32)
    )

    opt_params = {**ae.params, "centroids": centroids}
    optimizer = AdamOptimizerNumPy(opt_params, lr=exp_cfg.finetune_lr)
    monitor = UnsupervisedCollapseMonitor(n_clusters=exp_cfg.n_clusters)
    collapse_rescued = 0

    n_samples, input_dim = X_train.shape
    batch_size = exp_cfg.batch_size
    t_start = time.perf_counter()

    for epoch in range(exp_cfg.finetune_epochs):
        # Cơ chế BRB-inspired: Reset / Reclustering tâm cụm định kỳ
        if (
            method_cfg.family == "brb"
            and epoch > 0
            and epoch % max(method_cfg.brb_reset_interval, 1) == 0
        ):
            z_full, _ = ae.encode(X_train)
            q_full, _ = student_t_forward(z_full, centroids)
            lbls = np.argmax(q_full, axis=1)
            g_std = np.maximum(np.std(z_full, axis=0), 1e-4)
            for k in range(exp_cfg.n_clusters):
                mask = lbls == k
                if np.sum(mask) > 1:
                    centroids[k] = np.mean(z_full[mask], axis=0) + rng.normal(
                        0.0, g_std * method_cfg.brb_perturb_scale
                    ).astype(np.float32)

        # Cập nhật phân phối mục tiêu P đầu mỗi epoch cho họ Student-t (DEC, IDEC, DECRA, BRB)
        target_p_all = None
        if not is_rbf:
            z_all, _ = ae.encode(X_train)
            z_q = ae.forward_affine(z_all) if method_cfg.family == "decra" else z_all
            q_all, _ = student_t_forward(z_q, centroids)
            target_p_all = student_t_target_p(q_all)

        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            batch_x = X_train[b_idx]
            b_len = float(batch_x.shape[0])

            z, x_hat, cache = ae.forward(batch_x)
            grads: Dict[str, np.ndarray] = {}

            # Thành phần 1: Sai số tái tạo E_rec
            grad_z_total = np.zeros_like(z)
            if method_cfg.w_rec > 0.0:
                diff_rec = x_hat - batch_x
                grad_x_hat = method_cfg.w_rec * (2.0 / (b_len * input_dim)) * diff_rec
                gz_rec, grads_dec = ae.backward_decoder(grad_x_hat, cache["dec"])
                grad_z_total += gz_rec
                for k, v in grads_dec.items():
                    grads[k] = grads.get(k, 0.0) + v

            if is_rbf:
                # Cập nhật EMA sigma & kiểm tra bão hòa nhân RBF cho Ada-DCSS trước khi tính đạo hàm
                if method_cfg.sigma_mode == "adaptive_ema_rms":
                    q_pre, _ = rbf_forward(z, centroids, sigma, temperature=method_cfg.temperature)
                    new_sigma = _update_ema_sigma_numpy(
                        sigma,
                        z,
                        centroids,
                        q_pre,
                        method_cfg.ema_alpha,
                        auto_rescue=method_cfg.auto_rescue_collapse,
                    )
                    if abs(float(np.mean(new_sigma)) - float(np.mean(sigma))) > 0.5 * float(np.mean(sigma)):
                        collapse_rescued += 1
                    sigma = new_sigma
                q, rbf_cache = rbf_forward(z, centroids, sigma, temperature=method_cfg.temperature)

                # Soft Silhouette + Entropy H2(Q)
                _, gz_sil, gq_sil = soft_silhouette_forward_backward(z, q)
                _, gq_ent = entropy_reg_forward_backward(q)

                grad_q_total = method_cfg.lambda_sil * gq_sil + method_cfg.beta_ent * gq_ent
                grad_z_total += method_cfg.lambda_sil * gz_sil

                # Nhánh Consistency cho Consist-DCSS (Đóng góp 2)
                if method_cfg.use_affine_branch and method_cfg.eta_cons > 0.0:
                    z_tilde = ae.forward_affine(z)
                    x_hat_tilde, dec_tilde_cache = ae.decode(z_tilde)
                    q_tilde, rbf_tilde_cache = rbf_forward(
                        z_tilde, centroids, sigma, temperature=method_cfg.temperature
                    )

                    # Consistency tái tạo: MSE(x_hat_tilde, x_hat)
                    diff_cons = x_hat_tilde - x_hat
                    g_xhat_tilde = method_cfg.eta_cons * (2.0 / (b_len * input_dim)) * diff_cons
                    gz_tilde, grads_dec_tilde = ae.backward_decoder(g_xhat_tilde, dec_tilde_cache)
                    for k, v in grads_dec_tilde.items():
                        grads[k] = grads.get(k, 0.0) + v

                    if method_cfg.cons_on_assignments:
                        # Nhất quán phân phối gán cụm mềm giữa Q(z) và Q(z_tilde)
                        K_val = float(exp_cfg.n_clusters)
                        diff_q = q - q_tilde
                        grad_q_total += method_cfg.eta_cons * (2.0 / (b_len * K_val)) * diff_q
                        gq_tilde = -method_cfg.eta_cons * (2.0 / (b_len * K_val)) * diff_q
                        gz_t_rbf, gc_t_rbf = rbf_backward(gq_tilde, rbf_tilde_cache)
                        gz_tilde += gz_t_rbf
                        grads["centroids"] = grads.get("centroids", 0.0) + gc_t_rbf

                    grads["W_aff"] = z.T @ gz_tilde
                    grads["b_aff"] = np.sum(gz_tilde, axis=0)
                    grad_z_total += gz_tilde @ ae.params["W_aff"].T

                gz_rbf, gc_rbf = rbf_backward(grad_q_total, rbf_cache)
                grad_z_total += gz_rbf
                grads["centroids"] = grads.get("centroids", 0.0) + gc_rbf

            elif method_cfg.family == "decra":
                # DECRA (Eq. 3.6 & 3.7): L = 2*E + 0.6*E_cons + (0.1/K)*K_n
                z_tilde = ae.forward_affine(z)
                x_hat_tilde, dec_tilde_cache = ae.decode(z_tilde)
                q, inv_dist = student_t_forward(z_tilde, centroids)
                batch_p = target_p_all[b_idx]

                # Đạo hàm E_cons = MSE(x_hat_tilde, x_hat)
                diff_cons = x_hat_tilde - x_hat
                g_xhat_tilde = method_cfg.eta_cons * (2.0 / (b_len * input_dim)) * diff_cons
                gz_tilde_cons, grads_dec_tilde = ae.backward_decoder(g_xhat_tilde, dec_tilde_cache)
                for k, v in grads_dec_tilde.items():
                    grads[k] = grads.get(k, 0.0) + v

                # Đạo hàm KL (gamma_kl / K) trên z_tilde
                kl_weight = method_cfg.gamma_kl / float(exp_cfg.n_clusters)
                _, gz_kl, gc_kl = student_t_kl_backward(z_tilde, centroids, q, batch_p, inv_dist)
                gz_tilde_total = gz_tilde_cons + kl_weight * gz_kl

                grads[" centroids" if False else "centroids"] = kl_weight * gc_kl
                grads["W_aff"] = z.T @ gz_tilde_total
                grads["b_aff"] = np.sum(gz_tilde_total, axis=0)
                grad_z_total += gz_tilde_total @ ae.params["W_aff"].T

            else:
                # DEC & IDEC & BRB-inspired
                q, inv_dist = student_t_forward(z, centroids)
                batch_p = target_p_all[b_idx]
                _, gz_kl, gc_kl = student_t_kl_backward(z, centroids, q, batch_p, inv_dist)
                grad_z_total += method_cfg.gamma_kl * gz_kl
                grads["centroids"] = method_cfg.gamma_kl * gc_kl

            grads_enc = ae.backward_encoder(grad_z_total, cache["enc"])
            for k, v in grads_enc.items():
                grads[k] = grads.get(k, 0.0) + v

            optimizer.step(grads)

        # Kiểm tra bằng Unsupervised Monitor ở đầu quá trình fine-tune (epoch 0, 1, 3)
        if is_rbf and method_cfg.auto_rescue_collapse and epoch in (0, 1, 3):
            z_curr, _ = ae.encode(X_train)
            q_curr, _ = rbf_forward(z_curr, centroids, sigma, temperature=method_cfg.temperature)
            unsup_curr = compute_unsupervised_metrics(z_curr, q_curr, seed=seed)
            if monitor.check_collapse(unsup_curr):
                dist_sq = np.sum((z_curr[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
                rms_val = float(np.sqrt(max(np.mean(dist_sq), 1e-6)))
                sigma.fill(rms_val)
                collapse_rescued += 1

    finetune_time_sec = time.perf_counter() - t_start

    # Đánh giá cuối cùng trên tập Train (Transductive)
    z_final, _ = ae.encode(X_train)
    if is_rbf:
        q_final, _ = rbf_forward(z_final, centroids, sigma, temperature=method_cfg.temperature)
    elif method_cfg.family == "decra":
        q_final, _ = student_t_forward(ae.forward_affine(z_final), centroids)
    else:
        q_final, _ = student_t_forward(z_final, centroids)

    final_pred = np.argmax(q_final, axis=1).astype(np.int64)
    unsup_final = compute_unsupervised_metrics(z_final, q_final, seed=seed)
    sup_final = compute_all_supervised_metrics(y_train, final_pred)

    # Đánh giá trên tập Unseen (Inductive) nếu có
    ind_sup = {"ACC": float("nan"), "NMI": float("nan"), "ARI": float("nan")}
    if X_unseen is not None and len(X_unseen) > 0:
        z_unseen, _ = ae.encode(X_unseen)
        if is_rbf:
            q_unseen, _ = rbf_forward(z_unseen, centroids, sigma, temperature=method_cfg.temperature)
        elif method_cfg.family == "decra":
            q_unseen, _ = student_t_forward(ae.forward_affine(z_unseen), centroids)
        else:
            q_unseen, _ = student_t_forward(z_unseen, centroids)
        unseen_pred = np.argmax(q_unseen, axis=1).astype(np.int64)
        ind_sup = compute_all_supervised_metrics(y_unseen, unseen_pred)

    return RunResult(
        method_name=method_cfg.name,
        family=method_cfg.family,
        seed=seed,
        acc=sup_final["ACC"],
        nmi=sup_final["NMI"],
        ari=sup_final["ARI"],
        unsup_silhouette=unsup_final["unsup_silhouette"],
        active_clusters=unsup_final["active_clusters"],
        assignment_entropy=unsup_final["assignment_entropy"],
        finetune_time_sec=finetune_time_sec,
        pretrain_time_sec=pretrain_state.pretrain_time_sec,
        final_sigma_mean=float(np.mean(sigma)) if is_rbf else 0.0,
        final_sigma_std=float(np.std(sigma)) if is_rbf else 0.0,
        collapse_rescued=collapse_rescued,
        predictions=final_pred,
        inductive_acc=ind_sup["ACC"],
        inductive_nmi=ind_sup["NMI"],
        inductive_ari=ind_sup["ARI"],
    )
