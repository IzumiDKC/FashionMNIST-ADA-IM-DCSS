"""
Tiền huấn luyện Shared Autoencoder và khởi tạo tâm cụm K-Means theo từng seed.
Bám sát Mục 5.5 (Trang 16) của bài báo gốc DCSS (Vardakas et al., Machine Learning 2026):
- Tiền huấn luyện AE bằng Adam (beta_1=0.9, beta_2=0.999), lr = 1e-3, L2 regularization = 1e-5, batch_size = 256.
- Chạy K-Means với n_init = 100 lần khởi tạo trên không gian nhúng z = f_w(X), giữ nghiệm có tổng bình phương sai số nhỏ nhất.
- Mọi thuật toán trong cùng một seed đều xuất phát từ cùng một checkpoint Autoencoder và cùng một bộ tâm K-Means.
"""

import copy
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig
from ..losses.dec_idec_loss import reconstruction_mse_loss
from ..models.autoencoder import SharedAutoencoder


@dataclass
class PretrainState:
    """Lưu trạng thái sau bước Pretrain AE + K-Means của một seed để dùng chung cho mọi phương pháp."""

    seed: int
    ae_state_dict: Dict[str, torch.Tensor]
    initial_centroids: torch.Tensor
    kmeans_pred: np.ndarray
    kmeans_unseen_pred: np.ndarray
    pretrain_time_sec: float
    initial_rec_loss: float


def set_deterministic_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_kmeans_numpy(
    z: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int = 100,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy K-Means khởi tạo tâm cụm với n_init=100 theo đúng Mục 5.5 (Trang 16) bài báo DCSS.
    Trả về (centroids [K, D], labels [N]).
    """
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=seed)
        labels = km.fit_predict(z)
        return km.cluster_centers_.astype(np.float32), labels.astype(np.int64)
    except ImportError:
        rng = np.random.default_rng(seed)
        n_samples, dim = z.shape
        best_inertia = float("inf")
        best_centers = np.zeros((n_clusters, dim), dtype=np.float32)
        best_labels = np.zeros(n_samples, dtype=np.int64)

        for _ in range(n_init):
            centers = np.empty((n_clusters, dim), dtype=np.float32)
            first_idx = rng.integers(0, n_samples)
            centers[0] = z[first_idx]
            dist_sq = np.sum((z - centers[0]) ** 2, axis=1)
            for k in range(1, n_clusters):
                probs = dist_sq / max(dist_sq.sum(), 1e-12)
                next_idx = rng.choice(n_samples, p=probs)
                centers[k] = z[next_idx]
                dist_sq = np.minimum(dist_sq, np.sum((z - centers[k]) ** 2, axis=1))

            for _ in range(max_iter):
                d_mat = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
                labels = np.argmin(d_mat, axis=1)
                new_centers = np.copy(centers)
                for k in range(n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        new_centers[k] = z[mask].mean(axis=0)
                if np.allclose(centers, new_centers, atol=1e-4):
                    centers = new_centers
                    break
                centers = new_centers

            d_mat = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
            labels = np.argmin(d_mat, axis=1)
            inertia = float(np.sum(np.min(d_mat, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        return best_centers, best_labels


def pretrain_shared_autoencoder(
    X_train: torch.Tensor,
    exp_cfg: ExperimentConfig,
    seed: int,
    device: torch.device,
    X_unseen: torch.Tensor = None,
) -> PretrainState:
    """
    Tiền huấn luyện Shared Autoencoder (ConvAutoencoder28x28 cho ảnh 28x28)
    với lr = 1e-3, weight_decay = 1e-5 và khởi tạo K-Means (n_init = 100) theo Mục 5.5 bài báo DCSS.
    """
    set_deterministic_seed(seed)
    t0 = time.perf_counter()

    model = SharedAutoencoder(
        input_dim=exp_cfg.input_dim,
        hidden_dims=exp_cfg.hidden_dims,
        latent_dim=exp_cfg.latent_dim,
    ).to(device)

    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=exp_cfg.batch_size, shuffle=True, drop_last=False)
    weight_decay = getattr(exp_cfg, "pretrain_weight_decay", 1e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_cfg.pretrain_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    model.train()
    last_loss = 0.0
    for _ in range(exp_cfg.pretrain_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            _, x_hat = model(batch_x)
            loss = reconstruction_mse_loss(batch_x, x_hat)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        last_loss = epoch_loss / max(n_batches, 1)

    model.eval()
    z_chunks = []
    eval_loader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)
    with torch.no_grad():
        for (bx,) in eval_loader:
            z_chunks.append(model.encode(bx.to(device)).cpu().numpy())
    z_all = np.concatenate(z_chunks, axis=0)

    centroids_np, kmeans_pred = _run_kmeans_numpy(z_all, exp_cfg.n_clusters, seed=seed, n_init=100)

    kmeans_unseen_pred = np.array([], dtype=np.int64)
    if X_unseen is not None and len(X_unseen) > 0:
        with torch.no_grad():
            z_unseen = model.encode(X_unseen.to(device)).cpu().numpy()
        d_unseen = np.sum((z_unseen[:, None, :] - centroids_np[None, :, :]) ** 2, axis=-1)
        kmeans_unseen_pred = np.argmin(d_unseen, axis=1).astype(np.int64)

    elapsed = time.perf_counter() - t0
    return PretrainState(
        seed=seed,
        ae_state_dict=copy.deepcopy(model.state_dict()),
        initial_centroids=torch.from_numpy(centroids_np).float(),
        kmeans_pred=kmeans_pred,
        kmeans_unseen_pred=kmeans_unseen_pred,
        pretrain_time_sec=elapsed,
        initial_rec_loss=last_loss,
    )
