"""
Vòng lặp huấn luyện phân cụm sâu (Fine-tuning Trainer) bám sát Mục 5.5 (Trang 16) bài báo DCSS (2026)
cho toàn bộ 10 cấu hình đối chứng:
- AE + K-means
- DEC (2016)
- IDEC (2017)
- DECRA (2024)
- IDEC + BRB-inspired
- DCSS (sigma=1) & DCSS (RMS)
- Ada-DCSS (Ours - EMA Per-Cluster Sigma), IM-DCSS (Ours - InfoMax & Entropy Warmup), Ada-IM-DCSS (Ours - Full Proposed)
"""

import copy
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ExperimentConfig, MethodConfig
from ..evaluation.metrics import compute_all_supervised_metrics, compute_unsupervised_metrics
from ..losses.dcss_loss import DCSSLoss
from ..losses.dec_idec_loss import DECIDECLoss, reconstruction_mse_loss
from ..losses.decra_loss import DECRALoss
from ..models.autoencoder import SharedAutoencoder
from ..models.cluster_heads import RBFClusterHead, StudentTClusterHead
from .pretrainer import PretrainState, set_deterministic_seed
from .unsup_monitor import UnsupervisedCollapseMonitor


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
    embeddings: Optional[np.ndarray] = None
    inductive_acc: float = float("nan")
    inductive_nmi: float = float("nan")
    inductive_ari: float = float("nan")


@torch.no_grad()
def _encode_batched(ae: SharedAutoencoder, X_eval: torch.Tensor, device: torch.device, batch_size: int = 512) -> torch.Tensor:
    ae.eval()
    chunks = []
    for start in range(0, len(X_eval), batch_size):
        bx = X_eval[start : start + batch_size].to(device)
        chunks.append(ae.encode(bx))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _evaluate_current_state(
    ae: SharedAutoencoder,
    cluster_head: torch.nn.Module,
    X_eval: torch.Tensor,
    device: torch.device,
    is_rbf: bool,
    use_residual_for_q: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Trích xuất (z, q, y_pred, rec_loss) theo từng lô để tiết kiệm bộ nhớ RAM/CPU."""
    ae.eval()
    cluster_head.eval()
    z_list, q_list = [], []
    rec_sum = 0.0
    n_total = len(X_eval)

    for start in range(0, n_total, 512):
        bx = X_eval[start : start + 512].to(device)
        z, x_hat = ae(bx)
        rec_sum += float(reconstruction_mse_loss(bx, x_hat).item()) * len(bx)
        if is_rbf:
            q_main = cluster_head(z, update_sigma=False)
            if use_residual_for_q:
                z_tilde = ae.forward_affine(z)
                q_res = cluster_head(z_tilde, update_sigma=False)
                q = 0.5 * (q_main + q_res)
                z_q = 0.5 * (z + z_tilde)
            else:
                q = q_main
                z_q = z
        else:
            z_q = ae.forward_affine(z) if use_residual_for_q else z
            q = cluster_head(z_q)
        z_list.append(z_q.cpu().numpy())
        q_list.append(q.cpu().numpy())

    z_np = np.concatenate(z_list, axis=0)
    q_np = np.concatenate(q_list, axis=0)
    y_pred = np.argmax(q_np, axis=1).astype(np.int64)
    rec_loss = rec_sum / max(n_total, 1)
    return z_np, q_np, y_pred, rec_loss


def train_clustering_method(
    method_cfg: MethodConfig,
    exp_cfg: ExperimentConfig,
    pretrain_state: PretrainState,
    X_train: torch.Tensor,
    y_train: Optional[np.ndarray],
    device: torch.device,
    X_unseen: Optional[torch.Tensor] = None,
    y_unseen: Optional[np.ndarray] = None,
) -> RunResult:
    """
    Huấn luyện 1 cấu hình phân cụm xuất phát từ đúng `pretrain_state` của `seed` tương ứng.
    Lưu ý: `y_train` và `y_unseen` TUYỆT ĐỐI chỉ được dùng ở bước tính chỉ số hậu kiểm cuối cùng.
    """
    seed = pretrain_state.seed
    set_deterministic_seed(seed)

    ae = SharedAutoencoder(
        input_dim=exp_cfg.input_dim,
        hidden_dims=exp_cfg.hidden_dims,
        latent_dim=exp_cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(copy.deepcopy(pretrain_state.ae_state_dict))
    ae.reset_affine_identity()

    # Trường hợp 1: Đối chứng AE + K-means
    if method_cfg.family == "kmeans":
        z_np = _encode_batched(ae, X_train, device).cpu().numpy()
        q_km = np.eye(exp_cfg.n_clusters, dtype=np.float32)[pretrain_state.kmeans_pred]
        unsup = compute_unsupervised_metrics(z_np, q_km, seed=seed)
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
            embeddings=z_np,
            inductive_acc=ind_sup["ACC"],
            inductive_nmi=ind_sup["NMI"],
            inductive_ari=ind_sup["ARI"],
        )

    # Trường hợp 2: Các mô hình tối ưu phân cụm sâu (DEC, IDEC, DECRA, BRB, DCSS & Ours)
    is_rbf = method_cfg.family == "dcss"
    if is_rbf:
        cluster_head = RBFClusterHead(
            n_clusters=exp_cfg.n_clusters,
            latent_dim=exp_cfg.latent_dim,
            temperature=method_cfg.temperature,
            sigma_mode=method_cfg.sigma_mode,
            sigma_init=method_cfg.sigma_init,
            ema_alpha=method_cfg.ema_alpha,
            normalize_latent_scale=method_cfg.normalize_latent_scale,
        ).to(device)
        cluster_head.set_centroids(pretrain_state.initial_centroids)
        z_init = _encode_batched(ae, X_train, device)
        cluster_head.initialize_sigma_from_embeddings(z_init)
        loss_fn = DCSSLoss(
            w_rec=method_cfg.w_rec,
            lambda_sil=method_cfg.lambda_sil,
            beta_ent=method_cfg.beta_ent,
            use_infomax=getattr(method_cfg, "use_infomax", False),
            lambda_bal=getattr(method_cfg, "lambda_bal", 0.004),
            entropy_warmup_ratio=getattr(method_cfg, "entropy_warmup_ratio", 0.70),
            eta_cons=method_cfg.eta_cons if method_cfg.use_affine_branch else 0.0,
            cons_on_assignments=method_cfg.cons_on_assignments,
            balance_weight=getattr(method_cfg, "balance_weight", 0.0),
        )
    else:
        cluster_head = StudentTClusterHead(
            n_clusters=exp_cfg.n_clusters,
            latent_dim=exp_cfg.latent_dim,
        ).to(device)
        cluster_head.set_centroids(pretrain_state.initial_centroids)
        if method_cfg.family == "decra":
            loss_fn = DECRALoss(
                n_clusters=exp_cfg.n_clusters,
                w_rec=method_cfg.w_rec,
                eta_cons=method_cfg.eta_cons,
                gamma_kl=method_cfg.gamma_kl,
            )
        else:
            loss_fn = DECIDECLoss(
                w_rec=method_cfg.w_rec,
                gamma_kl=method_cfg.gamma_kl,
            )

    params = list(ae.parameters()) + list(cluster_head.parameters())
    # Theo Mục 5.5 (Trang 16) bài báo DCSS: Adam(0.9, 0.999), lr = 5e-4, không dùng L2 penalty ở pha Clustering
    optimizer = torch.optim.Adam(params, lr=exp_cfg.finetune_lr, betas=(0.9, 0.999), weight_decay=0.0)

    n_samples = len(X_train)
    batch_size = exp_cfg.batch_size

    monitor = UnsupervisedCollapseMonitor(n_clusters=exp_cfg.n_clusters)
    collapse_rescued_count = 0

    t_start = time.perf_counter()

    for epoch in range(exp_cfg.finetune_epochs):
        progress = float(epoch) / float(max(exp_cfg.finetune_epochs - 1, 1))
        if (
            method_cfg.family == "brb"
            and epoch > 0
            and epoch % max(method_cfg.brb_reset_interval, 1) == 0
        ):
            z_full = _encode_batched(ae, X_train, device)
            cluster_head.brb_reset_centroids(z_full, perturb_scale=method_cfg.brb_perturb_scale)

        target_p_all = None
        if not is_rbf:
            ae.eval()
            cluster_head.eval()
            with torch.no_grad():
                z_all = _encode_batched(ae, X_train, device)
                z_for_q = ae.forward_affine(z_all) if (method_cfg.family == "decra" and method_cfg.use_affine_branch) else z_all
                q_all = cluster_head(z_for_q)
                target_p_all = StudentTClusterHead.target_distribution(q_all)

        ae.train()
        cluster_head.train()

        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_x = X_train[batch_idx].to(device)
            optimizer.zero_grad()

            z, x_hat = ae(batch_x)

            if is_rbf:
                q = cluster_head(z, update_sigma=True)
                x_hat_tilde = None
                q_tilde = None
                if method_cfg.use_affine_branch:
                    z_tilde = ae.forward_affine(z)
                    x_hat_tilde = ae.decode(z_tilde)
                    q_tilde = cluster_head(z_tilde, update_sigma=False)
                loss_dict = loss_fn(
                    x=batch_x,
                    x_hat=x_hat,
                    z=z,
                    q=q,
                    progress=progress,
                    x_hat_tilde=x_hat_tilde,
                    q_tilde=q_tilde,
                )
            elif method_cfg.family == "decra":
                z_tilde = ae.forward_affine(z)
                x_hat_tilde = ae.decode(z_tilde)
                q = cluster_head(z_tilde)
                batch_p = target_p_all[batch_idx.to(device)]
                loss_dict = loss_fn(
                    x=batch_x,
                    x_hat=x_hat,
                    x_hat_tilde=x_hat_tilde,
                    q=q,
                    p=batch_p,
                )
            else:
                q = cluster_head(z)
                batch_p = target_p_all[batch_idx.to(device)]
                loss_dict = loss_fn(
                    x=batch_x,
                    x_hat=x_hat,
                    q=q,
                    p=batch_p,
                )

            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

        if is_rbf and method_cfg.auto_rescue_collapse and epoch in (0, 2):
            z_np_c, q_np_c, _, _ = _evaluate_current_state(ae, cluster_head, X_train, device, is_rbf=True)
            unsup_curr = compute_unsupervised_metrics(z_np_c, q_np_c, seed=seed)
            if monitor.check_collapse(unsup_curr):
                z_rescue = _encode_batched(ae, X_train, device)
                cluster_head.rescue_sigma_with_rms(z_rescue)
                collapse_rescued_count += 1

    finetune_time_sec = time.perf_counter() - t_start

    use_res_q = method_cfg.use_affine_branch and method_cfg.family in ("decra", "dcss")
    z_np, q_np, final_pred, _ = _evaluate_current_state(
        ae, cluster_head, X_train, device, is_rbf=is_rbf, use_residual_for_q=use_res_q
    )
    unsup_final = compute_unsupervised_metrics(z_np, q_np, seed=seed)
    sup_final = compute_all_supervised_metrics(y_train, final_pred)

    ind_sup = {"ACC": float("nan"), "NMI": float("nan"), "ARI": float("nan")}
    if X_unseen is not None and len(X_unseen) > 0:
        _, _, unseen_pred, _ = _evaluate_current_state(
            ae, cluster_head, X_unseen, device, is_rbf=is_rbf, use_residual_for_q=use_res_q
        )
        ind_sup = compute_all_supervised_metrics(y_unseen, unseen_pred)

    if is_rbf:
        sigmas_np = cluster_head.sigma.detach().cpu().numpy()
        sigma_mean = float(np.mean(sigmas_np))
        sigma_std = float(np.std(sigmas_np))
    else:
        sigma_mean = 0.0
        sigma_std = 0.0

    del ae, cluster_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        final_sigma_mean=sigma_mean,
        final_sigma_std=sigma_std,
        collapse_rescued=collapse_rescued_count,
        predictions=final_pred,
        embeddings=z_np,
        inductive_acc=ind_sup["ACC"],
        inductive_nmi=ind_sup["NMI"],
        inductive_ari=ind_sup["ARI"],
    )
