"""
Module nạp dữ liệu (Data Adapter) với kiến trúc cổng chờ (Plug-and-Play).
Hỗ trợ:
1. Định dạng IDX nén .gz chuẩn của Fashion-MNIST / MNIST (t10k-images-idx3-ubyte.gz, ...)
2. Định dạng .npz / .npy / .csv do người dùng cung cấp sau
3. Tập Digits (8x8) qua scikit-learn (nếu có)
4. Bộ sinh dữ liệu giả lập (Synthetic Mixture in Latent Space) để kiểm thử thông luồng.
"""

import gzip
import os
import struct
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@dataclass
class DatasetBundle:
    """Gói dữ liệu chuẩn hóa trả về cho pipeline thực nghiệm (hỗ trợ cả NumPy & PyTorch)."""

    name: str
    X_train: Any  # np.ndarray hoặc torch.Tensor
    y_train: Optional[np.ndarray]  # Chỉ dùng cho đánh giá hậu kiểm (post-hoc metrics)
    input_dim: int
    n_clusters: int
    X_unseen: Optional[Any] = None
    y_unseen: Optional[np.ndarray] = None
    X_train_np: Optional[np.ndarray] = None
    X_unseen_np: Optional[np.ndarray] = None


def _normalize_to_unit_range(x: np.ndarray) -> np.ndarray:
    """Đưa dữ liệu về kiểu float32 và chuẩn hóa thang đo về [0, 1] theo Mục 4.1."""
    x = x.astype(np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > 1.0 and x_min >= 0.0 and x_max <= 255.0:
        return x / 255.0
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    return x


def load_idx_gz(images_path: str, labels_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Đọc file ảnh và nhãn định dạng IDX (.gz hoặc không nén) của Fashion-MNIST / MNIST."""
    opener_img = gzip.open if images_path.endswith(".gz") else open
    with opener_img(images_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number ảnh IDX không hợp lệ: {magic} (kỳ vọng 2051)")
        buf = f.read(num_images * rows * cols)
        images = np.frombuffer(buf, dtype=np.uint8).reshape(num_images, rows * cols)

    labels = None
    if labels_path and os.path.exists(labels_path):
        opener_lbl = gzip.open if labels_path.endswith(".gz") else open
        with opener_lbl(labels_path, "rb") as f:
            magic_l, num_labels = struct.unpack(">II", f.read(8))
            if magic_l != 2049:
                raise ValueError(f"Magic number nhãn IDX không hợp lệ: {magic_l} (kỳ vọng 2049)")
            labels = np.frombuffer(f.read(num_labels), dtype=np.uint8).astype(np.int64)

    return _normalize_to_unit_range(images), labels


def load_numpy_or_csv(data_path: str, labels_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Đọc dữ liệu từ file .npz, .npy hoặc .csv bất kỳ."""
    ext = os.path.splitext(data_path)[1].lower()
    labels = None

    if ext == ".npz":
        archive = np.load(data_path)
        x_key = next((k for k in ["X", "x", "data", "images", "features"] if k in archive), archive.files[0])
        X = archive[x_key]
        y_key = next((k for k in ["y", "Y", "labels", "targets"] if k in archive), None)
        if y_key is not None:
            labels = archive[y_key].astype(np.int64)
    elif ext == ".npy":
        X = np.load(data_path)
    elif ext in (".csv", ".txt"):
        raw = np.loadtxt(data_path, delimiter="," if ext == ".csv" else None)
        if labels_path is None and raw.shape[1] > 2:
            # Nếu cột cuối là nhãn nguyên và không truyền labels_path riêng
            last_col = raw[:, -1]
            if np.all(np.equal(np.mod(last_col, 1), 0)) and len(np.unique(last_col)) < 100:
                X = raw[:, :-1]
                labels = last_col.astype(np.int64)
            else:
                X = raw
        else:
            X = raw
    else:
        raise ValueError(f"Định dạng file dữ liệu chưa được hỗ trợ: {ext}")

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if labels is None and labels_path and os.path.exists(labels_path):
        l_ext = os.path.splitext(labels_path)[1].lower()
        if l_ext == ".npy":
            labels = np.load(labels_path).astype(np.int64)
        else:
            labels = np.loadtxt(labels_path, delimiter="," if l_ext == ".csv" else None).astype(np.int64)

    return _normalize_to_unit_range(X), labels


def load_synthetic_smoke_test(
    n_samples: int = 1200,
    input_dim: int = 784,
    latent_dim: int = 10,
    n_clusters: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinh dữ liệu giả lập có cấu trúc cụm phi tuyến trong không gian tiềm ẩn rồi chiếu lên
    không gian input_dim chiều (ví dụ 784D) để kiểm thử toàn bộ luồng AE + Deep Clustering
    trước khi người dùng cung cấp Dataset thật.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_clusters, size=n_samples)
    centers = rng.normal(loc=0.0, scale=2.8, size=(n_clusters, latent_dim))

    # Tạo độ lệch bán kính cụm khác nhau để kiểm tra tính năng Per-Cluster Adaptive Sigma
    cluster_scales = np.linspace(0.45, 1.15, n_clusters)
    z_true = np.zeros((n_samples, latent_dim), dtype=np.float32)
    for k in range(n_clusters):
        mask = labels == k
        z_true[mask] = centers[k] + rng.normal(scale=cluster_scales[k], size=(mask.sum(), latent_dim))

    # Chiếu phi tuyến lên không gian quan sát D chiều và chuẩn hóa về [0, 1]
    proj_w1 = rng.normal(scale=1.0 / np.sqrt(latent_dim), size=(latent_dim, 128)).astype(np.float32)
    proj_w2 = rng.normal(scale=1.0 / np.sqrt(128), size=(128, input_dim)).astype(np.float32)
    hidden = np.tanh(z_true @ proj_w1)
    x_raw = 1.0 / (1.0 + np.exp(-(hidden @ proj_w2 + rng.normal(scale=0.15, size=(n_samples, input_dim)))))
    return _normalize_to_unit_range(x_raw), labels.astype(np.int64)


def load_dataset(
    dataset_name: str = "synthetic",
    data_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    n_clusters_default: int = 10,
    input_dim_default: int = 784,
    inductive_test_ratio: float = 0.0,
    seed: int = 42,
) -> DatasetBundle:
    """
    Hàm điều phối nạp dữ liệu thống nhất.
    Khi người dùng cung cấp dataset thật, chỉ cần truyền `data_path` (và `labels_path` nếu tách rời).
    """
    name_lower = dataset_name.lower()

    if data_path and os.path.exists(data_path):
        if data_path.endswith("ubyte.gz") or data_path.endswith("ubyte"):
            X, y = load_idx_gz(data_path, labels_path)
        else:
            X, y = load_numpy_or_csv(data_path, labels_path)
    elif name_lower == "digits":
        from sklearn.datasets import load_digits

        digits = load_digits()
        X = _normalize_to_unit_range(digits.data)
        y = digits.target.astype(np.int64)
    elif name_lower in ("fashion-mnist", "fashion_mnist", "fmnist"):
        # Tự động dò tìm file chuẩn của Fashion-MNIST trong thư mục hiện tại hoặc data/
        candidates = [
            ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
            ("data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"),
            ("data/fashion/t10k-images-idx3-ubyte.gz", "data/fashion/t10k-labels-idx1-ubyte.gz"),
        ]
        found_img, found_lbl = None, None
        for img_p, lbl_p in candidates:
            if os.path.exists(img_p):
                found_img, found_lbl = img_p, lbl_p
                break
        if found_img is None:
            raise FileNotFoundError(
                "Chưa tìm thấy file t10k-images-idx3-ubyte.gz. Vui lòng cung cấp dataset hoặc truyền --data-path."
            )
        X, y = load_idx_gz(found_img, found_lbl)
    else:
        X, y = load_synthetic_smoke_test(
            n_samples=1200,
            input_dim=input_dim_default,
            n_clusters=n_clusters_default,
            seed=seed,
        )

    n_clusters = int(len(np.unique(y))) if y is not None else n_clusters_default
    input_dim = int(X.shape[1])

    X_unseen_np = None
    X_unseen_tensor = None
    y_unseen_arr = None
    if 0.0 < inductive_test_ratio < 0.5:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        n_unseen = int(len(X) * inductive_test_ratio)
        unseen_idx, train_idx = perm[:n_unseen], perm[n_unseen:]
        X_unseen_np = X[unseen_idx].astype(np.float32)
        X_unseen_tensor = torch.from_numpy(X_unseen_np).float() if HAS_TORCH else X_unseen_np
        X = X[train_idx].astype(np.float32)
        if y is not None:
            y_unseen_arr = y[unseen_idx]
            y = y[train_idx]

    X_np = X.astype(np.float32)
    X_tensor = torch.from_numpy(X_np).float() if HAS_TORCH else X_np

    return DatasetBundle(
        name=dataset_name,
        X_train=X_tensor,
        y_train=y,
        input_dim=input_dim,
        n_clusters=n_clusters,
        X_unseen=X_unseen_tensor,
        y_unseen=y_unseen_arr,
        X_train_np=X_np,
        X_unseen_np=X_unseen_np,
    )
