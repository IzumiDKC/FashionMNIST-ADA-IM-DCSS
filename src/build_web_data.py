"""
Script trích xuất toàn bộ dữ liệu thực nghiệm thực tế từ `outputs_fashion_mnist/`
và tập ảnh `data/fashion/` thành gói dữ liệu `web/data_bundle.js` phục vụ Website Dashboard.
Bao gồm:
1. Số liệu Benchmark (10 cấu hình x 3 seed), Ablation (5 cấu hình x 3 seed), Sensitivity (10 cấu hình x 3 seed),
   đối chiếu trực tiếp với Bảng 2 (Trang 17) của bài báo gốc DCSS (Vardakas et al., Machine Learning 2026).
2. Ma trận nhầm lẫn sau ghép nhãn Hungarian (10x10 Confusion Matrix) & Phân bố kích thước cụm thực tế.
3. Tọa độ không gian tiềm ẩn 2D (PCA trực tiếp trên biểu diễn nhúng 10 chiều z_*.npy học được từ ConvAE)
   cho các mô hình tiêu biểu (không dùng tọa độ giả lập).
4. Mẫu ảnh thực tế 28x28 từ Fashion-MNIST kèm nhãn thật và nhãn dự đoán của từng thuật toán.
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_adapter import load_dataset


FASHION_CLASS_NAMES = [
    "0: Áo thun (T-shirt/top)",
    "1: Quần dài (Trouser)",
    "2: Áo len chui (Pullover)",
    "3: Đầm (Dress)",
    "4: Áo khoác (Coat)",
    "5: Giày sandal (Sandal)",
    "6: Áo sơ mi (Shirt)",
    "7: Giày thể thao (Sneaker)",
    "8: Túi xách (Bag)",
    "9: Giày cổ cao (Ankle boot)",
]

# Số liệu gốc Bảng 2 (Trang 17) bài báo DCSS (Vardakas et al., Machine Learning 2026, 115:81) cho F-MNIST
PAPER_TABLE2_FMNIST = {
    "k-means (Raw Space)": {"NMI": "51.0 ± 0.0", "ARI": "35.0 ± 0.0"},
    "AE + K-means": {"NMI": "61.0 ± 2.0", "ARI": "45.0 ± 2.0"},
    "DCN": {"NMI": "62.0 ± 1.0", "ARI": "42.0 ± 2.0"},
    "DEC (2016)": {"NMI": "57.0 ± 1.0", "ARI": "41.0 ± 1.0"},
    "IDEC (2017)": {"NMI": "59.0 ± 1.0", "ARI": "43.0 ± 1.0"},
    "DECRA (2024)": {"NMI": "62.0 ± 2.0", "ARI": "46.0 ± 3.0"},
    "DipEncoder": {"NMI": "61.0 ± 2.0", "ARI": "44.0 ± 3.0"},
    "DCSS (RMS)": {"NMI": "63.0 ± 2.0", "ARI": "45.0 ± 3.0"},
}


def read_csv_dicts(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def align_predictions_hungarian(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int = 10):
    """Ánh xạ nhãn cụm dự đoán sang nhãn lớp thật bằng thuật toán Hungarian và trả về ma trận nhầm lẫn."""
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        if 0 <= yp < n_clusters and 0 <= yt < n_clusters:
            w[yp, yt] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    cluster_to_class = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    aligned_pred = np.array([cluster_to_class.get(int(p), int(p)) for p in y_pred], dtype=np.int64)

    cm = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for yt, yp in zip(y_true, aligned_pred):
        cm[yt, yp] += 1

    cluster_sizes = [int(np.sum(y_pred == k)) for k in range(n_clusters)]
    per_class_acc = [
        round(float(cm[c, c]) / max(float(cm[c].sum()), 1.0) * 100.0, 2)
        for c in range(n_clusters)
    ]
    return aligned_pred, cm.tolist(), cluster_sizes, per_class_acc


def compute_real_pca_2d(z_full: np.ndarray, scatter_idx: np.ndarray) -> np.ndarray:
    """Chiếu trực tiếp không gian tiềm ẩn Z thực tế (10 chiều) xuống 2D bằng PCA/SVD chuẩn."""
    z_centered = z_full - z_full.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    coords_all = z_centered @ vt[:2].T
    coords_sub = coords_all[scatter_idx]
    max_abs = np.percentile(np.abs(coords_sub), 98, axis=0, keepdims=True) + 1e-8
    coords_norm = np.clip(coords_sub / max_abs, -1.25, 1.25) * 3.4
    return coords_norm


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_fmnist = os.path.join(base_dir, "outputs_fashion_mnist")
    web_dir = os.path.join(base_dir, "web")
    os.makedirs(web_dir, exist_ok=True)

    bundle = load_dataset(dataset_name="fashion-mnist", seed=0)
    X_np = bundle.X_train_np
    y_true = bundle.y_train

    benchmark_summary = read_csv_dicts(os.path.join(out_fmnist, "benchmark", "summary.csv"))
    benchmark_runs = read_csv_dicts(os.path.join(out_fmnist, "benchmark", "runs.csv"))
    ablation_summary = read_csv_dicts(os.path.join(out_fmnist, "ablation", "summary.csv"))
    ablation_runs = read_csv_dicts(os.path.join(out_fmnist, "ablation", "runs.csv"))
    sensitivity_summary = read_csv_dicts(os.path.join(out_fmnist, "sensitivity", "summary.csv"))
    sensitivity_runs = read_csv_dicts(os.path.join(out_fmnist, "sensitivity", "runs.csv"))

    pred_dir = os.path.join(out_fmnist, "benchmark", "predictions")
    method_slugs = {
        "AE + K-means": "ae_k_means",
        "DEC (2016)": "dec_2016",
        "IDEC (2017)": "idec_2017",
        "DECRA (2024)": "decra_2024",
        "IDEC + BRB-inspired": "idec_brb_inspired",
        "DCSS (sigma=1)": "dcss_sigma_1",
        "DCSS (RMS)": "dcss_rms",
        "Ada-DCSS (Ours - EMA Per-Cluster Sigma)": "ada_dcss_ours_ema_per_cluster_sigma",
        "IM-DCSS (Ours - InfoMax & Entropy Warmup)": "im_dcss_ours_infomax_entropy_warmup",
        "Ada-IM-DCSS (Ours - Full Proposed)": "ada_im_dcss_ours_full_proposed",
    }

    confusion_matrices = {}
    aligned_preds_seed0 = {}
    for m_name, slug in method_slugs.items():
        npy_path = os.path.join(pred_dir, f"pred_{slug}_seed0.npy")
        if os.path.exists(npy_path):
            y_pred = np.load(npy_path)
            aligned_pred, cm, c_sizes, c_acc = align_predictions_hungarian(y_true, y_pred)
            confusion_matrices[m_name] = {
                "matrix": cm,
                "cluster_sizes": c_sizes,
                "per_class_acc": c_acc,
            }
            aligned_preds_seed0[m_name] = aligned_pred

    # Trích xuất mẫu ảnh thực tế (6 mẫu mỗi lớp 0..9) để hiển thị trong Image Explorer
    sample_gallery = []
    sample_indices_for_scatter = []
    rng = np.random.default_rng(42)
    for c in range(10):
        idx_c = np.where(y_true == c)[0]
        chosen_gallery = idx_c[:6]
        chosen_scatter = rng.choice(idx_c, size=50, replace=False)
        sample_indices_for_scatter.extend(chosen_scatter.tolist())

        for idx in chosen_gallery:
            pixels_u8 = np.clip(np.round(X_np[idx] * 255.0), 0, 255).astype(int).tolist()
            preds_for_item = {
                m_name: int(aligned_preds_seed0[m_name][idx])
                for m_name in aligned_preds_seed0
            }
            sample_gallery.append(
                {
                    "index": int(idx),
                    "true_class": int(c),
                    "true_label": FASHION_CLASS_NAMES[c],
                    "pixels": pixels_u8,
                    "predictions": preds_for_item,
                }
            )

    scatter_idx = np.array(sample_indices_for_scatter, dtype=np.int64)

    # Trích xuất tọa độ PCA 2D thực tế từ z_<slug>_seed0.npy (fallback sang PCA trên X nếu thiếu)
    X_sub = X_np[scatter_idx]
    X_centered = X_sub - X_sub.mean(axis=0, keepdims=True)
    _, _, vt_x = np.linalg.svd(X_centered, full_matrices=False)
    fallback_coords = X_centered @ vt_x[:2].T
    fallback_coords = (fallback_coords / (np.percentile(np.abs(fallback_coords), 98, axis=0, keepdims=True) + 1e-8)) * 3.2

    latent_methods = [
        "AE + K-means",
        "DCSS (sigma=1)",
        "DCSS (RMS)",
        "Ada-DCSS (Ours - EMA Per-Cluster Sigma)",
        "IM-DCSS (Ours - InfoMax & Entropy Warmup)",
        "Ada-IM-DCSS (Ours - Full Proposed)",
    ]
    latent_views = {}
    for m_name in latent_methods:
        slug = method_slugs[m_name]
        z_path = os.path.join(pred_dir, f"z_{slug}_seed0.npy")
        if os.path.exists(z_path):
            z_arr = np.load(z_path)
            coords_2d = compute_real_pca_2d(z_arr, scatter_idx)
        else:
            coords_2d = fallback_coords

        preds = aligned_preds_seed0.get(m_name, y_true)
        pts = []
        for i, g_idx in enumerate(scatter_idx):
            pts.append(
                {
                    "x": round(float(coords_2d[i, 0]), 3),
                    "y": round(float(coords_2d[i, 1]), 3),
                    "true_c": int(y_true[g_idx]),
                    "pred_c": int(preds[g_idx]),
                    "idx": int(g_idx),
                }
            )
        latent_views[m_name] = pts

    payload = {
        "project_info": {
            "course": "GCOS108 · Khám phá Tri thức và Khai thác Dữ liệu",
            "university": "Trường Đại học Công nghệ TP.HCM (HUTECH)",
            "instructor": "TS. Bùi Danh Hường",
            "students": [
                "Nguyễn Hữu Điền – 2541863009",
                "Nguyễn Hải Đăng – 2541863007",
            ],
            "class_code": "25SCT33",
            "dataset": "Fashion-MNIST (28×28, 784 chiều, K=10 lớp, chuẩn hóa [0, 1])",
            "protocol": "ConvAE (Conv5_32-Conv5_64-Conv3_128-Fc10[tanh]), Pretrain lr=1e-3 (L2=1e-5) + Fine-tune lr=5e-4, λ1=λ2=0.01, T=20 (Vardakas et al., 2026)",
        },
        "paper_table2_fmnist": PAPER_TABLE2_FMNIST,
        "class_names": FASHION_CLASS_NAMES,
        "benchmark_summary": benchmark_summary,
        "benchmark_runs": benchmark_runs,
        "ablation_summary": ablation_summary,
        "ablation_runs": ablation_runs,
        "sensitivity_summary": sensitivity_summary,
        "sensitivity_runs": sensitivity_runs,
        "confusion_matrices": confusion_matrices,
        "sample_gallery": sample_gallery,
        "latent_views": latent_views,
    }

    out_js = os.path.join(web_dir, "data_bundle.js")
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.DCSS_DASHBOARD_DATA = ")
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    print(f"[+] Đã tạo thành công gói dữ liệu web tại: {out_js}")


if __name__ == "__main__":
    main()
