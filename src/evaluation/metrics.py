"""
Hiện thực hóa các chỉ số đánh giá theo Mục 4.3 của báo cáo Final.docx:
1. ACC (%): Độ chính xác phân cụm sau khi ghép nhãn tối ưu bằng thuật toán Hungarian.
2. NMI (%): Thông tin tương hỗ chuẩn hóa (Normalized Mutual Information).
3. ARI (%): Chỉ số Rand hiệu chỉnh (Adjusted Rand Index).
4. Pairwise ARI (thang gốc 0-1): Độ ổn định phân hoạch giữa các seed (không dùng nhãn thật).
5. Unsupervised Internal Metrics: Hard Silhouette nội tại trên Z, Entropy cụm, Số cụm hoạt động
   (phục vụ Đóng góp 3 - chọn siêu tham số / phát hiện sụp đổ cụm hoàn toàn không dùng nhãn).
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


def _hungarian_match(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Giải bài toán ghép cặp tối ưu Hungarian (ưu tiên scipy, có fallback thuần NumPy)."""
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fallback tham lam nếu môi trường thiếu scipy
        n = cost_matrix.shape[0]
        row_ind, col_ind = [], []
        used_cols = set()
        for r in range(n):
            best_c = min((c for c in range(n) if c not in used_cols), key=lambda c: cost_matrix[r, c])
            row_ind.append(r)
            col_ind.append(best_c)
            used_cols.add(best_c)
        return np.array(row_ind), np.array(col_ind)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính ACC (%) bằng thuật toán Hungarian theo Mục 4.3 (trả về thang phần trăm 0-100)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.size == y_pred.size, "Kích thước y_true và y_pred phải bằng nhau."

    d = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = _hungarian_match(w.max() - w)
    acc = float(w[row_ind, col_ind].sum()) / float(y_pred.size)
    return acc * 100.0


def nmi_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính Normalized Mutual Information (%) hoàn toàn độc lập hoặc qua sklearn."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = float(y_true.size)

    classes, class_inv = np.unique(y_true, return_inverse=True)
    clusters, cluster_inv = np.unique(y_pred, return_inverse=True)

    contingency = np.zeros((len(classes), len(clusters)), dtype=np.float64)
    np.add.at(contingency, (class_inv, cluster_inv), 1.0)

    pi = contingency.sum(axis=1) / n
    pj = contingency.sum(axis=0) / n
    pij = contingency / n

    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / (np.outer(pi, pj)[nz] + 1e-15)))

    h_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    h_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    if h_true + h_pred <= 1e-15:
        return 0.0
    return float((2.0 * mi / (h_true + h_pred)) * 100.0)


def ari_score(y_true: np.ndarray, y_pred: np.ndarray, as_percentage: bool = True) -> float:
    """Tính Adjusted Rand Index (% hoặc thang gốc 0-1 cho Pairwise ARI)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = y_true.size

    _, class_inv = np.unique(y_true, return_inverse=True)
    _, cluster_inv = np.unique(y_pred, return_inverse=True)

    contingency = np.zeros((class_inv.max() + 1, cluster_inv.max() + 1), dtype=np.int64)
    np.add.at(contingency, (class_inv, cluster_inv), 1)

    sum_comb_c = np.sum(contingency * (contingency - 1)) / 2.0
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    sum_comb_a = np.sum(a * (a - 1)) / 2.0
    sum_comb_b = np.sum(b * (b - 1)) / 2.0
    total_comb = n * (n - 1) / 2.0

    expected_index = (sum_comb_a * sum_comb_b) / max(total_comb, 1.0)
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected_index
    if abs(denom) < 1e-12:
        ari = 0.0
    else:
        ari = float((sum_comb_c - expected_index) / denom)
    return ari * 100.0 if as_percentage else ari


def compute_all_supervised_metrics(y_true: Optional[np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
    """Tính đồng thời ACC (%), NMI (%), ARI (%) khi có nhãn tham chiếu hậu kiểm."""
    if y_true is None:
        return {"ACC": float("nan"), "NMI": float("nan"), "ARI": float("nan")}
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": nmi_score(y_true, y_pred),
        "ARI": ari_score(y_true, y_pred, as_percentage=True),
    }


def compute_pairwise_ari(predictions_by_seed: List[np.ndarray]) -> float:
    """
    Tính Pairwise ARI trung bình (thang gốc 0-1) giữa tất cả các cặp phân hoạch từ các seed khác nhau
    theo đúng Bảng 5.2 của Final.docx.
    """
    if len(predictions_by_seed) < 2:
        return float("nan")
    scores = [
        ari_score(p1, p2, as_percentage=False)
        for p1, p2 in combinations(predictions_by_seed, 2)
    ]
    return float(np.mean(scores))


def compute_unsupervised_metrics(
    z: np.ndarray,
    q: np.ndarray,
    max_Subsample: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Tính các chỉ số chất lượng nội tại KHÔNG DÙNG NHÃN THẬT (Unsupervised Internal Metrics):
    1. `unsup_silhouette`: Hard Silhouette Score trên không gian tiềm ẩn Z (thang [-1, 1]).
    2. `active_clusters`: Số cụm thực sự có mẫu được gán (phát hiện sụp đổ cụm).
    3. `max_cluster_share`: Tỷ lệ mẫu lớn nhất rơi vào 1 cụm (nếu > 0.85 là dấu hiệu Collapse).
    4. `assignment_entropy`: Độ sắc nét trung bình của xác suất gán mềm Q.
    """
    y_pred = np.argmax(q, axis=1)
    n_samples, n_clusters = q.shape
    unique_clusters, counts = np.unique(y_pred, return_counts=True)
    active_clusters = len(unique_clusters)
    max_share = float(counts.max()) / float(n_samples)

    q_safe = np.clip(q, 1e-10, 1.0)
    assign_ent = float(-np.mean(np.sum(q_safe * np.log(q_safe), axis=1)))

    # Nếu tất cả mẫu dồn vào 1 cụm hoặc mỗi mẫu 1 cụm -> Silhouette không hợp lệ (-1.0)
    if active_clusters <= 1 or active_clusters >= n_samples:
        return {
            "unsup_silhouette": -1.0,
            "active_clusters": float(active_clusters),
            "max_cluster_share": max_share,
            "assignment_entropy": assign_ent,
        }

    # Lấy mẫu con nếu N lớn để tính Hard Silhouette cực nhanh trên CPU
    if n_samples > max_Subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, size=max_Subsample, replace=False)
        z_sub = z[idx]
        y_sub = y_pred[idx]
    else:
        z_sub = z
        y_sub = y_pred

    # Tính ma trận khoảng cách Euclid trên tập con
    diff = z_sub[:, None, :] - z_sub[None, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))

    n_sub = len(y_sub)
    sil_vals = np.zeros(n_sub, dtype=np.float32)
    sub_clusters = np.unique(y_sub)
    if len(sub_clusters) <= 1:
        hard_sil = -1.0
    else:
        for i in range(n_sub):
            c_i = y_sub[i]
            same_mask = (y_sub == c_i)
            same_mask[i] = False
            if np.any(same_mask):
                a_i = float(np.mean(dist_mat[i, same_mask]))
            else:
                a_i = 0.0

            b_i = float("inf")
            for c_other in sub_clusters:
                if c_other == c_i:
                    continue
                other_mask = (y_sub == c_other)
                if np.any(other_mask):
                    b_i = min(b_i, float(np.mean(dist_mat[i, other_mask])))
            denom = max(a_i, b_i)
            sil_vals[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
        hard_sil = float(np.mean(sil_vals))

    return {
        "unsup_silhouette": hard_sil,
        "active_clusters": float(active_clusters),
        "max_cluster_share": max_share,
        "assignment_entropy": assign_ent,
    }
