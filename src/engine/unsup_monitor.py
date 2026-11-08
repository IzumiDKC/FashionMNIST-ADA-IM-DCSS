"""
Đóng góp 3 (Methodology Contribution):
Bộ giám sát không giám sát (Unsupervised Collapse Monitor & Model Selection)
Giải quyết giới hạn tại Mục 6.4 của Final.docx:
- Không sử dụng nhãn thật `y_true` để phát hiện lỗi sụp đổ của sigma=1 hoặc chọn checkpoint.
- Dựa hoàn toàn trên:
  1. Tỷ lệ cụm hoạt động (Active Clusters) & Tỷ lệ mẫu tối đa trong 1 cụm (Max Cluster Share).
  2. Entropy gán cụm chuẩn hóa H(Q) / ln(K) (phát hiện RBF bị bão hòa tạo phân phối đều).
  3. Điểm Hard Silhouette nội tại trên không gian tiềm ẩn Z.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np


class UnsupervisedCollapseMonitor:
    """Theo dõi trạng thái phân cụm không cần nhãn và lưu checkpoint tốt nhất theo tiêu chuẩn nội tại."""

    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters
        self.max_entropy = float(np.log(max(n_clusters, 2)))
        self.best_unsup_score: float = -1e9
        self.best_epoch: int = 0
        self.best_pred: Optional[np.ndarray] = None
        self.best_unseen_pred: Optional[np.ndarray] = None
        self.collapse_detected_count: int = 0

    def check_collapse(self, unsup_metrics: Dict[str, float]) -> bool:
        """
        Phát hiện hiện tượng sụp đổ cụm (Cluster Collapse) hoặc bão hòa nhân RBF:
        - Số cụm có mẫu < 30% tổng số cụm K, HOẶC
        - Hơn 80% mẫu bị dồn vào duy nhất 1 cụm, HOẶC
        - Entropy gán cụm tiến sát ln(K) (> 98.5% max_entropy -> mọi mẫu có q_ij ~ 1/K).
        """
        active_c = unsup_metrics["active_clusters"]
        max_share = unsup_metrics["max_cluster_share"]
        assign_ent = unsup_metrics["assignment_entropy"]

        is_collapsed = (
            active_c < max(2, int(0.3 * self.n_clusters))
            or max_share > 0.80
            or assign_ent > 0.985 * self.max_entropy
        )
        if is_collapsed:
            self.collapse_detected_count += 1
        return is_collapsed

    def compute_composite_unsupervised_score(
        self, unsup_metrics: Dict[str, float], rec_loss: float
    ) -> float:
        """
        Tính điểm chất lượng nội tại tổng hợp (hoàn toàn không dùng nhãn thật y_true):
        Ưu tiên Hard Silhouette cao trên Z, đủ số cụm hoạt động, và phạt nếu sụp đổ cụm.
        """
        sil = unsup_metrics["unsup_silhouette"]
        active_ratio = unsup_metrics["active_clusters"] / float(self.n_clusters)
        max_share = unsup_metrics["max_cluster_share"]

        penalty = 0.0
        if max_share > 0.60:
            penalty += (max_share - 0.60) * 2.0
        if active_ratio < 0.70:
            penalty += (0.70 - active_ratio) * 1.5

        return float(sil + 0.15 * active_ratio - 0.1 * rec_loss - penalty)

    def update(
        self,
        epoch: int,
        unsup_metrics: Dict[str, float],
        rec_loss: float,
        current_pred: np.ndarray,
        current_unseen_pred: Optional[np.ndarray] = None,
    ) -> float:
        score = self.compute_composite_unsupervised_score(unsup_metrics, rec_loss)
        if score > self.best_unsup_score or self.best_pred is None:
            self.best_unsup_score = score
            self.best_epoch = epoch
            self.best_pred = copy.deepcopy(current_pred)
            if current_unseen_pred is not None:
                self.best_unseen_pred = copy.deepcopy(current_unseen_pred)
        return score
