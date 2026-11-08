"""
Module điều khiển quá trình huấn luyện (Engine):
- numpy_engine.py: Engine tính toán Vector hóa thuần NumPy (chạy trực tiếp không cần cài torch).
- pretrainer.py / trainer.py: Engine PyTorch (tự động kích hoạt khi môi trường có cài đặt torch).
- unsup_monitor.py: Giám sát chất lượng không giám sát & tự phục hồi khi sụp đổ cụm (Đóng góp 3).
"""

from .numpy_engine import (
    NumPyPretrainState,
    RunResult,
    pretrain_shared_autoencoder_numpy,
    train_clustering_method_numpy,
)
from .unsup_monitor import UnsupervisedCollapseMonitor

try:
    from .pretrainer import PretrainState, pretrain_shared_autoencoder
    from .trainer import train_clustering_method
except ImportError:
    PretrainState = None
    pretrain_shared_autoencoder = None
    train_clustering_method = None

__all__ = [
    "NumPyPretrainState",
    "RunResult",
    "pretrain_shared_autoencoder_numpy",
    "train_clustering_method_numpy",
    "UnsupervisedCollapseMonitor",
    "PretrainState",
    "pretrain_shared_autoencoder",
    "train_clustering_method",
]
