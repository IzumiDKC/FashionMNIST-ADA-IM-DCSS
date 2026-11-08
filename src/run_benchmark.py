"""
Script điều khiển chạy thực nghiệm chính (CLI Entry Point).
Đã nâng cấp lên kiến trúc ConvAutoencoder28x28 + PyTorch chuẩn theo Mục 5.3 & 5.5 bài báo gốc DCSS
(Vardakas et al., Machine Learning 2026, 115:81).

Hỗ trợ 4 chế độ chạy (--suite):
1. `benchmark`: Chạy đối chứng 10 cấu hình (7 cấu hình gốc + 3 cấu hình Đóng góp mới).
2. `ablation`: Chạy bảng Ablation tách biệt từng thành phần Loss của DCSS và Ada-DCSS / IM-DCSS / Ada-IM-DCSS.
3. `sensitivity`: Chạy bảng Độ nhạy siêu tham số (sigma_init in [0.1, 0.5, 1.0, 2.0, 5.0]).
4. `all`: Chạy đồng nhất toàn bộ 3 kịch bản từ cùng một bộ checkpoint Pretrained ConvAE.

Xuất đầy đủ:
- runs.csv, summary.csv, pairwise_ari.csv
- predictions/pred_<method>_seed<seed>.npy & predictions/z_<method>_seed0.npy (biểu diễn nhúng thực tế)
- checkpoints/pretrain_seed<seed>.pt (phục vụ kiểm chứng độc lập & Live Runner trên Website)
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np

try:
    import torch

    if os.cpu_count() and os.cpu_count() > torch.get_num_threads():
        torch.set_num_threads(min(os.cpu_count(), 12))
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    ExperimentConfig,
    MethodConfig,
    get_dcss_ablation_configs,
    get_sigma_sensitivity_configs,
    get_standard_benchmark_configs,
)
from src.data_adapter import load_dataset
from src.evaluation.metrics import compute_pairwise_ari


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _sample_std(values: List[float]) -> float:
    """Tính độ lệch chuẩn mẫu (Sample SD, ddof=1)."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def export_results(
    results: list,
    method_configs: List[MethodConfig],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    runs_csv_path = os.path.join(output_dir, "runs.csv")
    with open(runs_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "family",
                "seed",
                "ACC_pct",
                "NMI_pct",
                "ARI_pct",
                "Unsup_Silhouette",
                "Active_Clusters",
                "Assignment_Entropy",
                "Finetune_Time_s",
                "Pretrain_Time_s",
                "Final_Sigma_Mean",
                "Final_Sigma_Std",
                "Collapse_Rescued",
                "Inductive_ACC_pct",
                "Inductive_NMI_pct",
                "Inductive_ARI_pct",
            ]
        )
        for r in results:
            slug = _slugify(r.method_name)
            np.save(os.path.join(pred_dir, f"pred_{slug}_seed{r.seed}.npy"), r.predictions)
            if getattr(r, "embeddings", None) is not None and r.seed == 0:
                np.save(os.path.join(pred_dir, f"z_{slug}_seed{r.seed}.npy"), r.embeddings.astype(np.float32))
            writer.writerow(
                [
                    r.method_name,
                    r.family,
                    r.seed,
                    f"{r.acc:.4f}",
                    f"{r.nmi:.4f}",
                    f"{r.ari:.4f}",
                    f"{r.unsup_silhouette:.4f}",
                    f"{r.active_clusters:.1f}",
                    f"{r.assignment_entropy:.4f}",
                    f"{r.finetune_time_sec:.4f}",
                    f"{r.pretrain_time_sec:.4f}",
                    f"{r.final_sigma_mean:.4f}",
                    f"{r.final_sigma_std:.4f}",
                    r.collapse_rescued,
                    f"{r.inductive_acc:.4f}" if not np.isnan(r.inductive_acc) else "",
                    f"{r.inductive_nmi:.4f}" if not np.isnan(r.inductive_nmi) else "",
                    f"{r.inductive_ari:.4f}" if not np.isnan(r.inductive_ari) else "",
                ]
            )

    by_method: Dict[str, list] = defaultdict(list)
    for r in results:
        by_method[r.method_name].append(r)

    summary_csv_path = os.path.join(output_dir, "summary.csv")
    pairwise_csv_path = os.path.join(output_dir, "pairwise_ari.csv")

    print("\n" + "=" * 105, flush=True)
    print(
        f"{'Phương pháp / Cấu hình':<44} | {'ACC (%)':<15} | {'NMI (%)':<15} | {'ARI (%)':<15} | {'PairARI':<7} | {'Time(s)':<7}",
        flush=True,
    )
    print("-" * 105, flush=True)

    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f_sum, open(
        pairwise_csv_path, "w", newline="", encoding="utf-8-sig"
    ) as f_pw:
        sum_writer = csv.writer(f_sum)
        pw_writer = csv.writer(f_pw)

        sum_writer.writerow(
            [
                "method",
                "n_seeds",
                "ACC_mean",
                "ACC_sd",
                "ACC_formatted",
                "NMI_mean",
                "NMI_sd",
                "NMI_formatted",
                "ARI_mean",
                "ARI_sd",
                "ARI_formatted",
                "Pairwise_ARI_mean",
                "Unsup_Silhouette_mean",
                "Finetune_Time_mean_s",
            ]
        )
        pw_writer.writerow(["method", "Pairwise_ARI_mean", "Finetune_Time_mean_s"])

        for m_cfg in method_configs:
            m_name = m_cfg.name
            m_runs = by_method.get(m_name, [])
            if not m_runs:
                continue

            accs = [x.acc for x in m_runs]
            nmis = [x.nmi for x in m_runs]
            aris = [x.ari for x in m_runs]
            sils = [x.unsup_silhouette for x in m_runs]
            times = [x.finetune_time_sec for x in m_runs]
            preds = [x.predictions for x in m_runs]

            acc_m, acc_s = float(np.mean(accs)), _sample_std(accs)
            nmi_m, nmi_s = float(np.mean(nmis)), _sample_std(nmis)
            ari_m, ari_s = float(np.mean(aris)), _sample_std(aris)
            sil_m = float(np.mean(sils))
            time_m = float(np.mean(times))
            pw_ari = compute_pairwise_ari(preds)

            acc_fmt = f"{acc_m:.2f} ± {acc_s:.2f}"
            nmi_fmt = f"{nmi_m:.2f} ± {nmi_s:.2f}"
            ari_fmt = f"{ari_m:.2f} ± {ari_s:.2f}"
            pw_fmt = f"{pw_ari:.4f}" if not np.isnan(pw_ari) else "N/A"

            print(
                f"{m_name:<44} | {acc_fmt:<15} | {nmi_fmt:<15} | {ari_fmt:<15} | {pw_fmt:<7} | {time_m:<7.2f}",
                flush=True,
            )

            sum_writer.writerow(
                [
                    m_name,
                    len(m_runs),
                    f"{acc_m:.4f}",
                    f"{acc_s:.4f}",
                    acc_fmt,
                    f"{nmi_m:.4f}",
                    f"{nmi_s:.4f}",
                    nmi_fmt,
                    f"{ari_m:.4f}",
                    f"{ari_s:.4f}",
                    ari_fmt,
                    f"{pw_ari:.4f}" if not np.isnan(pw_ari) else "",
                    f"{sil_m:.4f}",
                    f"{time_m:.4f}",
                ]
            )
            pw_writer.writerow(
                [
                    m_name,
                    f"{pw_ari:.4f}" if not np.isnan(pw_ari) else "",
                    f"{time_m:.4f}",
                ]
            )

    print("=" * 105, flush=True)
    print(f"[+] Đã xuất kết quả chi tiết tại: {os.path.abspath(output_dir)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Khung chạy thực nghiệm Phân cụm sâu chuẩn bài báo DCSS (2026) & Cải tiến Ada-DCSS / IM-DCSS / Ada-IM-DCSS"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion-mnist",
        help="Tên dataset: 'fashion-mnist' | 'synthetic' | 'digits' | 'custom'",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Đường dẫn file dữ liệu (.gz, .npz, .npy, .csv)")
    parser.add_argument("--labels-path", type=str, default=None, help="Đường dẫn file nhãn (nếu tách rời)")
    parser.add_argument(
        "--suite",
        type=str,
        default="benchmark",
        choices=["benchmark", "ablation", "sensitivity", "all"],
        help="Kịch bản thực nghiệm cần chạy",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Danh sách hạt giống ngẫu nhiên")
    parser.add_argument("--pretrain-epochs", type=int, default=80, help="Số epoch tiền huấn luyện ConvAE (mặc định: 80)")
    parser.add_argument("--finetune-epochs", type=int, default=40, help="Số epoch tối ưu phân cụm (mặc định: 40)")
    parser.add_argument("--batch-size", type=int, default=256, help="Kích thước mini-batch (mặc định: 256)")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[500, 500, 2000],
        help="Kích thước tầng ẩn cho dữ liệu bảng (ảnh 28x28 tự động dùng ConvAE theo bài báo DCSS)",
    )
    parser.add_argument("--latent-dim", type=int, default=10, help="Số chiều không gian tiềm ẩn Z (mặc định: 10)")
    parser.add_argument("--inductive-ratio", type=float, default=0.0, help="Tỷ lệ tách dữ liệu Unseen")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "numpy", "pytorch"],
        help="Backend tính toán: ưu tiên 'pytorch' (ConvAE)",
    )
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Tái sử dụng checkpoint Pretrain nếu đã có")
    parser.add_argument("--output-dir", type=str, default="outputs_fashion_mnist", help="Thư mục lưu kết quả")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_pytorch = (args.backend == "pytorch") or (args.backend == "auto" and HAS_TORCH)
    if args.backend == "pytorch" and not HAS_TORCH:
        raise RuntimeError("Bạn đã chọn --backend pytorch nhưng môi trường chưa cài đặt torch.")

    backend_label = (
        f"PyTorch ConvAE ({'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'})"
        if use_pytorch
        else "NumPy Vectorized Fallback (CPU)"
    )

    bundle = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        labels_path=args.labels_path,
        inductive_test_ratio=args.inductive_ratio,
        seed=args.seeds[0] if args.seeds else 0,
    )

    exp_cfg = ExperimentConfig(
        dataset_name=bundle.name,
        data_path=args.data_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        input_dim=bundle.input_dim,
        n_clusters=bundle.n_clusters,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
        seeds=args.seeds,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        inductive_test_ratio=args.inductive_ratio,
    )

    print(
        f"[*] Dataset: {bundle.name} | N_train={len(bundle.X_train_np)}, D={bundle.input_dim}, K={bundle.n_clusters} | Backend={backend_label}",
        flush=True,
    )
    print(
        f"[*] Giao thức DCSS (2026): seeds={exp_cfg.seeds}, pretrain={exp_cfg.pretrain_epochs} ep (lr=1e-3, L2=1e-5), "
        f"finetune={exp_cfg.finetune_epochs} ep (lr=5e-4), batch={exp_cfg.batch_size}",
        flush=True,
    )

    ckpt_dir = os.path.join(exp_cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if use_pytorch:
        from src.engine.pretrainer import PretrainState, pretrain_shared_autoencoder
        from src.engine.trainer import train_clustering_method

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrain_states = {}
        for seed in exp_cfg.seeds:
            ckpt_file = os.path.join(ckpt_dir, f"pretrain_seed{seed}_ep{exp_cfg.pretrain_epochs}.pt")
            if args.reuse_checkpoints and os.path.exists(ckpt_file):
                st = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                pretrain_states[seed] = st
                print(f" -> Đã nạp checkpoint Pretrain sẵn có cho seed={seed} ({ckpt_file})", flush=True)
            else:
                print(f" -> Đang tiền huấn luyện Shared ConvAE + K-Means(n_init=100) cho seed={seed}...", end=" ", flush=True)
                st = pretrain_shared_autoencoder(
                    X_train=bundle.X_train,
                    exp_cfg=exp_cfg,
                    seed=seed,
                    device=device,
                    X_unseen=bundle.X_unseen,
                )
                torch.save(st, ckpt_file)
                pretrain_states[seed] = st
                print(f"Hoàn tất ({st.pretrain_time_sec:.2f}s, MSE={st.initial_rec_loss:.5f})", flush=True)
    else:
        from src.engine.numpy_engine import pretrain_shared_autoencoder_numpy, train_clustering_method_numpy

        pretrain_states = {}
        for seed in exp_cfg.seeds:
            print(f" -> Đang tiền huấn luyện Shared AE + K-Means cho seed={seed}...", end=" ", flush=True)
            st = pretrain_shared_autoencoder_numpy(
                X_train=bundle.X_train_np,
                exp_cfg=exp_cfg,
                seed=seed,
                X_unseen=bundle.X_unseen_np,
            )
            pretrain_states[seed] = st
            print(f"Hoàn tất ({st.pretrain_time_sec:.2f}s, MSE={st.initial_rec_loss:.5f})", flush=True)

    suites_to_run = []
    if args.suite in ("benchmark", "all"):
        suites_to_run.append(("benchmark", get_standard_benchmark_configs()))
    if args.suite in ("ablation", "all"):
        suites_to_run.append(("ablation", get_dcss_ablation_configs()))
    if args.suite in ("sensitivity", "all"):
        suites_to_run.append(("sensitivity", get_sigma_sensitivity_configs()))

    for suite_name, method_configs in suites_to_run:
        print(
            f"\n[*] Đang chạy kịch bản: {suite_name.upper()} ({len(method_configs)} cấu hình x {len(exp_cfg.seeds)} seed)",
            flush=True,
        )
        suite_results = []
        for m_cfg in method_configs:
            for seed in exp_cfg.seeds:
                if use_pytorch:
                    res = train_clustering_method(
                        method_cfg=m_cfg,
                        exp_cfg=exp_cfg,
                        pretrain_state=pretrain_states[seed],
                        X_train=bundle.X_train,
                        y_train=bundle.y_train,
                        device=device,
                        X_unseen=bundle.X_unseen,
                        y_unseen=bundle.y_unseen,
                    )
                else:
                    res = train_clustering_method_numpy(
                        method_cfg=m_cfg,
                        exp_cfg=exp_cfg,
                        pretrain_state=pretrain_states[seed],
                        X_train=bundle.X_train_np,
                        y_train=bundle.y_train,
                        X_unseen=bundle.X_unseen_np,
                        y_unseen=bundle.y_unseen,
                    )
                print(
                    f"   + {m_cfg.name:<44} [seed={seed}] -> ACC={res.acc:.2f}% | NMI={res.nmi:.2f}% | ARI={res.ari:.2f}% ({res.finetune_time_sec:.1f}s)",
                    flush=True,
                )
                suite_results.append(res)

        suite_out_dir = os.path.join(exp_cfg.output_dir, suite_name)
        export_results(suite_results, method_configs, suite_out_dir)


if __name__ == "__main__":
    main()
