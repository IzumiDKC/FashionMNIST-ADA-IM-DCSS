"""
Trình điều phối chạy thực nghiệm chuẩn bài báo DCSS (2026) & cải tiến IM-DCSS / Ada-IM-DCSS
cho kiến trúc PyTorch ConvAutoencoder28x28 trên Fashion-MNIST:
1. Tự động ưu tiên tăng tốc trên GPU CUDA (VD: NVIDIA GeForce GTX 1060 3GB) ở chế độ đơn tiến trình
   tiết kiệm VRAM (<150 MB VRAM), hoặc fallback sang đa tiến trình CPU nếu máy không có CUDA.
2. Tiền huấn luyện 3 seed (0, 1, 2) theo giao thức bài báo DCSS (2026) (hỗ trợ tái sử dụng checkpoint sẵn có).
3. Khử trùng lặp cấu hình toán học giống hệt nhau giữa 3 bảng (Benchmark, Ablation, Sensitivity)
   để đảm bảo tính nhất quán tuyệt đối 100% giữa mọi bảng số liệu.
4. Xuất đầy đủ CSV, .npy (pred & z), dọn dẹp tệp cũ và tự động cập nhật gói dữ liệu `web/data_bundle.js`.
"""

import argparse
import copy
import dataclasses
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch

from src.config import (
    ExperimentConfig,
    MethodConfig,
    get_dcss_ablation_configs,
    get_sigma_sensitivity_configs,
    get_standard_benchmark_configs,
)
from src.data_adapter import load_dataset
from src.engine.pretrainer import PretrainState, pretrain_shared_autoencoder
from src.engine.trainer import RunResult, train_clustering_method
from src.run_benchmark import export_results


def _config_hyperparam_signature(cfg: MethodConfig) -> Tuple:
    """Tạo chữ ký siêu tham số (không phụ thuộc tên hiển thị) để đồng bộ kết quả giữa các bảng."""
    d = dataclasses.asdict(cfg)
    d.pop("name", None)
    return tuple(sorted(d.items()))


def _worker_pretrain_seed_cpu(args: Tuple[int, ExperimentConfig, str]) -> Tuple[int, str, float, float]:
    seed, exp_cfg, ckpt_path = args
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    torch.set_num_threads(4)
    bundle = load_dataset(dataset_name=exp_cfg.dataset_name, seed=seed)
    device = torch.device("cpu")
    st = pretrain_shared_autoencoder(
        X_train=bundle.X_train,
        exp_cfg=exp_cfg,
        seed=seed,
        device=device,
        X_unseen=bundle.X_unseen,
    )
    torch.save(st, ckpt_path)
    return seed, ckpt_path, st.pretrain_time_sec, st.initial_rec_loss


def _worker_finetune_job_cpu(args: Tuple[MethodConfig, ExperimentConfig, str, int]) -> Tuple[Tuple, int, RunResult]:
    method_cfg, exp_cfg, ckpt_path, seed = args
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    torch.set_num_threads(4)
    bundle = load_dataset(dataset_name=exp_cfg.dataset_name, seed=seed)
    st: PretrainState = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    res = train_clustering_method(
        method_cfg=method_cfg,
        exp_cfg=exp_cfg,
        pretrain_state=st,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        device=device,
        X_unseen=bundle.X_unseen,
        y_unseen=bundle.y_unseen,
    )
    return _config_hyperparam_signature(method_cfg), seed, res


def _clean_stale_prediction_files(out_dir: str) -> None:
    """Xóa các file .npy cũ của biến thể Consist-DCSS trước đây để thư mục kết quả sạch tuyệt đối."""
    for sub in ("benchmark", "ablation", "sensitivity"):
        pred_dir = os.path.join(out_dir, sub, "predictions")
        if os.path.isdir(pred_dir):
            for pat in ("*consist*", "*ada_dcss_ours_sigma_init*"):
                for fp in glob.glob(os.path.join(pred_dir, pat)):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy chiến dịch thực nghiệm DCSS & Ada-IM-DCSS (Ưu tiên GPU CUDA)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Danh sách hạt giống (mặc định: 0 1 2)")
    parser.add_argument("--pretrain-epochs", type=int, default=80, help="Số epoch Pretrain ConvAE (mặc định: 80)")
    parser.add_argument("--finetune-epochs", type=int, default=40, help="Số epoch Fine-tune phân cụm (mặc định: 40)")
    parser.add_argument("--force-pretrain", action="store_true", help="Ép huấn luyện lại Pretrain kể cả khi đã có checkpoint")
    parser.add_argument("--only-benchmark", action="store_true", help="Chỉ chạy 10 cấu hình Benchmark (dùng khi kiểm tra nhanh)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_global = time.perf_counter()
    out_dir = os.path.join(BASE_DIR, "outputs_fashion_mnist")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    exp_cfg = ExperimentConfig(
        dataset_name="fashion-mnist",
        output_dir=out_dir,
        seeds=list(args.seeds),
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=256,
        pretrain_lr=1e-3,
        pretrain_weight_decay=1e-5,
        finetune_lr=5e-4,
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        torch.backends.cudnn.benchmark = True
        device_desc = f"CUDA GPU ({gpu_name}, {vram_gb:.1f} GB VRAM - Low-VRAM Batch Mode)"
    else:
        device_desc = "CPU Multi-Process (4 workers)"

    print(
        f"[*] Khởi chạy Chiến dịch Thực nghiệm chuẩn bài báo DCSS (2026) + Cải tiến IM-DCSS / Ada-IM-DCSS\n"
        f"    Thiết bị  : {device_desc}\n"
        f"    Kiến trúc : ConvAutoencoder28x28 (tanh) | Pretrain={exp_cfg.pretrain_epochs} ep | Finetune={exp_cfg.finetune_epochs} ep | Seeds={exp_cfg.seeds}",
        flush=True,
    )

    # Bước 1: Tiền huấn luyện hoặc nạp checkpoint 3 seed
    ckpt_paths: Dict[int, str] = {}
    missing_seeds: List[int] = []
    for s in exp_cfg.seeds:
        cp = os.path.join(ckpt_dir, f"pretrain_seed{s}_ep{exp_cfg.pretrain_epochs}.pt")
        ckpt_paths[s] = cp
        if args.force_pretrain or not os.path.exists(cp):
            missing_seeds.append(s)

    print("\n[Bước 1/3] Chuẩn bị Checkpoint Pretrain ConvAE + K-Means(n_init=100)...", flush=True)
    if not missing_seeds:
        for s in exp_cfg.seeds:
            st_tmp: PretrainState = torch.load(ckpt_paths[s], map_location="cpu", weights_only=False)
            print(
                f"   ✓ Tái sử dụng Pretrain Seed={s} sẵn có (MSE={st_tmp.initial_rec_loss:.5f}) -> {ckpt_paths[s]}",
                flush=True,
            )
    elif use_cuda:
        device = torch.device("cuda:0")
        for s in missing_seeds:
            bundle = load_dataset(dataset_name=exp_cfg.dataset_name, seed=s)
            st = pretrain_shared_autoencoder(
                X_train=bundle.X_train,
                exp_cfg=exp_cfg,
                seed=s,
                device=device,
                X_unseen=bundle.X_unseen,
            )
            # Đưa state_dict về CPU trước khi lưu để tương thích mọi thiết bị
            st.ae_state_dict = {k: v.cpu() for k, v in st.ae_state_dict.items()}
            torch.save(st, ckpt_paths[s])
            torch.cuda.empty_cache()
            print(
                f"   ✓ Hoàn tất Pretrain GPU Seed={s} trong {st.pretrain_time_sec:.2f}s (MSE={st.initial_rec_loss:.5f})",
                flush=True,
            )
    else:
        pretrain_jobs = [(s, exp_cfg, ckpt_paths[s]) for s in missing_seeds]
        with ProcessPoolExecutor(max_workers=min(3, len(pretrain_jobs))) as ex:
            futs = [ex.submit(_worker_pretrain_seed_cpu, j) for j in pretrain_jobs]
            for fut in as_completed(futs):
                s, cp, t_sec, mse = fut.result()
                print(f"   ✓ Hoàn tất Pretrain CPU Seed={s} trong {t_sec:.1f}s (MSE={mse:.5f}) -> {cp}", flush=True)

    # Bước 2: Tập hợp các cấu hình độc nhất
    bench_cfgs = get_standard_benchmark_configs()
    ablat_cfgs = [] if args.only_benchmark else get_dcss_ablation_configs()
    sens_cfgs = [] if args.only_benchmark else get_sigma_sensitivity_configs()

    unique_map: Dict[Tuple, MethodConfig] = {}
    for cfg in bench_cfgs + ablat_cfgs + sens_cfgs:
        sig = _config_hyperparam_signature(cfg)
        if sig not in unique_map:
            unique_map[sig] = cfg

    finetune_jobs = []
    for sig, rep_cfg in unique_map.items():
        for s in exp_cfg.seeds:
            finetune_jobs.append((rep_cfg, exp_cfg, ckpt_paths[s], s))

    results_cache: Dict[Tuple[Tuple, int], RunResult] = {}
    done_cnt = 0
    total_cnt = len(finetune_jobs)

    if use_cuda:
        print(
            f"\n[Bước 2/3] Đang huấn luyện tăng tốc trên GPU CUDA cho {total_cnt} lượt chạy độc nhất...",
            flush=True,
        )
        device = torch.device("cuda:0")
        bundles_by_seed = {s: load_dataset(dataset_name=exp_cfg.dataset_name, seed=s) for s in exp_cfg.seeds}
        pretrains_by_seed: Dict[int, PretrainState] = {
            s: torch.load(ckpt_paths[s], map_location="cpu", weights_only=False) for s in exp_cfg.seeds
        }

        for rep_cfg, _, _, s in finetune_jobs:
            sig = _config_hyperparam_signature(rep_cfg)
            bundle = bundles_by_seed[s]
            st = pretrains_by_seed[s]
            res = train_clustering_method(
                method_cfg=rep_cfg,
                exp_cfg=exp_cfg,
                pretrain_state=st,
                X_train=bundle.X_train,
                y_train=bundle.y_train,
                device=device,
                X_unseen=bundle.X_unseen,
                y_unseen=bundle.y_unseen,
            )
            results_cache[(sig, s)] = res
            done_cnt += 1
            print(
                f"   [{done_cnt:02d}/{total_cnt:02d}] {res.method_name:<46} (seed={s}) -> "
                f"ACC={res.acc:.2f}% | NMI={res.nmi:.2f}% | ARI={res.ari:.2f}% | Sil={res.unsup_silhouette:.4f} ({res.finetune_time_sec:.2f}s)",
                flush=True,
            )
    else:
        print(
            f"\n[Bước 2/3] Đang chạy song song {total_cnt} lượt huấn luyện độc nhất trên CPU (4 workers)...",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_worker_finetune_job_cpu, j) for j in finetune_jobs]
            for fut in as_completed(futs):
                sig, s, res = fut.result()
                results_cache[(sig, s)] = res
                done_cnt += 1
                print(
                    f"   [{done_cnt:02d}/{total_cnt:02d}] {res.method_name:<46} (seed={s}) -> "
                    f"ACC={res.acc:.2f}% | NMI={res.nmi:.2f}% | ARI={res.ari:.2f}% ({res.finetune_time_sec:.1f}s)",
                    flush=True,
                )

    if args.only_benchmark:
        suite_list = [("benchmark", bench_cfgs)]
    else:
        _clean_stale_prediction_files(out_dir)
        suite_list = [
            ("benchmark", bench_cfgs),
            ("ablation", ablat_cfgs),
            ("sensitivity", sens_cfgs),
        ]

    # Bước 3: Ánh xạ kết quả sang các bảng và xuất CSV + .npy + web/data_bundle.js
    print("\n[Bước 3/3] Đang xuất dữ liệu tổng hợp CSV + .npy...", flush=True)
    for suite_name, cfg_list in suite_list:
        suite_results: List[RunResult] = []
        for cfg in cfg_list:
            sig = _config_hyperparam_signature(cfg)
            for s in exp_cfg.seeds:
                base_res = results_cache[(sig, s)]
                cloned = copy.copy(base_res)
                cloned.method_name = cfg.name
                suite_results.append(cloned)
        export_results(suite_results, cfg_list, os.path.join(out_dir, suite_name))

    if not args.only_benchmark and 0 in exp_cfg.seeds:
        from src.build_web_data import main as build_web_bundle

        build_web_bundle()

    print(f"\n[HOÀN TẤT TOÀN BỘ CHIẾN DỊCH TRONG {(time.perf_counter() - t_global)/60.0:.2f} PHÚT]", flush=True)


if __name__ == "__main__":
    main()
