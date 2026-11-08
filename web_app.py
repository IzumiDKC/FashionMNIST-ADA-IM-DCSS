"""
Python Web Server & Live Experiment API (`web_app.py`) cho Phương án A.
- Phục vụ trực tiếp Website Dashboard tương tác tại: http://127.0.0.1:8050
- Cung cấp API `/api/status` và `/api/run_live` kết nối trực tiếp vào Engine PyTorch ConvAutoencoder28x28
  (ưu tiên tăng tốc trên GPU CUDA như GTX 1060) để chạy trực tiếp mô hình từ Tab 5 khi báo cáo.
"""

import glob
import json
import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

try:
    import torch

    if os.cpu_count() and os.cpu_count() > torch.get_num_threads():
        torch.set_num_threads(min(os.cpu_count(), 12))
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.config import ExperimentConfig, MethodConfig
from src.data_adapter import load_dataset

# Bộ nhớ đệm (cache) dữ liệu và checkpoint Pretrain để chạy Live Demo cực nhanh (~1-3 giây trên GPU CUDA)
_CACHED_BUNDLE = None
_CACHED_PRETRAIN = None


def _get_device_label() -> str:
    if HAS_TORCH and torch.cuda.is_available():
        return f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
    if HAS_TORCH:
        return "PyTorch ConvAutoencoder28x28 (CPU)"
    return "NumPy Vectorized Fallback"


def _get_live_state():
    global _CACHED_BUNDLE, _CACHED_PRETRAIN
    if _CACHED_BUNDLE is None:
        full_bundle = load_dataset(dataset_name="fashion-mnist", seed=0)
        # Lấy mẫu phân tầng 3.000 ảnh đại diện để chạy Live ConvAE phản hồi tức thì khi thuyết trình
        rng = np.random.default_rng(0)
        sub_idx = []
        for c in range(10):
            idx_c = np.where(full_bundle.y_train == c)[0]
            sub_idx.extend(rng.choice(idx_c, size=300, replace=False).tolist())
        sub_idx = np.array(sub_idx, dtype=np.int64)
        full_bundle.X_train_np = full_bundle.X_train_np[sub_idx]
        full_bundle.y_train = full_bundle.y_train[sub_idx]
        if HAS_TORCH:
            full_bundle.X_train = torch.from_numpy(full_bundle.X_train_np).float()
        _CACHED_BUNDLE = full_bundle

    if _CACHED_PRETRAIN is None:
        exp_cfg = ExperimentConfig(
            dataset_name="fashion-mnist-live",
            input_dim=_CACHED_BUNDLE.input_dim,
            n_clusters=_CACHED_BUNDLE.n_clusters,
            latent_dim=10,
            pretrain_epochs=40,
            finetune_epochs=20,
            batch_size=256,
        )
        if HAS_TORCH:
            from src.engine.pretrainer import PretrainState, _run_kmeans_numpy, pretrain_shared_autoencoder
            from src.models.autoencoder import SharedAutoencoder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_candidates = sorted(
                glob.glob(os.path.join(BASE_DIR, "outputs_fashion_mnist", "checkpoints", "pretrain_seed0_*.pt"))
            )
            if ckpt_candidates:
                saved_st = torch.load(ckpt_candidates[-1], map_location="cpu", weights_only=False)
                ae_tmp = SharedAutoencoder(input_dim=784, latent_dim=10).to(device)
                ae_tmp.load_state_dict(saved_st.ae_state_dict)
                ae_tmp.eval()
                z_chunks = []
                with torch.no_grad():
                    for start in range(0, len(_CACHED_BUNDLE.X_train), 512):
                        bx = _CACHED_BUNDLE.X_train[start : start + 512].to(device)
                        z_chunks.append(ae_tmp.encode(bx).cpu().numpy())
                z_sub = np.concatenate(z_chunks, axis=0)
                del ae_tmp
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                c_np, km_pred = _run_kmeans_numpy(z_sub, 10, seed=0, n_init=50)
                st = PretrainState(
                    seed=0,
                    ae_state_dict=saved_st.ae_state_dict,
                    initial_centroids=torch.from_numpy(c_np).float(),
                    kmeans_pred=km_pred,
                    kmeans_unseen_pred=np.array([], dtype=np.int64),
                    pretrain_time_sec=saved_st.pretrain_time_sec,
                    initial_rec_loss=saved_st.initial_rec_loss,
                )
            else:
                st = pretrain_shared_autoencoder(_CACHED_BUNDLE.X_train, exp_cfg, seed=0, device=device)
            _CACHED_PRETRAIN = (exp_cfg, st)
        else:
            from src.engine.numpy_engine import pretrain_shared_autoencoder_numpy

            _CACHED_PRETRAIN = (
                exp_cfg,
                pretrain_shared_autoencoder_numpy(_CACHED_BUNDLE.X_train_np, exp_cfg, seed=0),
            )
    return _CACHED_BUNDLE, _CACHED_PRETRAIN[0], _CACHED_PRETRAIN[1]


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "engine": _get_device_label()}).encode("utf-8"))
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/run_live"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            method_key = body.get("method", "ada_im_dcss")
            sigma_init = float(body.get("sigma_init", 1.0))
            lambda_sil = float(body.get("lambda_sil", 0.01))
            beta_ent = float(body.get("beta_ent", 0.01))

            bundle, exp_cfg, pretrain_st = _get_live_state()

            # Với IM-DCSS / Ada-IM-DCSS: nếu người dùng giữ mặc định 0.01 trên UI thì tự dùng mức chuẩn 0.0025 sau Warmup
            im_beta = 0.0025 if abs(beta_ent - 0.01) < 1e-6 else beta_ent
            im_lambda = 0.012 if abs(lambda_sil - 0.01) < 1e-6 else lambda_sil

            if method_key == "ada_im_dcss":
                m_cfg = MethodConfig(
                    name="Ada-IM-DCSS (Ours - Full Proposed Live Run)",
                    family="dcss",
                    w_rec=1.0,
                    lambda_sil=im_lambda,
                    beta_ent=im_beta,
                    temperature=0.05,
                    sigma_mode="adaptive_ema_rms",
                    sigma_init=sigma_init,
                    ema_alpha=0.15,
                    normalize_latent_scale=True,
                    use_infomax=True,
                    lambda_bal=0.004,
                    entropy_warmup_ratio=0.70,
                    auto_rescue_collapse=True,
                )
            elif method_key == "im_dcss":
                m_cfg = MethodConfig(
                    name="IM-DCSS (Ours - InfoMax & Entropy Warmup Live Run)",
                    family="dcss",
                    w_rec=1.0,
                    lambda_sil=im_lambda,
                    beta_ent=im_beta,
                    temperature=0.05,
                    sigma_mode="static_rms",
                    use_infomax=True,
                    lambda_bal=0.004,
                    entropy_warmup_ratio=0.70,
                )
            elif method_key == "ada_dcss":
                m_cfg = MethodConfig(
                    name="Ada-DCSS (Ours - Per-Cluster EMA Live Run)",
                    family="dcss",
                    w_rec=1.0,
                    lambda_sil=im_lambda,
                    beta_ent=beta_ent,
                    balance_weight=0.15,
                    temperature=0.05,
                    sigma_mode="adaptive_ema_rms",
                    sigma_init=sigma_init,
                    ema_alpha=0.15,
                    normalize_latent_scale=True,
                    auto_rescue_collapse=True,
                )
            elif method_key == "dcss_rms":
                m_cfg = MethodConfig(
                    name="DCSS (RMS - Live Run)",
                    family="dcss",
                    w_rec=1.0,
                    lambda_sil=lambda_sil,
                    beta_ent=beta_ent,
                    temperature=0.05,
                    sigma_mode="static_rms",
                )
            elif method_key == "dcss_fixed":
                m_cfg = MethodConfig(
                    name=f"DCSS Fixed sigma={sigma_init} (Live Run)",
                    family="dcss",
                    w_rec=1.0,
                    lambda_sil=lambda_sil,
                    beta_ent=beta_ent,
                    temperature=0.05,
                    sigma_mode="fixed",
                    sigma_init=sigma_init,
                )
            elif method_key == "decra":
                m_cfg = MethodConfig(
                    name="DECRA (2024 - Live Run)",
                    family="decra",
                    w_rec=1.0,
                    gamma_kl=0.5,
                    use_affine_branch=True,
                    eta_cons=0.5,
                )
            elif method_key == "idec":
                m_cfg = MethodConfig(name="IDEC (2017 - Live Run)", family="idec", w_rec=1.0, gamma_kl=0.1)
            else:
                m_cfg = MethodConfig(name="DEC (2016 - Live Run)", family="dec", w_rec=0.0, gamma_kl=1.0)

            if HAS_TORCH:
                from src.engine.trainer import train_clustering_method

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                res = train_clustering_method(
                    method_cfg=m_cfg,
                    exp_cfg=exp_cfg,
                    pretrain_state=pretrain_st,
                    X_train=bundle.X_train,
                    y_train=bundle.y_train,
                    device=device,
                )
            else:
                from src.engine.numpy_engine import train_clustering_method_numpy

                res = train_clustering_method_numpy(
                    method_cfg=m_cfg,
                    exp_cfg=exp_cfg,
                    pretrain_state=pretrain_st,
                    X_train=bundle.X_train_np,
                    y_train=bundle.y_train,
                )

            log_text = (
                "================================================================================\n"
                f" [{_get_device_label().upper()}] KẾT QUẢ CHẠY TRỰC TIẾP TRÊN FASHION-MNIST\n"
                "================================================================================\n"
                f" • Mô hình vừa chạy     : {res.method_name}\n"
                f" • Siêu tham số         : sigma_init={sigma_init} | lambda_1={m_cfg.lambda_sil} | lambda_2={m_cfg.beta_ent} | T=20\n"
                "--------------------------------------------------------------------------------\n"
                f" • ACC (Hungarian)      : {res.acc:.2f} %\n"
                f" • NMI                  : {res.nmi:.2f} %   (Bài báo DCSS 2026: 63 ± 2 %)\n"
                f" • ARI                  : {res.ari:.2f} %   (Bài báo DCSS 2026: 45 ± 3 %)\n"
                f" • Unsup Silhouette (Z) : {res.unsup_silhouette:.4f}\n"
                f" • Số cụm hoạt động     : {int(res.active_clusters)} / 10\n"
                f" • Băng thông σ cuối    : {res.final_sigma_mean:.4f} ± {res.final_sigma_std:.4f}\n"
                f" • Thời gian Fine-tune  : {res.finetune_time_sec:.2f} giây\n"
                "================================================================================"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "log": log_text}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def main() -> None:
    port = 8050
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"[+] Đang khởi động DCSS Research Dashboard Server ({_get_device_label()}) tại: {url}")
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardRequestHandler)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đã dừng server.")


if __name__ == "__main__":
    main()
