"""Auditable DCSS / ADA-IM experiments. Training functions never receive labels.

See PROTOCOL.md before interpreting a result. No claim of bitwise reproduction of
the published DCSS scores: the paper, released code and legacy code disagree.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import struct
import time
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

VERSION = "full-fmnist-2026-11-09-v1"
UPSTREAM_COMMIT = "fa3d84c0f3e7c682729cd5ba6f856bb3ef532637"
FILES = {
    "train-images-idx3-ubyte": ("1XjvEO46beyAhBPorMEaRaFozU1m6raSS", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train-labels-idx1-ubyte": ("1vMq9NhjOpFW71TxQB-VpAdc5UtykGLvL", "25c81989df183df01b3e8a0aad5dffbe"),
    "t10k-images-idx3-ubyte": ("12klOdWX0FAUq3WB7RkakYRZviZTG1BoI", "bef4ecab320f06d8554ea6380940ec79"),
    "t10k-labels-idx1-ubyte": ("1yXJ_8hlNKFrVPWclDQuInAvauEErtKGF", "bb300cfdad3c16e7a12a480ee83cd310"),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prepare_data(folder, raw_hashes):
    """Prefer the supplied Drive files; official fallback is explicit and checked.

    raw_hashes are independently checked against the official gz files locally.
    Every Colab download must match those hashes. Never silently select 10k.
    """
    import gdown
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (file_id, md5) in FILES.items():
        path = folder / name
        if not path.exists():
            tmp = path.with_name(name + ".download")
            try:
                result = gdown.download(id=file_id, output=str(tmp), quiet=False)
                if result is None:
                    raise RuntimeError("Drive did not return a file")
                sources[name] = "user_drive:" + file_id
            except Exception as exc:
                print(f"Drive download unavailable for {name}: {type(exc).__name__}; using official identical bytes.", flush=True)
                url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name + ".gz"
                with urllib.request.urlopen(url, timeout=120) as r:
                    compressed = r.read()
                if hashlib.md5(compressed).hexdigest() != md5:
                    raise ValueError("Official compressed checksum mismatch: " + name)
                tmp.write_bytes(gzip.decompress(compressed))
                sources[name] = url
            if digest(tmp) != raw_hashes[name]:
                raise ValueError("Downloaded data differ from canonical Fashion-MNIST: " + name)
            os.replace(tmp, path)
        else:
            sources[name] = "verified_cache"
        if digest(path) != raw_hashes[name]:
            raise ValueError("Cached data checksum mismatch: " + name)
    atomic_json(folder / "download_sources.json", sources)
    return folder


def read_idx(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated IDX")
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2051:
        if len(raw) < 16:
            raise ValueError("Truncated image header")
        rows, cols = struct.unpack(">II", raw[8:16])
        if (rows, cols) != (28, 28) or len(raw) != 16 + n * rows * cols:
            raise ValueError("Wrong IDX image shape/length")
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, 1, 28, 28).copy()
    if magic == 2049 and len(raw) == 8 + n:
        labels = np.frombuffer(raw, np.uint8, offset=8).copy()
        if np.any(labels > 9):
            raise ValueError("Label outside Fashion-MNIST 0..9")
        return labels
    raise ValueError("Wrong IDX magic/length")


def load_data(folder, expected_hashes, split="all70k"):
    folder = Path(folder)
    hashes = {n: digest(folder / n) for n in FILES}
    if hashes != expected_hashes:
        raise ValueError("Dataset hashes do not match independently verified Fashion-MNIST")
    a, b = read_idx(folder / "train-images-idx3-ubyte"), read_idx(folder / "train-labels-idx1-ubyte")
    c, d = read_idx(folder / "t10k-images-idx3-ubyte"), read_idx(folder / "t10k-labels-idx1-ubyte")
    if (len(a), len(b), len(c), len(d)) != (60000, 60000, 10000, 10000):
        raise ValueError("Expected exactly 60,000 training and 10,000 test images/labels")
    for y, n in [(b, 6000), (d, 1000)]:
        if not np.array_equal(np.bincount(y, minlength=10), np.full(10, n)):
            raise ValueError("Unexpected class counts")
    if split == "all70k":
        images, labels = np.concatenate([a, c]), np.concatenate([b, d])
    elif split == "train60k":
        images, labels = a, b
    elif split == "legacy10k":
        images, labels = c, d
    else:
        raise ValueError(split)
    # Keep uint8 in CPU RAM. Convert only a mini-batch to float32 on the device.
    meta = {"sha256": hashes, "split": split, "n": len(images), "order": "train_then_test" if split == "all70k" else split,
            "normalization": "float32(pixel)/255", "class_counts_audit_only": np.bincount(labels, minlength=10).tolist()}
    return torch.from_numpy(images), labels, meta


@dataclass(frozen=True)
class Method:
    name: str
    sigma: str = "one_trainable"
    sil: float = .01
    ent: float = .01
    balance: float = 0.
    warmup: float = 0.
    ema: float = .15
    rescue: bool = False
    objective: str = "paper_normalized"
    entropy_base: str = "e"
    temperature: float = .05


METHODS = {
    "dcss_paper": Method("dcss_paper"),
    "dcss_rms": Method("dcss_rms", sigma="rms_trainable"),
    "im_rms": Method("im_rms", sigma="rms_trainable", sil=.012, ent=.0025, balance=.004, warmup=.70),
    "ada_im": Method("ada_im", sigma="adaptive", sil=.012, ent=.0025, balance=.004, warmup=.70, rescue=True),
    # Optional controls: not run by default; not needed to select the best seed.
    "ada_only": Method("ada_only", sigma="adaptive", sil=.012, balance=.0015, rescue=True),
    "author_objective_control": Method("author_objective_control", objective="author_unnormalized", entropy_base="2", temperature=.25),
}


@dataclass(frozen=True)
class Config:
    profile: str = "paper70k"
    split: str = "all70k"
    backbone: str = "author_conv_bn"
    seeds: tuple = tuple(range(10))
    methods: tuple = ("dcss_paper", "dcss_rms", "im_rms", "ada_im")
    pretrain_epochs: int = 100
    cluster_epochs: int = 100
    batch_size: int = 256
    latent_dim: int = 10
    n_clusters: int = 10
    pretrain_lr: float = 1e-3
    cluster_lr: float = 5e-4
    pretrain_weight_decay: float = 1e-5
    kmeans_n_init: int = 100
    grad_clip: float | None = None
    require_cuda: bool = True
    primary_metric: str = "ARI"
    cpu_threads: int = 2


def profile_config(name):
    if name == "paper70k":
        return Config()
    if name in {"legacy70k", "legacy10k"}:
        return Config(profile=name, split="all70k" if name == "legacy70k" else "legacy10k",
                      backbone="legacy_conv", seeds=(0, 1, 2), methods=("dcss_rms", "im_rms", "ada_im"),
                      pretrain_epochs=80, cluster_epochs=40, grad_clip=5.)
    raise ValueError(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Autoencoder(nn.Module):
    """Author Conv/BN topology with paper He init, or the legacy local topology.

    Both choices and their differences are recorded in every manifest.
    """
    def __init__(self, latent=10, backbone="author_conv_bn"):
        super().__init__()
        bn = backbone == "author_conv_bn"
        if backbone not in {"author_conv_bn", "legacy_conv"}:
            raise ValueError(backbone)
        enc = []
        for ci, co, k, p in [(1, 32, 5, 2), (32, 64, 5, 2), (64, 128, 3, 0)]:
            enc += [nn.Conv2d(ci, co, k, stride=2, padding=p), nn.ReLU()]
            if bn:
                enc += [nn.BatchNorm2d(co)]
        enc += [nn.Flatten(), nn.Linear(1152, latent), nn.Tanh()]
        if bn:
            enc += [nn.BatchNorm1d(latent)]
        dec = [nn.Linear(latent, 1152), nn.ReLU()]
        if bn:
            dec += [nn.BatchNorm1d(1152)]
        dec += [nn.Unflatten(1, (128, 3, 3))]
        for ci, co, k, p, op in [(128, 64, 3, 0, 0), (64, 32, 5, 2, 1), (32, 1, 5, 2, 1)]:
            dec += [nn.ConvTranspose2d(ci, co, k, stride=2, padding=p, output_padding=op)]
            if co != 1 or bn:
                dec += [nn.ReLU()]
                if bn:
                    dec += [nn.BatchNorm2d(co)]
            else:
                dec += [nn.Sigmoid()]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        if not bn:
            nn.init.xavier_uniform_(self.encoder[-2].weight)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def squared_distances(x, centers):
    # B*K*D is small; avoids sqrt singularity in a Gaussian RBF.
    return (x[:, None, :] - centers[None, :, :]).square().sum(-1)


class RBFHead(nn.Module):
    def __init__(self, centers, embeddings, method):
        super().__init__()
        self.method = method
        self.centers = nn.Parameter(centers.clone())
        self.register_buffer("sigma", torch.ones(len(centers), dtype=centers.dtype))
        self.register_buffer("guard_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("rescue_count", torch.zeros((), dtype=torch.long))
        d2 = squared_distances(embeddings, centers)
        g = d2.mean().clamp_min(1e-6).sqrt()
        if method.sigma == "rms_trainable":
            self.sigma.fill_(g)
        elif method.sigma == "adaptive":
            nearest = d2.argmin(1)
            local = torch.ones_like(self.sigma) * g
            for j in range(len(centers)):
                mask = nearest == j
                if mask.sum() > 2:
                    local[j] = d2[mask, j].mean().clamp_min(1e-6).sqrt()
            self.sigma.copy_((g * (.65 + .35 * local / local.mean().clamp_min(1e-6))).clamp(.55 * g, 1.80 * g))
        elif method.sigma != "one_trainable":
            raise ValueError(method.sigma)
        if method.sigma != "adaptive":
            self.log_sigma = nn.Parameter(self.sigma.log())
        else:
            self.register_parameter("log_sigma", None)

    def active_sigma(self):
        if self.log_sigma is None:
            return self.sigma
        s = self.log_sigma.exp()
        return s.clamp(.15, 4.5) if self.method.sigma == "rms_trainable" else s

    @torch.no_grad()
    def target_sigma(self, d2, q):
        g = d2.mean().clamp_min(1e-6).sqrt()
        mass = q.sum(0).clamp_min(1e-6)
        local = ((q * d2).sum(0) / mass).clamp_min(1e-6).sqrt()
        relative = local / local.mean().clamp_min(1e-6)
        correction = ((1. / q.shape[1]) / (mass / len(q)).clamp_min(1e-4)).pow(.05).clamp(.90, 1.12)
        return (g * (.65 + .35 * relative) * correction).clamp(.55 * g, 1.85 * g)

    def forward(self, z):
        d2 = squared_distances(z, self.centers)
        if self.training and self.method.sigma == "adaptive":
            with torch.no_grad():
                mean_phi = (-d2.detach() / self.sigma.square().clamp_min(1e-8)).clamp(-60., 0.).exp().mean()
                if mean_phi < .08 or mean_phi > .78:
                    q0 = torch.softmax(torch.exp(-d2.detach() / d2.detach().mean().clamp_min(1e-6)) / self.method.temperature, 1)
                    self.sigma.copy_(self.target_sigma(d2.detach(), q0))
                    self.guard_count.add_(1)
        # Clone buffer: later EMA changes cannot invalidate backward's saved tensor.
        s = self.active_sigma().clone()
        phi = torch.exp(-d2 / s.square().clamp_min(1e-8))
        q = torch.softmax(phi / self.method.temperature, 1)
        target = self.target_sigma(d2.detach(), q.detach()) if self.training and self.method.sigma == "adaptive" else None
        return q, target

    @torch.no_grad()
    def after_step(self, target):
        if target is not None:
            self.sigma.lerp_(target, self.method.ema)

    @torch.no_grad()
    def rescue(self, z):
        self.sigma.fill_(squared_distances(z, self.centers).mean().clamp_min(1e-6).sqrt())
        self.rescue_count.add_(1)


def soft_silhouette(z, q, objective="paper_normalized"):
    if len(z) < 2 or q.shape[1] < 2:
        raise ValueError("Soft silhouette requires B>=2, K>=2")
    # Explicit diagonal zero, direct cdist: gradients through both z and q.
    dist = torch.linalg.vector_norm(z[:, None, :] - z[None, :, :], dim=-1)
    dist = dist.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), 0.)
    a = dist @ q
    if objective == "paper_normalized":
        a = a / (q.sum(0, keepdim=True) - q).clamp_min(1e-8)
        k = q.shape[1]
        candidates = a[:, None, :].expand(-1, k, -1).masked_fill(torch.eye(k, dtype=torch.bool, device=z.device)[None], float("inf"))
        b = candidates.min(-1).values
    elif objective == "author_unnormalized":
        # Literal released Objectives.py indexing, preserved only as a diagnostic.
        k = q.shape[1]
        expanded = a.unsqueeze(2).repeat(1, 1, k)
        other = torch.masked_select(expanded, ~torch.eye(k, dtype=torch.bool, device=z.device)[None]).view(len(z), k - 1, k)
        b = other.min(1).values[:, torch.arange(k - 1, -1, -1, device=z.device)]
    else:
        raise ValueError(objective)
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-8)
    return (q * s).sum(1).mean()


def entropy_terms(q):
    h_cond = -(q * q.clamp_min(1e-10).log()).sum(1).mean()
    marginal = q.mean(0)
    kl_uniform = (marginal * (marginal.clamp_min(1e-10) * q.shape[1]).log()).sum()
    return h_cond, kl_uniform


def warmup_weight(epoch, epochs, ratio):
    if ratio == 0:
        return 1.
    progress = epoch / max(epochs - 1, 1)
    t = min(1., max(0., (progress - ratio) / (1. - ratio)))
    return .5 * (1. - math.cos(math.pi * t))


def loss_terms(x, reconstruction, z, q, method, epoch, epochs):
    rec = (x - reconstruction).square().mean()
    sil = soft_silhouette(z, q, method.objective)
    ent, bal = entropy_terms(q)
    ent = ent / math.log(2) if method.entropy_base == "2" else ent
    total = rec + method.sil * (1 - sil) + method.ent * warmup_weight(epoch, epochs, method.warmup) * ent + method.balance * bal
    return total, {"loss": total, "reconstruction": rec, "soft_silhouette": sil, "conditional_entropy": ent, "balance_kl": bal}


def metrics(y, pred):
    _, y = np.unique(y, return_inverse=True)
    _, pred = np.unique(pred, return_inverse=True)
    k = max(y.max(), pred.max()) + 1
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (pred, y), 1)
    row, col = linear_sum_assignment(-counts)
    return {"ACC": float(counts[row, col].sum() / len(y)),
            "NMI": float(normalized_mutual_info_score(y, pred, average_method="arithmetic")),
            "ARI": float(adjusted_rand_score(y, pred))}


def unsupervised(q):
    counts = np.bincount(q.argmax(1), minlength=q.shape[1])
    ent = float(-(q * np.log(np.maximum(q, 1e-10))).sum(1).mean())
    active, largest = int((counts > 0).sum()), float(counts.max() / len(q))
    return {"active_clusters": active, "max_cluster_share": largest, "assignment_entropy": ent,
            "cluster_counts": counts.tolist(),
            "collapsed": bool(active < max(2, int(.3 * q.shape[1])) or largest > .80 or ent > .985 * math.log(q.shape[1]))}


def batch_tensor(images, indices, device):
    x = images[indices].to(device=device, dtype=torch.float32)
    return x / 255. if images.dtype == torch.uint8 else x


@torch.no_grad()
def infer(model, images, device, batch=256, head=None):
    model.eval()
    if head is not None:
        head.eval()
    zs, qs = [], []
    for start in range(0, len(images), batch):
        z = model.encoder(batch_tensor(images, slice(start, start + batch), device))
        zs.append(z.cpu())
        if head is not None:
            qs.append(head(z)[0].cpu())
    return torch.cat(zs), torch.cat(qs) if qs else None


def epoch_batches(n, batch, seed, phase, epoch):
    # Independent deterministic shuffles, identical across compared methods.
    rng = torch.Generator().manual_seed(int(seed) * 1000003 + phase * 100003 + epoch)
    order = torch.randperm(n, generator=rng)
    chunks = list(order.split(batch))
    if len(chunks) > 1 and len(chunks[-1]) == 1:  # BatchNorm requires >=2.
        last = chunks.pop()
        chunks[-1] = torch.cat([chunks[-1], last])
    return chunks


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])


def load_checkpoint(path, identity, device="cpu"):
    # Only our own checkpoints, under this run directory. Never load arbitrary pkl.
    obj = torch.load(path, map_location=device, weights_only=False)
    if obj["identity"] != identity:
        raise ValueError("Checkpoint identity differs; refusing to mix protocols/data/code")
    return obj


def train_phase(model, images, cfg, seed, device, directory, identity, head=None, stop_after=None):
    """No labels here. Save a complete resumable state at every epoch boundary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.pt"
    is_cluster = head is not None
    total_epochs = cfg.cluster_epochs if is_cluster else cfg.pretrain_epochs
    parameters = list(model.parameters()) + (list(head.parameters()) if is_cluster else [])
    optimizer = torch.optim.Adam(parameters, lr=cfg.cluster_lr if is_cluster else cfg.pretrain_lr,
                                 weight_decay=0. if is_cluster else cfg.pretrain_weight_decay, betas=(.9, .999))
    start, history = 0, []
    if path.exists():
        # Load on CPU first. Optimizer.load_state_dict moves moment tensors to
        # parameter devices while leaving Adam's non-capturable step counter on CPU.
        saved = load_checkpoint(path, identity, "cpu")
        model.load_state_dict(saved["model"])
        if is_cluster:
            head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history = saved["next_epoch"], saved["history"]
        print(f"Resume {directory.name}: epoch {start}/{total_epochs}", flush=True)
    for epoch in range(start, total_epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if is_cluster:
            head.train()
        sums, seen = {}, 0
        for indices in epoch_batches(len(images), cfg.batch_size, seed, int(is_cluster), epoch):
            x = batch_tensor(images, indices, device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, z = model(x)
            target = None
            if is_cluster:
                q, target = head(z)
                loss, terms = loss_terms(x, reconstruction, z, q, head.method, epoch, total_epochs)
            else:
                loss = (x - reconstruction).square().mean()
                terms = {"loss": loss, "reconstruction": loss}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {directory.name}, epoch {epoch}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, cfg.grad_clip if cfg.grad_clip is not None and is_cluster else float("inf"), error_if_nonfinite=True)
            optimizer.step()
            if is_cluster:
                head.after_step(target)
            for key, val in terms.items():
                sums[key] = sums.get(key, 0.) + float(val.detach()) * len(x)
            seen += len(x)
        diagnostics = {}
        if is_cluster:
            z_all, q_all = infer(model, images, device, cfg.batch_size, head)
            diagnostics = unsupervised(q_all.numpy())
            if head.method.rescue and epoch in (0, 2) and diagnostics["collapsed"]:
                head.rescue(z_all.to(device))
                diagnostics["rescue_applied_after_epoch"] = True
            diagnostics.update(sigma_min=float(head.active_sigma().min().detach()), sigma_max=float(head.active_sigma().max().detach()),
                               guard_count=int(head.guard_count), rescue_count=int(head.rescue_count))
        elapsed = time.perf_counter() - t0
        row = {"epoch": epoch + 1, **{k: v / seen for k, v in sums.items()}, **diagnostics,
               "seconds": elapsed, "last_batch_grad_norm": float(norm), "seen": seen,
               "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.}
        history.append(row)
        atomic_torch(path, {"identity": identity, "next_epoch": epoch + 1, "model": model.state_dict(),
                           "head": head.state_dict() if is_cluster else None, "optimizer": optimizer.state_dict(),
                           "rng": capture_rng(), "history": history})
        atomic_json(directory / "history.json", history)
        if epoch == start or (epoch + 1) % 5 == 0 or epoch + 1 == total_epochs:
            print(f"{directory.name} {epoch+1}/{total_epochs}: loss={row['loss']:.5f}, {elapsed:.1f}s/epoch, ~{elapsed*(total_epochs-epoch-1)/3600:.2f}h remaining in this phase", flush=True)
    return history


def environment(device):
    packages = {}
    for name in ["torch", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "gdown", "threadpoolctl"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device": str(device), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "vram_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else None,
            "deterministic_algorithms": True, "amp": False, "tf32": False}


def make_identity(cfg, data_meta, env):
    manifest = {"version": VERSION, "config": asdict(cfg), "methods": [asdict(METHODS[m]) for m in cfg.methods],
                "data": data_meta, "environment": env, "code_sha256": digest(__file__), "upstream_commit": UPSTREAM_COMMIT}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    return identity, manifest


def run_experiment(images, labels, data_meta, cfg, output_root):
    if len(images) != len(labels) or len(images) != data_meta["n"]:
        raise ValueError("Dataset / labels / metadata length mismatch")
    if cfg.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Full experiment requires a CUDA GPU; no CPU/NumPy fallback. Select a GPU runtime in Colab.")
    device = torch.device("cuda" if cfg.require_cuda else "cpu")
    torch.set_num_threads(cfg.cpu_threads)
    seed_all(cfg.seeds[0])
    identity, manifest = make_identity(cfg, data_meta, environment(device))
    root = Path(output_root) / (cfg.profile + "_" + identity[:12])
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "manifest.json", manifest)
    # Preserve executable source beside the results and checkpoint metadata.
    (root / "experiment.py").write_bytes(Path(__file__).read_bytes())
    print("RESULT DIRECTORY:", root, flush=True)
    status = {"state": "running", "expected_runs": len(cfg.seeds) * len(cfg.methods), "identity": identity}
    atomic_json(root / "status.json", status)
    try:
        for seed in cfg.seeds:
            seed_all(seed)
            seed_dir = root / f"seed_{seed:02d}"
            model = Autoencoder(cfg.latent_dim, cfg.backbone).to(device)
            train_phase(model, images, cfg, seed, device, seed_dir / "pretrain", identity + f":{seed}:pretrain")
            pretrained = copy.deepcopy(model.state_dict())
            init_path = seed_dir / "initialization.pt"
            init_id = identity + f":{seed}:initialization"
            if init_path.exists():
                init = load_checkpoint(init_path, init_id)
                z_init, centers = init["embeddings"], init["centers"]
            else:
                z_init, _ = infer(model, images, device, cfg.batch_size)
                with threadpool_limits(limits=cfg.cpu_threads):
                    km = KMeans(n_clusters=cfg.n_clusters, n_init=cfg.kmeans_n_init, random_state=seed, algorithm="lloyd").fit(z_init.numpy())
                centers = torch.from_numpy(km.cluster_centers_).float()
                atomic_torch(init_path, {"identity": init_id, "embeddings": z_init, "centers": centers})
            # This baseline is a final evaluation of a fixed pretraining procedure.
            km_pred = squared_distances(z_init, centers).argmin(1).numpy()
            atomic_json(seed_dir / "ae_kmeans_evaluation.json", {"seed": seed, **metrics(labels, km_pred)})
            for method_name in cfg.methods:
                method = METHODS[method_name]
                method_dir = seed_dir / method_name
                result_path = method_dir / "result.json"
                run_id = identity + f":{seed}:{method_name}"
                if result_path.exists():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    if existing["identity"] != run_id:
                        raise ValueError("Result identity mismatch")
                    print(f"Already complete: seed {seed}, {method_name}", flush=True)
                    continue
                seed_all(seed)
                model.load_state_dict(pretrained)
                head = RBFHead(centers, z_init, method).to(device)
                history = train_phase(model, images, cfg, seed, device, method_dir, run_id, head)
                z, q = infer(model, images, device, cfg.batch_size, head)
                q_np, z_np = q.numpy(), z.numpy()
                pred = q_np.argmax(1)
                diagnostics = unsupervised(q_np)
                idx = np.random.default_rng(10000 + seed).choice(len(images), min(2000, len(images)), replace=False)
                unique = np.unique(pred[idx])
                sampled_sil = float(silhouette_score(z_np[idx], pred[idx])) if 1 < len(unique) < len(idx) else None
                result = {"identity": run_id, "seed": seed, "method": method_name, "profile": cfg.profile,
                          "n": len(images), "epochs": cfg.cluster_epochs, "status": "complete",
                          **metrics(labels, pred), **diagnostics, "sampled_hard_silhouette": sampled_sil,
                          "finetune_seconds": sum(h["seconds"] for h in history), "peak_gpu_gb": max(h["peak_gpu_gb"] for h in history),
                          "guard_count": int(head.guard_count), "rescue_count": int(head.rescue_count)}
                # Keep scores, predictions, labels (evaluation only) and embeddings for independent rechecking.
                tmp = method_dir / "predictions.npz.tmp"
                with tmp.open("wb") as f:
                    np.savez_compressed(f, predictions=pred, labels=labels, probabilities=q_np, embeddings=z_np, sample_indices=idx)
                os.replace(tmp, method_dir / "predictions.npz")
                atomic_json(result_path, result)
                print({k: result[k] for k in ("seed", "method", "ARI", "NMI", "ACC", "collapsed")}, flush=True)
                summarize(root, cfg)
                del head
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="interrupted_or_failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(root / "status.json", status)
        summarize(root, cfg)
        raise
    atomic_json(root / "status.json", status)
    summarize(root, cfg)
    return root


def paired_statistics(deltas):
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "mean_delta": float(d.mean()) if n else None, "sd": None, "ci95_low": None, "ci95_high": None, "sign_flip_p_two_sided": None}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    margin = float(stats.t.ppf(.975, n - 1) * sd / math.sqrt(n))
    if n <= 16:
        signs = np.asarray(list(itertools.product([-1., 1.], repeat=n)))
        p = float(np.mean(np.abs((signs * d).mean(1)) >= abs(mean) - 1e-15))
    else:
        p = None
    return {"n": n, "mean_delta": mean, "sd": sd, "ci95_low": mean - margin, "ci95_high": mean + margin, "sign_flip_p_two_sided": p}


def summarize(root, cfg):
    import pandas as pd
    root = Path(root)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("seed_*/*/result.json"))]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "per_seed.csv", index=False)
    summaries = []
    for method in cfg.methods:
        group = frame[frame.method == method]
        row = {"method": method, "completed_seeds": len(group), "expected_seeds": len(cfg.seeds),
               "complete": len(group) == len(cfg.seeds), "collapsed_runs": int(group.collapsed.sum())}
        for metric in ("ARI", "NMI", "ACC"):
            row[metric + "_mean"] = float(group[metric].mean()) if len(group) else None
            row[metric + "_sample_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    pairs = []
    for baseline in ("dcss_paper", "dcss_rms", "im_rms"):
        if baseline not in cfg.methods or "ada_im" not in cfg.methods:
            continue
        for metric in ("ARI", "NMI", "ACC"):
            a = frame[frame.method == "ada_im"].set_index("seed")[metric]
            b = frame[frame.method == baseline].set_index("seed")[metric]
            shared = a.index.intersection(b.index).sort_values()
            pairs.append({"comparison": "ada_im - " + baseline, "metric": metric,
                          "paired_seeds": shared.tolist(), "complete": len(shared) == len(cfg.seeds),
                          "confirmatory": cfg.profile == "paper70k" and baseline == "im_rms" and metric == cfg.primary_metric,
                          **paired_statistics((a.loc[shared] - b.loc[shared]).to_numpy())})
    atomic_json(root / "paired_comparisons.json", pairs)
    # Never auto-declare thesis success, particularly from partial/three-seed runs.
    atomic_json(root / "completeness.json", {"completed": len(rows), "expected": len(cfg.seeds) * len(cfg.methods),
                                           "complete": len(rows) == len(cfg.seeds) * len(cfg.methods),
                                           "selection": "fixed_final_epoch_all_seeds_including_collapse", "units": "fractions_not_percent"})


def plots(root):
    import pandas as pd
    import matplotlib.pyplot as plt
    root = Path(root)
    frame = pd.read_csv(root / "per_seed.csv")
    names = list(frame.method.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("ARI", "NMI", "ACC")):
        for seed, g in frame.groupby("seed"):
            axis.plot([names.index(x) for x in g.method], 100 * g[metric], "o-", alpha=.45, label=str(seed))
        axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
        axis.set_ylabel(metric + " (%)")
        axis.grid(alpha=.2)
    fig.suptitle("All completed seeds; fixed final epoch (check completeness.json)")
    fig.savefig(root / "paired_scores.png", dpi=180)
    plt.show()
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for path in sorted(root.glob("seed_00/*/history.json")):
        if path.parent.name == "pretrain":
            continue
        history = pd.read_json(path)
        axis.plot(history.epoch, history.soft_silhouette, label=path.parent.name)
    axis.set(xlabel="Epoch", ylabel="Mini-batch soft silhouette (seed 0)")
    axis.legend()
    fig.savefig(root / "training_soft_silhouette.png", dpi=180)
    plt.show()


def technical_smoke(device="cpu"):
    """Small numerical/architecture check, explicitly NOT experimental evidence."""
    seed_all(987654)
    torch.set_num_threads(2)
    device = torch.device(device)
    for backbone in ("author_conv_bn", "legacy_conv"):
        ae = Autoencoder(backbone=backbone).to(device)
        x = torch.rand(16, 1, 28, 28, device=device)
        rec, z = ae(x)
        assert rec.shape == x.shape and z.shape == (16, 10)
        head = RBFHead(z[:10].detach().cpu(), z.detach().cpu(), METHODS["ada_im"]).to(device)
        q, target = head(z)
        loss, _ = loss_terms(x, rec, z, q, head.method, 99, 100)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in list(ae.parameters()) + list(head.parameters()))
        head.after_step(target)
    print("Technical smoke passed on", device, "(not a research result).")


if __name__ == "__main__":
    technical_smoke()
