"""Auditable DCSS / ADA-IM experiments. Training functions never receive labels.

See PROTOCOL.md before interpreting a result. No claim of bitwise reproduction of
the published DCSS scores: the paper, released code and legacy code disagree.
"""
from __future__ import annotations

import copy
import ast
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
import uuid
import warnings
import urllib.request
from datetime import datetime, timezone
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

VERSION = "full-fmnist-2026-11-10-portable-v2"
LEGACY_CODE_SHA256 = "b29c6c15ccc41ac03adea4edca8527f9971c70ab923c291d81c889421bbe5703"
SCIENTIFIC_VERSION = "dcss-equations-ada-2026-11-09"
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


@dataclass(frozen=True)
class RuntimeOptions:
    # Execution settings are deliberately outside the scientific Config/hash.
    device: str = "auto"
    allow_cpu: bool = True
    checkpoint_seconds: float = 300.
    max_session_hours: float = 0.
    allow_software_change: bool = True
    resume_folder: str | None = None


class PauseRequested(RuntimeError):
    """A requested time budget ended at a completed optimizer-step boundary."""


class ResumeCompatibilityError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_device(runtime):
    if runtime.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto/cuda/cpu. TPU requires a separate XLA implementation.")
    if runtime.device != "cpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if runtime.device == "cuda" or not runtime.allow_cpu:
        raise RuntimeError("No CUDA GPU available. Select A100/L4/T4 (or another CUDA GPU) in Colab, or enable allow_cpu.")
    print("CPU MODE: same PyTorch model and checkpoint; training will be much slower. GPU selection remains in the Colab UI.", flush=True)
    return torch.device("cpu")


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


def restore_rng(rng, fallback_seed=0):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    note = "restored_cpu_rng"
    if torch.cuda.is_available():
        if rng.get("cuda"):
            # The experiment uses one GPU. Restore device 0 even if the number
            # of visible devices changed; never pass nonexistent device indices.
            try:
                torch.cuda.set_rng_state(rng["cuda"][0].cpu(), device=0)
                note = "restored_cpu_and_cuda0_rng"
            except RuntimeError as exc:
                torch.cuda.manual_seed_all(fallback_seed)
                note = "cuda_rng_reseeded_incompatible_state:" + str(exc)
        else:
            torch.cuda.manual_seed_all(fallback_seed)
            note = "cuda_rng_seeded_from_seed_after_cpu_checkpoint"
    elif rng.get("cuda"):
        note = "restored_cpu_rng_cuda_state_not_used_on_cpu"
    return note


def load_checkpoint(path, identity, device="cpu"):
    # Only our own checkpoints, under this run directory. Never load arbitrary pkl.
    obj = torch.load(path, map_location=device, weights_only=False)
    if obj["identity"] != identity:
        raise ValueError("Checkpoint identity differs; refusing to mix protocols/data/code")
    return obj


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def save_training_checkpoint(directory, obj):
    """Two immutable-until-next-generation slots with a committed JSON pointer.

    A crash during an inactive-slot write leaves the previous committed slot
    intact. The old v1 checkpoint.pt is never overwritten by this format.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "checkpoint_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    current = index.get("current")
    # If current is damaged but previous is valid, preserve the valid slot.
    for ref in [index.get("current"), index.get("previous")]:
        if ref and (directory / ref["file"]).exists() and digest(directory / ref["file"]) == ref["sha256"]:
            current = ref
            break
    filename = "checkpoint_B.pt" if current and current["file"] == "checkpoint_A.pt" else "checkpoint_A.pt"
    path = directory / filename
    atomic_torch(path, cpu_state(obj))
    ref = {"file": filename, "sha256": digest(path), "next_epoch": obj["next_epoch"],
           "next_batch": obj.get("next_batch", 0), "saved_at": utc_now()}
    atomic_json(index_path, {"schema": 2, "identity": obj["identity"], "current": ref, "previous": current})
    return ref


def load_training_checkpoint(directory, identity):
    directory = Path(directory)
    index_path = directory / "checkpoint_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["identity"] != identity:
            raise ResumeCompatibilityError("Checkpoint index identity differs")
        errors = []
        for position, ref in enumerate([index.get("current"), index.get("previous")]):
            if not ref:
                continue
            path = directory / ref["file"]
            if path.parent.resolve() != directory.resolve():
                raise ResumeCompatibilityError("Checkpoint slot outside run directory")
            try:
                if digest(path) != ref["sha256"]:
                    raise IOError("SHA256 mismatch")
                obj = load_checkpoint(path, identity)
                if (obj["next_epoch"], obj.get("next_batch", 0)) != (ref["next_epoch"], ref["next_batch"]):
                    raise ResumeCompatibilityError("Checkpoint progress differs from committed index")
                if position:
                    warnings.warn("Newest checkpoint damaged; recovered previous committed checkpoint: " + str(path))
                obj["_recovered_previous_slot"] = bool(position)
                return obj
            except (OSError, RuntimeError, EOFError) as exc:
                errors.append(f"{path.name}: {exc}")
        raise IOError("No valid committed checkpoint. Refusing to restart from zero: " + "; ".join(errors))
    path = directory / "checkpoint.pt"
    if path.exists():
        return load_checkpoint(path, identity)
    return None


def restore_optimizer(optimizer, state, device):
    optimizer.load_state_dict(state)
    # Adam moments must follow the parameters. Non-capturable Adam keeps its
    # scalar step counter on CPU, including when continuing a CPU checkpoint on CUDA.
    for group in optimizer.param_groups:
        capture = bool(group.get("capturable", False) or group.get("fused", False))
        for parameter in group["params"]:
            for key, value in optimizer.state[parameter].items():
                if torch.is_tensor(value):
                    target = device if key != "step" or capture else torch.device("cpu")
                    optimizer.state[parameter][key] = value.to(target)


def train_phase(model, images, cfg, seed, device, directory, identity, head=None, stop_after=None,
                runtime=None, session=None, stop_after_batches=None):
    """Label-free training; resumable at committed optimizer-step boundaries."""
    runtime = runtime or RuntimeOptions(device=str(device).split(":")[0])
    session = session or {"session_id": "local", "environment": environment(device), "deadline": None}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    is_cluster = head is not None
    total_epochs = cfg.cluster_epochs if is_cluster else cfg.pretrain_epochs
    parameters = list(model.parameters()) + (list(head.parameters()) if is_cluster else [])
    optimizer = torch.optim.Adam(parameters, lr=cfg.cluster_lr if is_cluster else cfg.pretrain_lr,
                                 weight_decay=0. if is_cluster else cfg.pretrain_weight_decay, betas=(.9, .999))
    start, history, resume_batch, sums, seen, carried_seconds = 0, [], 0, {}, 0, 0.
    saved = load_training_checkpoint(directory, identity)
    if saved is not None:
        model.load_state_dict(saved["model"])
        if is_cluster:
            head.load_state_dict(saved["head"])
        restore_optimizer(optimizer, saved["optimizer"], device)
        rng_note = restore_rng(saved["rng"], seed)
        start, history = saved["next_epoch"], saved["history"]
        resume_batch = saved.get("next_batch", 0)
        sums, seen = saved.get("partial_sums", {}), saved.get("partial_seen", 0)
        carried_seconds = saved.get("partial_seconds", 0.)
        if not 0 <= start <= total_epochs or len(history) != start:
            raise ResumeCompatibilityError("Checkpoint epoch/history are inconsistent")
        atomic_json(directory / ("resume_" + session["session_id"] + ".json"), {
            "session_id": session["session_id"], "at": utc_now(), "epoch": start, "next_batch": resume_batch,
            "rng": rng_note, "previous_environment": saved.get("last_environment"),
            "current_environment": session["environment"], "previous_slot_recovered": saved.get("_recovered_previous_slot", False),
        })
        print(f"Resume {directory.parent.name}/{directory.name}: {start}/{total_epochs} epochs done, next batch {resume_batch}", flush=True)

    trace = list(saved.get("execution_trace", [])) if saved else []
    if saved and not trace:
        trace.append({"session_id": "legacy_v1", "start_epoch": 0, "start_batch": 0,
                      "environment": session.get("legacy_environment", saved.get("last_environment")),
                      "note": "Imported v1 checkpoint; original manifest environment, exact session boundaries unavailable"})
    if start < total_epochs:
        trace.append({"session_id": session["session_id"], "start_epoch": start, "start_batch": resume_batch,
                      "environment": session["environment"]})

    def commit(epoch, batch, partial, count, seconds, peak):
        payload = {"schema": 2, "identity": identity, "next_epoch": epoch, "next_batch": batch,
                   "model": model.state_dict(), "head": head.state_dict() if is_cluster else None,
                   "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history,
                   "partial_sums": partial, "partial_seen": count, "partial_seconds": seconds,
                   "partial_peak_gpu_gb": peak, "last_session_id": session["session_id"],
                   "last_environment": session["environment"], "execution_trace": trace}
        save_training_checkpoint(directory, payload)
        atomic_json(directory / "history.json", history)

    if saved is None:
        commit(0, 0, {}, 0, 0., 0.)
    last_commit = time.monotonic()
    steps_this_call = 0
    for epoch in range(start, total_epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if is_cluster:
            head.train()
        if epoch != start or resume_batch == 0:
            sums, seen, carried_seconds = {}, 0, 0.
        peak = saved.get("partial_peak_gpu_gb", 0.) if saved and epoch == start and resume_batch else 0.
        batches = epoch_batches(len(images), cfg.batch_size, seed, int(is_cluster), epoch)
        first_batch = resume_batch if epoch == start else 0
        if not 0 <= first_batch <= len(batches) or seen != sum(len(b) for b in batches[:first_batch]):
            raise ResumeCompatibilityError("Checkpoint batch cursor does not match sample count")
        for batch_number in range(first_batch, len(batches)):
            if session.get("deadline") and time.monotonic() >= session["deadline"]:
                commit(epoch, batch_number, sums, seen, carried_seconds + time.perf_counter() - t0, peak)
                raise PauseRequested("Session time budget reached; checkpoint saved. Run all again to continue.")
            indices = batches[batch_number]
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
            steps_this_call += 1
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device) / 1024**3)
            if time.monotonic() - last_commit >= runtime.checkpoint_seconds or (stop_after_batches is not None and steps_this_call >= stop_after_batches):
                elapsed = carried_seconds + time.perf_counter() - t0
                commit(epoch, batch_number + 1, sums, seen, elapsed, peak)
                last_commit = time.monotonic()
                if stop_after_batches is not None and steps_this_call >= stop_after_batches:
                    return history
        diagnostics = {}
        if is_cluster:
            z_all, q_all = infer(model, images, device, cfg.batch_size, head)
            diagnostics = unsupervised(q_all.numpy())
            if head.method.rescue and epoch in (0, 2) and diagnostics["collapsed"]:
                head.rescue(z_all.to(device))
                diagnostics["rescue_applied_after_epoch"] = True
            diagnostics.update(sigma_min=float(head.active_sigma().min().detach()), sigma_max=float(head.active_sigma().max().detach()),
                               guard_count=int(head.guard_count), rescue_count=int(head.rescue_count))
        elapsed = carried_seconds + time.perf_counter() - t0
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 1024**3)
        row = {"epoch": epoch + 1, **{k: v / seen for k, v in sums.items()}, **diagnostics,
               "seconds": elapsed, "last_batch_grad_norm": float(norm) if first_batch < len(batches) else None, "seen": seen,
               "peak_gpu_gb": peak, "session_id": session["session_id"],
               "epoch_completed_device": str(device), "epoch_completed_gpu": session["environment"].get("gpu"),
               "epoch_resumed_midway": bool(first_batch)}
        history.append(row)
        commit(epoch + 1, 0, {}, 0, 0., 0.)
        last_commit = time.monotonic()
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
            "device": device.type, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "vram_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else None,
            "deterministic_algorithms": True, "amp": False, "tf32": False}


def scientific_config(config):
    config = asdict(config) if isinstance(config, Config) else dict(config)
    # require_cuda/cpu_threads in v1 affected execution, not the mathematical protocol.
    return {k: v for k, v in config.items() if k not in {"require_cuda", "cpu_threads"}}


def mathematical_code_hash(source=None):
    # Hash the AST of all numerical definitions, ignoring formatting/comments.
    # Orchestration and runtime provenance may evolve without changing the science.
    source = Path(__file__).read_text(encoding="utf-8") if source is None else source
    names = {"Autoencoder", "RBFHead", "squared_distances", "soft_silhouette", "entropy_terms",
             "warmup_weight", "loss_terms", "metrics", "unsupervised", "batch_tensor", "infer", "epoch_batches"}
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise ResumeCompatibilityError("Missing numerical definition in implementation")
    # Also fingerprint the actual Adam constructor, optimization step and
    # epoch-end ADA rescue, while excluding checkpoint/logging orchestration.
    trainer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_phase")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    optimizer = next(node for node in ast.walk(trainer) if assigns(node, "optimizer"))
    loop = next(node for node in ast.walk(trainer) if isinstance(node, ast.For) and any(assigns(s, "x") for s in node.body))
    first = next(i for i, node in enumerate(loop.body) if assigns(node, "x"))
    last = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "seen")
    epoch_loop = next(node for node in ast.walk(trainer) if isinstance(node, ast.For) and any(assigns(s, "diagnostics") for s in node.body))
    diagnostic_pos = next(i for i, node in enumerate(epoch_loop.body) if assigns(node, "diagnostics"))
    nodes += [optimizer, *loop.body[first:last + 1], epoch_loop.body[diagnostic_pos + 1]]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def scientific_spec(cfg, data_meta):
    return json_normalize({"scientific_version": SCIENTIFIC_VERSION, "config": scientific_config(cfg),
                           "methods": [asdict(METHODS[m]) for m in cfg.methods], "data": data_meta,
                           "mathematical_code_sha256": mathematical_code_hash(),
                           "training_semantics": "Adam(.9,.999),fp32,noAMP,noTF32,deterministic,epoch_batches_v1,no_label_selection"})


def make_identity(cfg, data_meta, env):
    spec = scientific_spec(cfg, data_meta)
    identity = hash_json(spec)
    manifest = {"schema": 2, "version": VERSION, "identity": identity, "scientific_spec": spec,
                "config": asdict(cfg), "methods": spec["methods"], "data": data_meta,
                "environment": env, "code_sha256": digest(__file__), "upstream_commit": UPSTREAM_COMMIT}
    return identity, manifest


def compatible_run(root, cfg, data_meta):
    """Return original checkpoint identity if compatible; never select by score."""
    root = Path(root)
    path = root / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    spec = scientific_spec(cfg, data_meta)
    portable = root / "portable_resume.json"
    if portable.exists():
        adapter = json.loads(portable.read_text(encoding="utf-8"))
        if adapter["scientific_spec"] != spec:
            return None
        return adapter["checkpoint_identity"], manifest
    if manifest.get("schema") == 2:
        if manifest["scientific_spec"] != spec or manifest["identity"] != hash_json(spec):
            return None
        return manifest["identity"], manifest
    # Import only the exact previously delivered implementation, with identical
    # data, all method coefficients and the full scientific config.
    if manifest.get("code_sha256") != LEGACY_CODE_SHA256:
        return None
    if json_normalize(scientific_config(manifest["config"])) != spec["config"] or manifest["data"] != spec["data"] or manifest["methods"] != spec["methods"]:
        return None
    old_code = root / "experiment.py"
    if not old_code.exists() or digest(old_code) != LEGACY_CODE_SHA256:
        raise ResumeCompatibilityError("v1 source is missing/modified; refusing unverifiable migration: " + str(root))
    if mathematical_code_hash(old_code.read_text(encoding="utf-8")) != spec["mathematical_code_sha256"]:
        raise ResumeCompatibilityError("v1 mathematics differs; cannot resume as the same experiment")
    old_identity = hash_json(manifest)
    # Status is useful but never trusted instead of recomputing the v1 hash.
    status = root / "status.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8")).get("identity") != old_identity:
        raise ResumeCompatibilityError("v1 status/manifest identity mismatch")
    return old_identity, manifest


def resolve_run(output_root, cfg, data_meta, env, runtime):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    identity, new_manifest = make_identity(cfg, data_meta, env)
    if runtime.resume_folder:
        root = Path(runtime.resume_folder)
        if not root.is_absolute():
            root = output_root / root
        match = compatible_run(root, cfg, data_meta)
        if match is None:
            raise ResumeCompatibilityError("Selected folder is not compatible: " + str(root))
        identity, manifest = match
    else:
        candidates, incompatible_progress = [], []
        for root in sorted(output_root.glob(cfg.profile + "_*")):
            if not root.is_dir():
                continue
            match = compatible_run(root, cfg, data_meta)
            if match:
                candidates.append((root, *match))
            elif any(root.glob("seed_*/*/checkpoint*.pt")) or any(root.glob("seed_*/*/result.json")):
                incompatible_progress.append(str(root))
        if len(candidates) > 1:
            raise ResumeCompatibilityError("Several compatible runs exist; set RESUME_FOLDERS for this profile instead of mixing them: " + str([str(x[0]) for x in candidates]))
        if candidates:
            root, identity, manifest = candidates[0]
        else:
            if incompatible_progress:
                raise ResumeCompatibilityError("Existing progress has an incompatible config/code/dataset. Refusing a silent restart. Inspect or choose a new OUTPUT_ROOT explicitly: " + str(incompatible_progress))
            root = output_root / (cfg.profile + "_" + identity[:12])
            if root.exists() and any(root.iterdir()):
                raise ResumeCompatibilityError("Target folder has files but no compatible manifest")
            root.mkdir(parents=True, exist_ok=True)
            manifest = new_manifest
            atomic_json(root / "manifest.json", manifest)
            (root / "experiment.py").write_bytes(Path(__file__).read_bytes())
    # Preserve original manifest, source, .pt and scores. Adapter is additive.
    portable_path = root / "portable_resume.json"
    if not portable_path.exists():
        atomic_json(portable_path, {"schema": 2, "created_at": utc_now(), "checkpoint_identity": identity,
                                   "scientific_spec": new_manifest["scientific_spec"], "original_manifest_sha256": digest(root / "manifest.json"),
                                   "migrated_from_v1": manifest.get("schema") != 2})
    else:
        adapter = json.loads(portable_path.read_text(encoding="utf-8"))
        if adapter["original_manifest_sha256"] != digest(root / "manifest.json"):
            raise ResumeCompatibilityError("Original manifest changed after portable resume registration")
    return root, identity, manifest


def start_session(root, original_manifest, device, runtime):
    env = environment(device)
    previous = original_manifest["environment"]
    existing = sorted((root / "sessions").glob("*.json")) if (root / "sessions").exists() else []
    if existing:
        previous = json.loads(existing[-1].read_text(encoding="utf-8"))["environment"]
    software_keys = ("python", "packages", "cuda", "cudnn")
    software_changed = any(previous.get(k) != env.get(k) for k in software_keys)
    hardware_changed = any(previous.get(k) != env.get(k) for k in ("device", "gpu", "vram_gb"))
    if software_changed and not runtime.allow_software_change:
        raise ResumeCompatibilityError("Software changed. Restore the previous runtime version or enable recorded software migration.")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    session = {"session_id": session_id, "started_at": utc_now(), "state": "running", "environment": env,
               "previous_environment": previous, "software_changed": software_changed, "hardware_changed": hardware_changed,
               "runtime_options": asdict(runtime), "code_sha256": digest(__file__),
               "cross_environment_bitwise_equivalence_claimed": False}
    atomic_json(root / "sessions" / (session_id + ".json"), session)
    code_archive = root / "implementations" / (digest(__file__) + ".py")
    code_archive.parent.mkdir(exist_ok=True)
    if not code_archive.exists():
        code_archive.write_bytes(Path(__file__).read_bytes())
    if software_changed or hardware_changed:
        print("Continuing saved state on a changed runtime; change recorded. Cross-device bitwise equality is not guaranteed.", flush=True)
    return {**session, "deadline": time.monotonic() + runtime.max_session_hours * 3600 if runtime.max_session_hours > 0 else None,
            "legacy_environment": original_manifest["environment"]}


def checkpoint_trace(directory, identity, legacy_environment):
    checkpoint = load_training_checkpoint(directory, identity)
    if checkpoint is None:
        return []
    return checkpoint.get("execution_trace") or [{"session_id": "legacy_v1", "environment": legacy_environment}]


def run_experiment(images, labels, data_meta, cfg, output_root, runtime=None):
    if len(images) != len(labels) or len(images) != data_meta["n"]:
        raise ValueError("Dataset / labels / metadata length mismatch")
    # Keep CPU-only test callers working; notebook supplies explicit runtime policy.
    runtime = runtime or RuntimeOptions(device="auto" if cfg.require_cuda else "cpu")
    device = select_device(runtime)
    torch.set_num_threads(cfg.cpu_threads)
    seed_all(cfg.seeds[0])
    root, identity, manifest = resolve_run(output_root, cfg, data_meta, environment(device), runtime)
    session = start_session(root, manifest, device, runtime)
    print("RESULT DIRECTORY:", root, flush=True)
    status = {"state": "running", "expected_runs": len(cfg.seeds) * len(cfg.methods), "identity": identity}
    atomic_json(root / "status.json", status)
    try:
        for seed in cfg.seeds:
            seed_all(seed)
            seed_dir = root / f"seed_{seed:02d}"
            done = []
            for name in cfg.methods:
                result = seed_dir / name / "result.json"
                if result.exists():
                    record = json.loads(result.read_text(encoding="utf-8"))
                    if record["identity"] != identity + f":{seed}:{name}":
                        raise ResumeCompatibilityError("Existing result has a different identity")
                    done.append(name)
            if len(done) == len(cfg.methods):
                print(f"Already complete: seed {seed}, all methods. No GPU training needed.", flush=True)
                continue
            model = Autoencoder(cfg.latent_dim, cfg.backbone).to(device)
            train_phase(model, images, cfg, seed, device, seed_dir / "pretrain", identity + f":{seed}:pretrain", runtime=runtime, session=session)
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
            if not (seed_dir / "ae_kmeans_evaluation.json").exists():
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
                history = train_phase(model, images, cfg, seed, device, method_dir, run_id, head, runtime=runtime, session=session)
                z, q = infer(model, images, device, cfg.batch_size, head)
                q_np, z_np = q.numpy(), z.numpy()
                pred = q_np.argmax(1)
                diagnostics = unsupervised(q_np)
                idx = np.random.default_rng(10000 + seed).choice(len(images), min(2000, len(images)), replace=False)
                unique = np.unique(pred[idx])
                sampled_sil = float(silhouette_score(z_np[idx], pred[idx])) if 1 < len(unique) < len(idx) else None
                trace = checkpoint_trace(seed_dir / "pretrain", identity + f":{seed}:pretrain", manifest["environment"])
                trace += checkpoint_trace(method_dir, run_id, manifest["environment"])
                trace += [{"session_id": session["session_id"], "phase": "final_evaluation", "environment": session["environment"]}]
                environments = {hash_json(x.get("environment")) for x in trace}
                result = {"identity": run_id, "seed": seed, "method": method_name, "profile": cfg.profile,
                          "n": len(images), "epochs": cfg.cluster_epochs, "status": "complete",
                          **metrics(labels, pred), **diagnostics, "sampled_hard_silhouette": sampled_sil,
                          "finetune_seconds": sum(h["seconds"] for h in history), "peak_gpu_gb": max(h["peak_gpu_gb"] for h in history),
                          "guard_count": int(head.guard_count), "rescue_count": int(head.rescue_count),
                          "mixed_environment": len(environments) > 1, "environment_count": len(environments),
                          "execution_trace": trace, "evaluation_environment": session["environment"]}
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
        status.update(state="paused" if isinstance(exc, PauseRequested) else "interrupted_or_failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(root / "status.json", status)
        summarize(root, cfg)
        raise
    finally:
        report = {k: v for k, v in session.items() if k not in {"deadline", "legacy_environment"}}
        report.update(ended_at=utc_now(), state=status["state"])
        atomic_json(root / "sessions" / (session["session_id"] + ".json"), report)
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
        atomic_json(root / "completeness.json", {"completed": 0, "expected": len(cfg.seeds) * len(cfg.methods), "complete": False})
        return
    manifest_path = root / "manifest.json"
    original_env = json.loads(manifest_path.read_text(encoding="utf-8")).get("environment") if manifest_path.exists() else None
    all_envs = set()
    for row in rows:
        trace = row.get("execution_trace") or [{"environment": original_env}]
        known = [t.get("environment") for t in trace if t.get("environment")]
        all_envs.update(hash_json(env) for env in known)
        row["mixed_environment"] = row.get("mixed_environment", False)
        row["environment_provenance"] = "recorded" if row.get("execution_trace") else "legacy_manifest_only" if original_env else "unknown"
    frame = pd.DataFrame(rows)
    campaign_mixed = len(all_envs) > 1 or bool(frame.mixed_environment.any())
    frame.to_csv(root / "per_seed.csv", index=False)
    summaries = []
    for method in cfg.methods:
        group = frame[frame.method == method]
        row = {"method": method, "completed_seeds": len(group), "expected_seeds": len(cfg.seeds),
               "complete": len(group) == len(cfg.seeds), "collapsed_runs": int(group.collapsed.sum()),
               "mixed_environment_runs": int(group.mixed_environment.sum()), "campaign_mixed_environment": campaign_mixed}
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
                          "planned_primary": cfg.profile == "paper70k" and baseline == "im_rms" and metric == cfg.primary_metric,
                          "confirmatory": cfg.profile == "paper70k" and baseline == "im_rms" and metric == cfg.primary_metric and not campaign_mixed and len(shared) == len(cfg.seeds),
                          "campaign_mixed_environment": campaign_mixed,
                          **paired_statistics((a.loc[shared] - b.loc[shared]).to_numpy())})
    atomic_json(root / "paired_comparisons.json", pairs)
    # Never auto-declare thesis success, particularly from partial/three-seed runs.
    atomic_json(root / "completeness.json", {"completed": len(rows), "expected": len(cfg.seeds) * len(cfg.methods),
                                           "complete": len(rows) == len(cfg.seeds) * len(cfg.methods),
                                           "selection": "fixed_final_epoch_all_seeds_including_collapse", "units": "fractions_not_percent",
                                           "campaign_mixed_environment": campaign_mixed,
                                           "interpretation": "Hardware/software changed: results retained, timings confounded; inspect execution traces before thesis claims." if campaign_mixed else "Inspect all seeds and paired intervals; no automatic success declaration."})


def progress_inventory(output_root):
    """Read small JSON files only, including v1 histories, before GPU work."""
    rows = []
    for manifest_path in sorted(Path(output_root).glob("*/manifest.json")):
        root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for seed in manifest.get("config", {}).get("seeds", []):
            for phase in ["pretrain", *manifest["config"]["methods"]]:
                directory = root / f"seed_{seed:02d}" / phase
                index, history, result = directory / "checkpoint_index.json", directory / "history.json", directory / "result.json"
                if not any(p.exists() for p in [index, history, result, directory / "checkpoint.pt"]):
                    continue
                record = {"folder": root.name, "seed": seed, "phase": phase, "complete_result": result.exists(), "saved_epochs": 0, "next_batch": 0}
                if index.exists():
                    current = json.loads(index.read_text(encoding="utf-8"))["current"]
                    record.update(saved_epochs=current["next_epoch"], next_batch=current["next_batch"], format="portable_v2")
                elif history.exists():
                    record.update(saved_epochs=len(json.loads(history.read_text(encoding="utf-8"))), format="legacy_v1")
                rows.append(record)
    return rows


def export_review(output_root, roots=None):
    import zipfile
    output_root = Path(output_root)
    roots = [Path(p) for p in roots] if roots is not None else [p.parent for p in output_root.glob("*/manifest.json")]
    path = output_root / "results_for_review.zip"
    temp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for shared in ("PROTOCOL_portable_v2.md", "canonical_hashes.json", "verified_dataset.json", "runtime_validation_latest.json", "latest_completed_roots.json", "scale_comparison_paired_seeds.csv"):
            file = output_root / shared
            if file.exists():
                archive.write(file, file.name)
        for root in roots:
            if not root.resolve().is_relative_to(output_root.resolve()):
                raise ValueError("Review archive only includes runs inside OUTPUT_ROOT")
            for file in root.rglob("*"):
                if file.is_file() and file.suffix in {".json", ".csv", ".png", ".py"}:
                    archive.write(file, file.relative_to(output_root))
    os.replace(temp, path)
    return path


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


def technical_smoke(device="cpu", batch_size=16):
    """Small numerical/architecture check, explicitly NOT experimental evidence."""
    seed_all(987654)
    torch.set_num_threads(2)
    device = torch.device(device)
    for backbone in ("author_conv_bn", "legacy_conv"):
        ae = Autoencoder(backbone=backbone).to(device)
        x = torch.rand(batch_size, 1, 28, 28, device=device)
        rec, z = ae(x)
        assert rec.shape == x.shape and z.shape == (batch_size, 10)
        head = RBFHead(z[:10].detach().cpu(), z.detach().cpu(), METHODS["ada_im"]).to(device)
        q, target = head(z)
        loss, _ = loss_terms(x, rec, z, q, head.method, 99, 100)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in list(ae.parameters()) + list(head.parameters()))
        head.after_step(target)
    print("Technical smoke passed on", device, "(not a research result).")


def portability_smoke(device):
    """Exercise real optimizer migration on the current runtime before long work."""
    import tempfile
    device = torch.device(device)
    technical_smoke(str(device), batch_size=256 if device.type == "cuda" else 16)
    cfg = replace(Config(), require_cuda=False, batch_size=16, pretrain_epochs=1, cluster_epochs=1)
    seed_all(987654)
    images = torch.randint(0, 256, (48, 1, 28, 28), dtype=torch.uint8)
    model = Autoencoder().to(device)
    z, _ = infer(model, images, device, 16)
    centers = z[:10]
    with tempfile.TemporaryDirectory(prefix="dcss_portability_") as folder:
        head = RBFHead(centers, z, METHODS["ada_im"]).to(device)
        train_phase(model, images, cfg, 987654, device, folder, "technical_only", head, stop_after_batches=1)
        model = Autoencoder()
        head = RBFHead(centers, z, METHODS["ada_im"])
        train_phase(model, images, cfg, 987654, torch.device("cpu"), folder, "technical_only", head, stop_after_batches=1)
        model = Autoencoder().to(device)
        head = RBFHead(centers, z, METHODS["ada_im"]).to(device)
        train_phase(model, images, cfg, 987654, device, folder, "technical_only", head)
        final = load_training_checkpoint(folder, "technical_only")
        assert final["next_epoch"] == 1 and final["next_batch"] == 0
        assert all(value.device.type == "cpu" for value in final["model"].values())
    return {"type": "technical_only_not_research", "environment": environment(device), "at": utc_now(),
            "batch256_gpu_check": device.type == "cuda", "optimizer_roundtrip": str(device) + " -> cpu -> " + str(device),
            "passed": True}


if __name__ == "__main__":
    technical_smoke()
