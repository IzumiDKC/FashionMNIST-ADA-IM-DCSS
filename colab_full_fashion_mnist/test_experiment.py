"""Numerical and restart checks. These are not experimental quality scores."""
import copy
import importlib.util
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import experiment as e

torch.set_num_threads(2)


def brute_soft_silhouette(z, q):
    z, q = z.detach().numpy(), q.detach().numpy()
    result = []
    for i in range(len(z)):
        a = []
        for k in range(q.shape[1]):
            numerator = sum(q[j, k] * np.linalg.norm(z[i] - z[j]) for j in range(len(z)) if j != i)
            denominator = sum(q[j, k] for j in range(len(z)) if j != i)
            a.append(numerator / max(denominator, 1e-8))
        s = []
        for k in range(q.shape[1]):
            b = min(a[j] for j in range(q.shape[1]) if j != k)
            s.append((b - a[k]) / max(b, a[k], 1e-8))
        result.append(np.dot(q[i], s))
    return np.mean(result)


def test_equations_13_to_17_and_gradient():
    e.seed_all(17)
    z = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    logits = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
    q = logits.softmax(1)
    observed = e.soft_silhouette(z, q)
    assert observed.item() == pytest.approx(brute_soft_silhouette(z, q), abs=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: e.soft_silhouette(a, b.softmax(1)), (z, logits))
    gz, gl = torch.autograd.grad(observed, (z, logits))
    assert gz.abs().sum() > 0 and gl.abs().sum() > 0
    assert -1 <= observed <= 1


def test_degenerate_distances_and_empty_cluster_are_finite():
    z = torch.zeros(5, 3, requires_grad=True)
    q = torch.tensor([[1., 0., 0.]] * 5, requires_grad=True)
    loss = e.soft_silhouette(z, q)
    loss.backward()
    assert loss == 0 and torch.isfinite(z.grad).all() and torch.isfinite(q.grad).all()
    with pytest.raises(ValueError):
        e.soft_silhouette(z[:1], q[:1])


def test_entropy_signs_and_shot_identity():
    confident_balanced = torch.eye(4)
    collapsed = torch.tensor([[1., 0., 0., 0.]] * 4)
    uniform = torch.ones(4, 4) / 4
    h1, kl1 = e.entropy_terms(confident_balanced)
    h2, kl2 = e.entropy_terms(collapsed)
    h3, kl3 = e.entropy_terms(uniform)
    assert h1 == 0 and kl1 == 0 and h2 == 0
    assert float(kl2) == pytest.approx(math.log(4))
    assert float(h3) == pytest.approx(math.log(4)) and kl3 == 0
    q = torch.softmax(torch.randn(20, 4), 1)
    h, kl = e.entropy_terms(q)
    hm = -(q.mean(0) * q.mean(0).log()).sum()
    assert float(h + kl - math.log(4)) == pytest.approx(float(h - hm), abs=2e-7)
    assert e.warmup_weight(0, 100, .70) == 0
    assert e.warmup_weight(69, 100, .70) == 0
    assert e.warmup_weight(99, 100, .70) == 1


@pytest.mark.parametrize("method_name", ["dcss_paper", "dcss_rms", "im_rms", "ada_im"])
def test_rbf_gradients_and_eval_does_not_change_state(method_name):
    e.seed_all(1)
    z = torch.randn(25, 10, requires_grad=True)
    head = e.RBFHead(z[:10].detach(), z.detach(), e.METHODS[method_name])
    q, target = head(z)
    assert torch.allclose(q.sum(1), torch.ones(25))
    loss = e.soft_silhouette(z, q) + e.entropy_terms(q)[0]
    loss.backward()
    assert head.centers.grad.abs().sum() > 0 and torch.isfinite(z.grad).all()
    if head.log_sigma is not None:
        assert head.log_sigma.grad.abs().sum() > 0
    head.after_step(target)
    head.eval()
    previous = copy.deepcopy(head.state_dict())
    for _ in range(3):
        head(z)
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in previous.items())


def test_ada_scale_equivariance_and_guard():
    z = torch.randn(30, 10)
    h = e.RBFHead(z[:10], z, e.METHODS["ada_im"])
    d2 = e.squared_distances(z, h.centers)
    q = torch.softmax(torch.randn(30, 10), 1)
    assert torch.allclose(h.target_sigma(d2 * 9, q), 3 * h.target_sigma(d2, q), atol=1e-5)
    h.sigma.fill_(1e-8)
    q, target = h(z)
    assert int(h.guard_count) == 1 and torch.isfinite(q).all()
    h.after_step(target)
    assert (h.sigma > 0).all()


def test_metrics_and_degenerate_ari():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([2, 2, 0, 0, 1, 1])
    assert e.metrics(y, p) == {"ACC": 1., "ARI": 1., "NMI": 1.}
    p = np.array([0, 1, 1, 1, 0, 2])
    result = e.metrics(y, p)
    assert result["ARI"] == adjusted_rand_score(y, p)
    assert result["NMI"] == normalized_mutual_info_score(y, p)
    assert e.metrics(np.zeros(5), np.zeros(5))["ARI"] == 1.


def test_idx_strictness(tmp_path):
    path = tmp_path / "x.idx"
    path.write_bytes(struct.pack(">IIII", 2051, 2, 28, 28) + bytes(2 * 784))
    assert e.read_idx(path).shape == (2, 1, 28, 28)
    for raw in [b"oops", struct.pack(">IIII", 2051, 2, 28, 28) + bytes(784), struct.pack(">II", 2049, 1) + bytes([99])]:
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            e.read_idx(path)


def test_batches_cover_every_sample_once_and_pair_methods():
    for n in [257, 513, 70000]:
        a = e.epoch_batches(n, 256, 7, 1, 15)
        b = e.epoch_batches(n, 256, 7, 1, 15)
        assert torch.equal(torch.cat(a).sort().values, torch.arange(n))
        assert all(len(x) >= 2 for x in a)
        assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("cluster", [False, True])
@pytest.mark.parametrize("backbone", ["legacy_conv", "author_conv_bn"])
def test_epoch_boundary_resume_exact(tmp_path, cluster, backbone):
    cfg = replace(e.Config(), require_cuda=False, pretrain_epochs=3, cluster_epochs=3, batch_size=8, backbone=backbone)
    e.seed_all(37)
    x = torch.randint(0, 256, (24, 1, 28, 28), dtype=torch.uint8)
    model = e.Autoencoder(backbone=cfg.backbone)
    init = copy.deepcopy(model.state_dict())
    z, _ = e.infer(model, x, torch.device("cpu"), 8)
    h = e.RBFHead(z[:10], z, e.METHODS["ada_im"]) if cluster else None
    h0 = copy.deepcopy(h.state_dict()) if h else None
    e.train_phase(model, x, cfg, 37, torch.device("cpu"), tmp_path / "continuous", "same", h)
    expected = copy.deepcopy(model.state_dict())
    expected_h = copy.deepcopy(h.state_dict()) if h else None
    model.load_state_dict(init)
    if h:
        h.load_state_dict(h0)
    e.train_phase(model, x, cfg, 37, torch.device("cpu"), tmp_path / "resumed", "same", h, stop_after=1)
    # Simulate a new runtime with arbitrary initial parameters and RNG state.
    e.seed_all(999)
    model = e.Autoencoder(backbone=cfg.backbone)
    h = e.RBFHead(z[:10], z, e.METHODS["ada_im"]) if cluster else None
    e.train_phase(model, x, cfg, 37, torch.device("cpu"), tmp_path / "resumed", "same", h)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    if h:
        assert all(torch.equal(v, h.state_dict()[k]) for k, v in expected_h.items())
    with pytest.raises(ValueError):
        e.load_training_checkpoint(tmp_path / "resumed", "DIFFERENT_DATA")


def test_paired_statistics_and_incomplete_reporting(tmp_path):
    result = e.paired_statistics([.01] * 10)
    assert result["mean_delta"] == pytest.approx(.01)
    assert result["sign_flip_p_two_sided"] == pytest.approx(2 / 1024)
    assert e.paired_statistics([.01])["ci95_low"] is None
    cfg = e.Config()
    for method in cfg.methods:
        e.atomic_json(tmp_path / "seed_00" / method / "result.json", {"seed": 0, "method": method, "ARI": .4, "NMI": .5, "ACC": .6, "collapsed": False})
    e.summarize(tmp_path, cfg)
    completeness = json.loads((tmp_path / "completeness.json").read_text())
    assert not completeness["complete"] and completeness["completed"] == 4 and completeness["expected"] == 40
    pairs = json.loads((tmp_path / "paired_comparisons.json").read_text())
    assert not any(x["complete"] for x in pairs)


def test_full_verified_dataset():
    root = Path(__file__).resolve().parent
    if not (root / "data" / "train-images-idx3-ubyte").exists():
        pytest.skip("Full data test is local audit only; notebook re-verifies before training")
    hashes = json.loads((root / "canonical_hashes.json").read_text())
    x, y, meta = e.load_data(root / "data", hashes)
    assert x.shape == (70000, 1, 28, 28) and x.dtype == torch.uint8
    assert np.array_equal(np.bincount(y), [7000] * 10)
    assert meta["split"] == "all70k"


def test_optional_author_objective_matches_pinned_source():
    path = Path(__file__).resolve().parent / "audit" / "upstream" / "Objectives.py"
    if not path.exists():
        pytest.skip("Upstream archive only in local audit")
    spec = importlib.util.spec_from_file_location("pinned_objectives", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    z, q = torch.randn(30, 10, dtype=torch.double), torch.softmax(torch.randn(30, 10, dtype=torch.double), 1)
    upstream = mod.Objectives(torch.device("cpu")).soft_silhouette(z, q, requires_distance_grad=True)
    ours = e.soft_silhouette(z, q, "author_unnormalized")
    assert float(ours) == pytest.approx(float(upstream), abs=1e-8)
    assert not torch.allclose(ours, e.soft_silhouette(z, q))


def test_both_backbones_smoke():
    e.technical_smoke()
