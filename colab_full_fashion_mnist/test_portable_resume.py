"""Restart/migration regression tests. CUDA round-trip runs only with real CUDA."""
import copy
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import experiment as e


def load_v1():
    path = Path(__file__).parent / "audit" / "v1" / "experiment.py"
    spec = importlib.util.spec_from_file_location("verified_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def small_config():
    return replace(e.Config(), profile="portable_test", require_cuda=False, seeds=(0,),
                   methods=("ada_im",), batch_size=8, pretrain_epochs=2, cluster_epochs=2, kmeans_n_init=2)


def fixture_data():
    e.seed_all(15)
    x = torch.randint(0, 256, (24, 1, 28, 28), dtype=torch.uint8)
    y = np.arange(24) % 10
    return x, y, {"n": len(x), "split": "technical_only", "sha256": "synthetic_fixture"}


def test_identity_portable_but_scientific_changes_rejected():
    cfg = e.Config()
    data = {"n": 70000, "sha256": "abc"}
    envs = [{"gpu": "Tesla T4"}, {"gpu": "A100", "python": "3.12"}, {"device": "cpu", "packages": {"torch": "changed"}}]
    ids = [e.make_identity(cfg, data, env)[0] for env in envs]
    assert len(set(ids)) == 1
    assert ids[0] == e.make_identity(replace(cfg, require_cuda=False, cpu_threads=4), data, {})[0]
    for modified in [replace(cfg, batch_size=512), replace(cfg, cluster_epochs=50), replace(cfg, backbone="legacy_conv")]:
        assert ids[0] != e.make_identity(modified, data, {})[0]
    assert ids[0] != e.make_identity(cfg, {**data, "sha256": "changed"}, {})[0]


def test_v1_math_unchanged():
    path = Path(__file__).parent / "audit" / "v1" / "experiment.py"
    assert e.digest(path) == e.LEGACY_CODE_SHA256
    assert e.mathematical_code_hash() == e.mathematical_code_hash(path.read_text(encoding="utf-8"))
    changed = Path(e.__file__).read_text(encoding="utf-8").replace("betas=(.9, .999)", "betas=(.8, .999)")
    assert e.mathematical_code_hash(changed) != e.mathematical_code_hash()


def test_auto_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert e.select_device(e.RuntimeOptions()).type == "cpu"
    with pytest.raises(RuntimeError):
        e.select_device(e.RuntimeOptions(allow_cpu=False))
    with pytest.raises(ValueError):
        e.select_device(e.RuntimeOptions(device="tpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert e.select_device(e.RuntimeOptions()).type == "cuda"
    assert e.select_device(e.RuntimeOptions(device="cpu")).type == "cpu"


@pytest.mark.parametrize("cluster", [False, True])
@pytest.mark.parametrize("stop_batch", [1, 3])
def test_mid_epoch_exact_restart(tmp_path, cluster, stop_batch):
    cfg = small_config()
    x, _, _ = fixture_data()
    model = e.Autoencoder(backbone=cfg.backbone)
    initial = copy.deepcopy(model.state_dict())
    z, _ = e.infer(model, x, torch.device("cpu"), 8)
    head = e.RBFHead(z[:10], z, e.METHODS["ada_im"]) if cluster else None
    initial_h = copy.deepcopy(head.state_dict()) if head else None
    e.train_phase(model, x, cfg, 0, torch.device("cpu"), tmp_path / "full", "id", head)
    expected = copy.deepcopy(model.state_dict())
    expected_h = copy.deepcopy(head.state_dict()) if head else None
    model.load_state_dict(initial)
    if head:
        head.load_state_dict(initial_h)
    e.train_phase(model, x, cfg, 0, torch.device("cpu"), tmp_path / "partial", "id", head, stop_after_batches=stop_batch)
    saved = e.load_training_checkpoint(tmp_path / "partial", "id")
    assert saved["next_epoch"] == 0 and saved["next_batch"] == stop_batch
    assert saved["partial_seen"] == stop_batch * 8
    e.seed_all(123)
    model = e.Autoencoder(backbone=cfg.backbone)
    head = e.RBFHead(z[:10], z, e.METHODS["ada_im"]) if cluster else None
    e.train_phase(model, x, cfg, 0, torch.device("cpu"), tmp_path / "partial", "id", head)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    if head:
        assert all(torch.equal(v, head.state_dict()[k]) for k, v in expected_h.items())
    assert e.load_training_checkpoint(tmp_path / "partial", "id")["next_epoch"] == 2


def test_two_slot_recovery_and_interrupted_write(tmp_path, monkeypatch):
    def state(epoch):
        return {"identity": "same", "next_epoch": epoch, "tensor": torch.tensor([float(epoch)])}
    e.save_training_checkpoint(tmp_path, state(1))
    e.save_training_checkpoint(tmp_path, state(2))
    index = json.loads((tmp_path / "checkpoint_index.json").read_text())
    (tmp_path / index["current"]["file"]).write_bytes(b"interrupted disk write")
    with pytest.warns(UserWarning, match="recovered previous"):
        recovered = e.load_training_checkpoint(tmp_path, "same")
    assert recovered["next_epoch"] == 1
    e.save_training_checkpoint(tmp_path, state(3))
    assert e.load_training_checkpoint(tmp_path, "same")["next_epoch"] == 3
    index_before = (tmp_path / "checkpoint_index.json").read_bytes()
    def fail_write(path, obj):
        path.with_suffix(".tmp").write_bytes(b"partial")
        raise OSError("simulated session killed during write")
    monkeypatch.setattr(e, "atomic_torch", fail_write)
    with pytest.raises(OSError):
        e.save_training_checkpoint(tmp_path, state(4))
    assert (tmp_path / "checkpoint_index.json").read_bytes() == index_before
    assert e.load_training_checkpoint(tmp_path, "same")["next_epoch"] == 3


def test_time_budget_saves_and_pauses(tmp_path):
    cfg = small_config()
    x, _, _ = fixture_data()
    model = e.Autoencoder()
    session = {"session_id": "test", "environment": e.environment(torch.device("cpu")), "deadline": time.monotonic() - 1}
    with pytest.raises(e.PauseRequested):
        e.train_phase(model, x, cfg, 0, torch.device("cpu"), tmp_path, "id", session=session)
    state = e.load_training_checkpoint(tmp_path, "id")
    assert state["next_epoch"] == 0 and state["next_batch"] == 0


def test_v1_full_import_preserves_files_and_scores(tmp_path):
    old = load_v1()
    cfg = small_config()
    x, y, meta = fixture_data()
    old_cfg = old.Config(**asdict(cfg))
    # Produce a real v1 CPU checkpoint with its original implementation.
    root = old.run_experiment(x, y, meta, old_cfg, tmp_path)
    original = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # Runtime identity is deliberately different, but no model training repeats.
    fake_environment = {**e.environment(torch.device("cpu")), "gpu": "A100-label-for-discovery-test-only"}
    selected, identity, manifest = e.resolve_run(tmp_path, cfg, meta, fake_environment, e.RuntimeOptions(device="cpu"))
    assert selected == root
    resumed = e.run_experiment(x, y, meta, cfg, tmp_path, e.RuntimeOptions(device="cpu"))
    assert resumed == root
    for relative, data in original.items():
        if relative.endswith("checkpoint.pt") or relative.endswith("result.json") or relative in {"manifest.json", "experiment.py"}:
            assert (root / relative).read_bytes() == data
    assert json.loads((root / "portable_resume.json").read_text())["migrated_from_v1"]


def test_v1_partial_checkpoint_continues_exactly(tmp_path):
    old = load_v1()
    cfg = small_config()
    x, _, meta = fixture_data()
    env = old.environment(torch.device("cpu"))
    identity, manifest = old.make_identity(old.Config(**asdict(cfg)), meta, env)
    root = tmp_path / (cfg.profile + "_" + identity[:12])
    e.atomic_json(root / "manifest.json", manifest)
    shutil.copyfile(old.__file__, root / "experiment.py")
    model = old.Autoencoder()
    initial = copy.deepcopy(model.state_dict())
    directory = root / "seed_00" / "pretrain"
    old.train_phase(model, x, old.Config(**asdict(cfg)), 0, torch.device("cpu"), directory, identity + ":0:pretrain", stop_after=1)
    checkpoint_before = (directory / "checkpoint.pt").read_bytes()
    # Independent uninterrupted v1 reference.
    reference = old.Autoencoder()
    reference.load_state_dict(initial)
    old.train_phase(reference, x, old.Config(**asdict(cfg)), 0, torch.device("cpu"), tmp_path / "reference", "ref")
    selected, actual_id, _ = e.resolve_run(tmp_path, cfg, meta, env, e.RuntimeOptions(device="cpu"))
    assert selected == root and actual_id == identity
    fresh = e.Autoencoder()
    e.train_phase(fresh, x, cfg, 0, torch.device("cpu"), directory, identity + ":0:pretrain")
    assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in reference.state_dict().items())
    assert checkpoint_before == (directory / "checkpoint.pt").read_bytes()


def test_multiple_candidates_never_choose_by_score(tmp_path):
    cfg = small_config()
    _, _, meta = fixture_data()
    env = e.environment(torch.device("cpu"))
    root, _, _ = e.resolve_run(tmp_path, cfg, meta, env, e.RuntimeOptions())
    other = tmp_path / (cfg.profile + "_second")
    shutil.copytree(root, other)
    with pytest.raises(e.ResumeCompatibilityError, match="Several compatible"):
        e.resolve_run(tmp_path, cfg, meta, env, e.RuntimeOptions())
    selected, _, _ = e.resolve_run(tmp_path, cfg, meta, env, e.RuntimeOptions(resume_folder=root.name))
    assert selected == root


def test_changed_config_never_silently_restarts(tmp_path):
    cfg = small_config()
    _, _, meta = fixture_data()
    env = e.environment(torch.device("cpu"))
    root, identity, _ = e.resolve_run(tmp_path, cfg, meta, env, e.RuntimeOptions())
    e.atomic_torch(root / "seed_00" / "pretrain" / "checkpoint.pt", {"identity": identity})
    with pytest.raises(e.ResumeCompatibilityError, match="incompatible"):
        e.resolve_run(tmp_path, replace(cfg, batch_size=16), meta, env, e.RuntimeOptions())


def test_environment_change_logged_and_strict_mode(tmp_path):
    cfg = small_config()
    _, _, meta = fixture_data()
    env = e.environment(torch.device("cpu"))
    old_env = {**env, "gpu": "Tesla T4", "device": "cuda", "python": "3.11.0"}
    root, _, manifest = e.resolve_run(tmp_path, cfg, meta, old_env, e.RuntimeOptions())
    with pytest.raises(e.ResumeCompatibilityError, match="Software changed"):
        e.start_session(root, manifest, torch.device("cpu"), e.RuntimeOptions(allow_software_change=False))
    session = e.start_session(root, manifest, torch.device("cpu"), e.RuntimeOptions())
    assert session["software_changed"] and session["hardware_changed"]
    assert (root / "sessions" / (session["session_id"] + ".json")).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA GPU unavailable; not claimed as verified on A100/T4")
def test_real_cuda_cpu_cuda_roundtrip(tmp_path):
    cfg = small_config()
    x, _, _ = fixture_data()
    model = e.Autoencoder().cuda()
    e.train_phase(model, x, cfg, 0, torch.device("cuda:0"), tmp_path, "id", stop_after_batches=1)
    model = e.Autoencoder()
    e.train_phase(model, x, cfg, 0, torch.device("cpu"), tmp_path, "id", stop_after_batches=1)
    model = e.Autoencoder().cuda()
    e.train_phase(model, x, cfg, 0, torch.device("cuda:0"), tmp_path, "id")
    assert e.load_training_checkpoint(tmp_path, "id")["next_epoch"] == cfg.pretrain_epochs
