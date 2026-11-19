"""End-to-end CPU integration on real images. Never a thesis benchmark."""
import json
from dataclasses import replace
from pathlib import Path
import torch
import experiment as e

root = Path(__file__).resolve().parent
hashes = json.loads((root / "canonical_hashes.json").read_text())
x, y, meta = e.load_data(root / "data", hashes)
cfg = replace(e.Config(), profile="TECHNICAL_ONLY_256_IMAGES_CPU", seeds=(0, 1), require_cuda=False,
              pretrain_epochs=2, cluster_epochs=2, batch_size=256, kmeans_n_init=2)
meta = {**meta, "n": 256, "split": "TECHNICAL_ONLY_first256", "not_research_results": True}
out = e.run_experiment(x[:256], y[:256], meta, cfg, root / "test_runs" / "portable_v2_validation")
before = {str(p): e.digest(p) for p in out.glob("seed_*/*/result.json")}
e.run_experiment(x[:256], y[:256], meta, cfg, root / "test_runs" / "portable_v2_validation")
after = {str(p): e.digest(p) for p in out.glob("seed_*/*/result.json")}
assert before == after and len(after) == 8
e.plots(out)
e.atomic_json(root / "audit" / "local_validation_portable.json", {
    "type": "technical_validation_not_research_evidence", "device": "cpu", "real_images": 256,
    "seeds": 2, "methods": 4, "pretrain_epochs": 2, "cluster_epochs": 2,
    "all_8_runs_finished": True, "rerun_preserves_results": True, "environment": e.environment(torch.device("cpu")),
    "experiment_sha256": e.digest(e.__file__), "data_equals_official": True,
})
print("Integration passed. Scores under test_runs are technical only and must not enter the thesis.")
