"""Read-only completion/statistics audit of the 2026-11-15 Drive snapshot."""
from pathlib import Path
import json
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import dcss_core as core
from multi_analysis import holm

BASE = Path(__file__).parent
SNAPSHOT = BASE / "audit/results_2026_11_15"
REVIEW = SNAPSHOT / "review"
EXPECTED = {"emnist": (131600, 47), "har": (10299, 6), "pendigits": (10992, 10), "waveform": (5000, 3), "personas": (100000, 20)}
METHODS = ("ae_kmeans", "dcss_paper", "dcss_rms", "im_rms", "ada_im")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    findings, failures, primary = [], [], []
    maximum_ci_roundtrip_difference = 0.
    history_rows = record_count = 0
    def check(ok, message):
        if not ok:
            failures.append(message)
    reference = read(BASE / "audit/data_verification.json")["datasets"]
    saved_primary = read(REVIEW / "analysis/primary_comparisons_holm.json")
    for key, (n, k) in EXPECTED.items():
        roots = list(REVIEW.glob(key + "_*/manifest.json"))
        check(len(roots) == 1, key + ": unique run")
        root = roots[0].parent
        manifest = read(roots[0])
        cfg, meta, identity = manifest["config"], manifest["data"], manifest["identity"]
        check(core.hash_json(manifest["scientific_spec"]) == identity, key + ": scientific identity")
        check(cfg["seeds"] == list(range(10)) and cfg["methods"] == list(METHODS[1:]), key + ": seeds/methods")
        check(cfg["pretrain_epochs"] == cfg["cluster_epochs"] == 100 and cfg["batch_size"] == 256, key + ": locked epochs/batch")
        check(meta["n"] == n and cfg["n_clusters"] == k, key + ": N/K")
        if key != "personas":
            for name in ("features_sha256", "labels_sha256"):
                check(meta[name] == reference[key][name], key + ": " + name)
        else:
            for name in ("global_indices_sha256", "identifiers_sha256"):
                check(meta["sample"][name] == reference[key][name], key + ": sample " + name)
        for filename in ("completeness.json", "status.json"):
            check(read(root / filename) == read(SNAPSHOT / "live" / root.name / filename), key + ": live " + filename)
        done = read(root / "completeness.json")
        check(done["complete"] and done["completed"] == done["expected"] == 50 and not done["missing"], key + ": completion")
        check(read(root / "status.json")["state"] == "complete", key + ": final status")
        rows = []
        for seed in range(10):
            seed_root = root / f"seed_{seed:02d}"
            for phase in ("pretrain", *METHODS[1:]):
                h = read(seed_root / phase / "history.json")
                check(len(h) == 100 and [r["epoch"] for r in h] == list(range(1, 101)), f"{key}/{seed}/{phase}: 100 epochs")
                check(all(r["seen"] == n and math.isfinite(r["loss"]) for r in h), f"{key}/{seed}/{phase}: full epochs/finite loss")
                pointer = read(seed_root / phase / "checkpoint_index.json")
                check(pointer["current"]["next_epoch"] == 100 and pointer["current"]["next_batch"] == 0, f"{key}/{seed}/{phase}: checkpoint end")
                check(pointer["identity"] == identity + f":{seed}:{phase}", f"{key}/{seed}/{phase}: checkpoint identity")
                history_rows += len(h)
            for method in METHODS:
                r = read(seed_root / method / "result.json")
                check(r["seed"] == seed and r["method"] == method and r["profile"] == key, f"{key}/{seed}/{method}: key")
                check(r["identity"] == identity + f":{seed}:{method}" and r["status"] == "complete", f"{key}/{seed}/{method}: identity")
                check(r["n"] == n and sum(r["cluster_counts"]) == n, f"{key}/{seed}/{method}: counts")
                check(len(r["predictions_sha256"]) == 64, f"{key}/{seed}/{method}: prediction commit marker")
                check(r["hard_active_clusters"] == sum(c > 0 for c in r["cluster_counts"]), f"{key}/{seed}/{method}: active count")
                check(all(r[m] is None for m in ("ARI", "NMI", "ACC")) if key == "personas" else
                      all(math.isfinite(r[m]) for m in ("ARI", "NMI", "ACC")), f"{key}/{seed}/{method}: label metrics")
                rows.append(r)
        check(len(list(root.glob("seed_*/*/result.json"))) == len(rows) == 50, key + ": 50 unique result files")
        check(len({r["common_sample_sha256"] for r in rows}) == 1, key + ": same evaluation sample")
        record_count += len(rows)
        summary = pd.read_csv(root / "summary.csv").set_index("method")
        metrics = ("silhouette_common", "DB_common", "CH_common") + (() if key == "personas" else ("ARI", "NMI", "ACC"))
        for method in METHODS:
            for metric in metrics:
                v = np.array([r[metric] for r in rows if r["method"] == method and r[metric] is not None], dtype=float)
                check(len(v) == summary.loc[method, metric + "_valid_seeds"], f"{key}/{method}/{metric}: valid seeds")
                for suffix, expected in [("_mean", v.mean() if len(v) else None), ("_sample_sd", v.std(ddof=1) if len(v) > 1 else None)]:
                    if expected is not None:
                        check(np.isclose(expected, summary.loc[method, metric + suffix], atol=1e-12, rtol=0), f"{key}/{method}/{metric}: {suffix}")
        metric = cfg["primary_metric"]
        a = {r["seed"]: r[metric] for r in rows if r["method"] == "ada_im"}
        b = {r["seed"]: r[metric] for r in rows if r["method"] == "im_rms"}
        stats = core.paired_statistics([a[s] - b[s] for s in range(10)])
        published = next(p for p in saved_primary if p["dataset"] == key)
        for name, value in stats.items():
            # The t quantile is recomputed on a different SciPy build. Retain
            # the measured difference and use a specified numerical tolerance.
            ci = name in {"ci95_low", "ci95_high"}
            if ci:
                maximum_ci_roundtrip_difference = max(maximum_ci_roundtrip_difference, abs(value-published[name]))
            check(np.isclose(value, published[name], atol=1e-10 if ci else 1e-12, rtol=0), key + ": paired " + name)
        primary.append(dict(dataset=key, metric=metric, **stats))
        sessions = [read(p) for p in (root / "sessions").glob("*.json")]
        finished = [s["ended_at"] for s in sessions if s.get("ended_at") and s.get("state") == "complete"]
        first = min(finished)
        methods = []
        for method in METHODS:
            g = [r for r in rows if r["method"] == method]
            methods.append(dict(method=method, completed=len(g), flagged_collapsed=sum(bool(r.get("collapsed")) for r in g),
                single_cluster_runs=sum(r["hard_active_clusters"] == 1 for r in g),
                minimum_active_clusters=min(r["hard_active_clusters"] for r in g),
                maximum_active_clusters=max(r["hard_active_clusters"] for r in g),
                undefined_internal_metrics=sum(r["silhouette_common"] is None for r in g)))
        findings.append(dict(dataset=key, n=n, k=k, results=50, complete=True, first_complete=first, last_complete=max(finished),
            first_complete_vietnam=datetime.fromisoformat(first).astimezone(timezone(timedelta(hours=7))).isoformat(),
            campaign_mixed_environment=done["campaign_mixed_environment"], methods=methods))
    for p, adj in zip(primary, holm([p["sign_flip_p_two_sided"] for p in primary])):
        p["holm_p_family5"] = adj
        check(np.isclose(adj, next(r for r in saved_primary if r["dataset"] == p["dataset"])["holm_p_family5"], atol=1e-12, rtol=0), p["dataset"] + ": Holm")
    notebook = read(SNAPSHOT / "colab_run.ipynb")
    errors = [o for c in notebook["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check(not errors, "No stored notebook errors")
    report = dict(audit_date="2026-11-15", complete=not failures, checked_results=record_count,
        checked_phase_histories=history_rows // 100, checked_epoch_records=history_rows,
        live_status_files_match_archive=True, notebook_error_count=len(errors), issues=failures,
        ci_absolute_tolerance=1e-10, maximum_ci_roundtrip_difference=maximum_ci_roundtrip_difference,
        datasets=findings, primary_comparisons=primary,
        scope="Verified persisted results, histories, checkpoint indices, input fingerprints, summaries and live completion/status. Model tensors and all predictions.npz were not downloaded or independently recomputed.",
        archive_sha256=core.digest(SNAPSHOT / "results_for_review.zip"), notebook_sha256=core.digest(SNAPSHOT / "colab_run.ipynb"))
    core.atomic_json(SNAPSHOT / "completion_audit.json", report)
    pd.DataFrame([{k: v for k, v in d.items() if k != "methods"} for d in findings]).to_csv(SNAPSHOT / "completion_audit.csv", index=False)
    pd.DataFrame([dict(dataset=d["dataset"], **m) for d in findings for m in d["methods"]]).to_csv(SNAPSHOT / "cluster_health_audit.csv", index=False)
    print(json.dumps({k:v for k,v in report.items() if k not in {"datasets","primary_comparisons"}}, indent=2, ensure_ascii=False))
    if failures:
        raise AssertionError(f"Audit has {len(failures)} issues")


if __name__ == "__main__":
    main()
