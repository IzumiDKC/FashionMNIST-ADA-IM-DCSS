"""Read-only analysis of completed DCSS experiments. No torch, training or pickle.

Recompute scores from predictions.npz; export new artifacts to a separate folder.
All point estimates are fractions; *_pp columns use percentage points.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import platform
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import matplotlib.pyplot as plt

METHOD_LABELS = {"dcss_paper": "DCSS theo phương trình", "dcss_rms": "DCSS-RMS",
                 "im_rms": "IM-DCSS", "ada_im": "ADA-IM-DCSS", "ae_kmeans": "AE + K-means"}
COLORS = {"dcss_paper": "#4C78A8", "dcss_rms": "#72B7B2", "im_rms": "#F2A541", "ada_im": "#B45A87"}
CLASS_NAMES = ["Áo thun", "Quần dài", "Áo chui đầu", "Váy", "Áo khoác", "Xăng-đan", "Áo sơ mi", "Giày thể thao", "Túi", "Bốt"]
METRICS = ("ARI", "NMI", "ACC")
ANALYSIS_VERSION = "2026-11-10-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def scores_from_predictions(y, pred):
    _, yy = np.unique(y, return_inverse=True)
    _, pp = np.unique(pred, return_inverse=True)
    matrix = np.zeros((max(yy.max(), pp.max()) + 1,) * 2, dtype=np.int64)
    np.add.at(matrix, (pp, yy), 1)
    rows, cols = linear_sum_assignment(-matrix)
    return {"ARI": float(adjusted_rand_score(y, pred)),
            "NMI": float(normalized_mutual_info_score(y, pred, average_method="arithmetic")),
            "ACC": float(matrix[rows, cols].sum() / len(y))}


def paired_stats(delta):
    d = np.asarray(delta, dtype=float)
    n = len(d)
    result = {"n_pairs": n, "delta_mean_pp": float(d.mean() * 100) if n else None,
              "delta_sd_pp": None, "ci95_low_pp": None, "ci95_high_pp": None,
              "p_signflip_two_sided": None, "wins": int((d > 1e-12).sum()),
              "ties": int((np.abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum())}
    if n >= 2:
        sd = float(d.std(ddof=1))
        margin = float(stats.t.ppf(.975, n - 1) * sd / math.sqrt(n))
        result.update(delta_sd_pp=100 * sd, ci95_low_pp=100 * (float(d.mean()) - margin), ci95_high_pp=100 * (float(d.mean()) + margin))
        if n <= 16:
            signs = np.asarray(list(itertools.product([-1., 1.], repeat=n)))
            result["p_signflip_two_sided"] = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-15))
    return result


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return p
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def normalized_environment(env):
    if not env:
        return None
    # Reporting/plotting library changes alone do not imply a new training environment.
    packages = env.get("packages", {})
    return {"device": str(env.get("device", "unknown")).split(":")[0], "gpu": env.get("gpu"),
            "python": env.get("python"), "torch": packages.get("torch"), "numpy": packages.get("numpy"),
            "scipy": packages.get("scipy"), "sklearn": packages.get("scikit-learn"),
            "cuda": env.get("cuda"), "cudnn": env.get("cudnn"), "amp": env.get("amp"),
            "tf32": env.get("tf32"), "deterministic": env.get("deterministic_algorithms")}


def environment_info(record, manifest):
    trace = record.get("execution_trace")
    provenance = "execution_trace" if trace else "original_manifest_only"
    environments = [t.get("environment") for t in trace] if trace else [manifest.get("environment")]
    if record.get("evaluation_environment"):
        environments.append(record["evaluation_environment"])
    normalized = [normalized_environment(env) for env in environments if env]
    signatures = sorted({hash_object(env) for env in normalized})
    devices = sorted({env.get("gpu") or env["device"].upper() for env in normalized})
    mixed = len(signatures) > 1 or bool(record.get("mixed_environment", False))
    return {"environment_provenance": provenance, "device_history": " → ".join(devices),
            "mixed_environment": mixed, "environment_signatures": "|".join(signatures),
            "strict_single_environment": bool(trace) and not mixed and all(t.get("environment") for t in trace)}


def discover_profiles(data_root, selected=None):
    selected = selected or {}
    groups = {}
    for path in sorted(Path(data_root).glob("*/manifest.json")):
        manifest = read_json(path)
        profile = manifest.get("config", {}).get("profile")
        if profile in {"paper70k", "legacy10k", "legacy70k"}:
            groups.setdefault(profile, []).append(path.parent)
    if not groups:
        raise FileNotFoundError("Không thấy manifest của các hồ sơ trong DATA_ROOT. Chọn thư mục DCSS_FashionMNIST_Full đã tải về hoặc gắn đúng Drive.")
    roots = {}
    for profile, candidates in groups.items():
        if profile in selected:
            matches = [p for p in candidates if p.name == selected[profile]]
            if len(matches) != 1:
                raise ValueError(f"Tên thư mục đã chọn không khớp {profile}: {selected[profile]}")
            roots[profile] = matches[0]
        elif len(candidates) != 1:
            raise ValueError(f"Có nhiều lượt {profile}: {[p.name for p in candidates]}. Điền SELECTED_RUNS, không chọn theo điểm.")
        else:
            roots[profile] = candidates[0]
    return {profile: roots[profile] for profile in ("paper70k", "legacy10k", "legacy70k") if profile in roots}


def audit_and_recompute(data_root, output_dir, selected=None, verify_predictions=True, raw_data_dir=None):
    """Fail on inconsistent results; list missing runs without inventing values."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = discover_profiles(data_root, selected)
    rows, epoch_rows, inventory, issues, sources, ae_rows = [], [], [], [], [], []
    labels_by_profile = {}
    canonical = read_json(data_root / "canonical_hashes.json")
    def track(path):
        path = Path(path)
        sources.append({"path": path.relative_to(data_root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    for profile, root in roots.items():
        manifest = read_json(root / "manifest.json")
        track(root / "manifest.json")
        cfg = manifest["config"]
        identity = manifest.get("identity") if manifest.get("schema") == 2 else hash_object(manifest)
        if manifest.get("schema") == 2 and identity != hash_object(manifest["scientific_spec"]):
            raise ValueError(f"Sai mã định danh manifest: {profile}")
        if manifest["data"]["sha256"] != canonical:
            raise ValueError(f"Mã kiểm tra dữ liệu khác bộ chuẩn đã ghi: {profile}")
        if (root / "portable_resume.json").exists():
            portable = read_json(root / "portable_resume.json")
            track(root / "portable_resume.json")
            if portable["checkpoint_identity"] != identity or portable["original_manifest_sha256"] != sha256(root / "manifest.json"):
                raise ValueError(f"Ánh xạ checkpoint không khớp manifest: {profile}")
        source = root / "experiment.py"
        if source.exists():
            track(source)
            if sha256(source) != manifest["code_sha256"]:
                raise ValueError(f"Mã gốc bị đổi: {source}")
        for path in sorted((root / "sessions").glob("*.json")):
            track(path)
        expected = {(int(seed), method) for seed in cfg["seeds"] for method in cfg["methods"]}
        actual = set()
        for seed in cfg["seeds"]:
            seed_root = root / f"seed_{seed:02d}"
            ae = seed_root / "ae_kmeans_evaluation.json"
            if ae.exists():
                track(ae)
                ae_rows.append({"profile": profile, "seed": seed, "method": "ae_kmeans", **read_json(ae)})
            pretrain = seed_root / "pretrain" / "history.json"
            if pretrain.exists():
                track(pretrain)
                history = read_json(pretrain)
                if len(history) != cfg["pretrain_epochs"] or [x["epoch"] for x in history] != list(range(1, cfg["pretrain_epochs"] + 1)):
                    issues.append({"profile": profile, "seed": seed, "severity": "warning", "issue": "Tiền huấn luyện chưa đủ lịch sử epoch"})
                epoch_rows.extend({"profile": profile, "seed": seed, "method": "pretrain", **h} for h in history)
            for method in cfg["methods"]:
                directory = seed_root / method
                path = directory / "result.json"
                if not path.exists():
                    issues.append({"profile": profile, "seed": seed, "method": method, "severity": "warning", "issue": "Thiếu result.json"})
                    continue
                track(path)
                record = read_json(path)
                if record.get("status") != "complete":
                    raise ValueError(f"result.json chưa hoàn tất: {path}")
                if (record["seed"], record["method"], record["profile"], record["n"], record["epochs"], record["identity"]) != (seed, method, profile, manifest["data"]["n"], cfg["cluster_epochs"], f"{identity}:{seed}:{method}"):
                    raise ValueError(f"Kết quả không khớp cấu hình: {path}")
                history_path = directory / "history.json"
                if not history_path.exists():
                    raise ValueError(f"Thiếu lịch sử của kết quả đã hoàn tất: {history_path}")
                track(history_path)
                history = read_json(history_path)
                if len(history) != cfg["cluster_epochs"] or [h["epoch"] for h in history] != list(range(1, cfg["cluster_epochs"] + 1)):
                    raise ValueError(f"Thiếu/trùng epoch: {history_path}")
                if any(h.get("seen") != record["n"] for h in history):
                    raise ValueError(f"Số ảnh mỗi epoch không đúng: {history_path}")
                epoch_rows.extend({"profile": profile, "seed": seed, "method": method, **h} for h in history)
                recalculated = {key: float(record[key]) for key in METRICS}
                verified = False
                prediction_path = directory / "predictions.npz"
                if verify_predictions and prediction_path.exists():
                    track(prediction_path)
                    with np.load(prediction_path, allow_pickle=False) as archive:
                        y, pred = archive["labels"], archive["predictions"]
                        q = archive["probabilities"]
                        if y.shape != (record["n"],) or pred.shape != y.shape or q.shape != (len(y), cfg["n_clusters"]):
                            raise ValueError(f"Sai kích thước dự đoán: {prediction_path}")
                        if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
                            raise ValueError("Nhãn/dự đoán phải là số nguyên")
                        if y.min() < 0 or y.max() >= cfg["n_clusters"] or pred.min() < 0 or pred.max() >= cfg["n_clusters"]:
                            raise ValueError("Mã nhãn/cụm vượt phạm vi")
                        if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-5) or not np.array_equal(q.argmax(1), pred):
                            raise ValueError(f"Xác suất và dự đoán không nhất quán: {prediction_path}")
                        if profile in labels_by_profile and not np.array_equal(labels_by_profile[profile], y):
                            raise ValueError(f"Nhãn/thứ tự ảnh khác giữa các seed: {prediction_path}")
                        labels_by_profile.setdefault(profile, y.copy())
                        counts = np.bincount(pred, minlength=cfg["n_clusters"])
                        if not np.array_equal(counts, record["cluster_counts"]):
                            raise ValueError(f"Số mẫu từng cụm khác: {prediction_path}")
                        if not np.array_equal(np.bincount(y, minlength=10), manifest["data"]["class_counts_audit_only"]):
                            raise ValueError("Số lượng nhãn lớp khác manifest")
                        recalculated = scores_from_predictions(y, pred)
                        if any(abs(recalculated[key] - record[key]) > 1e-9 for key in METRICS):
                            raise ValueError(f"Điểm tính lại khác result.json: {path}: {recalculated}")
                        verified = True
                elif verify_predictions:
                    issues.append({"profile": profile, "seed": seed, "method": method, "severity": "warning", "issue": "Không có predictions.npz; chỉ đọc điểm đã lưu"})
                actual.add((seed, method))
                rows.append({"profile": profile, "run_folder": root.name, "seed": seed, "method": method,
                             "n_images": record["n"], "epochs": record["epochs"], **recalculated,
                             "predictions_verified": verified,
                             **{key: record.get(key) for key in ("collapsed", "active_clusters", "max_cluster_share", "assignment_entropy", "guard_count", "rescue_count", "finetune_seconds", "peak_gpu_gb", "sampled_hard_silhouette")},
                             **environment_info(record, manifest)})
        extra = [p for p in root.glob("seed_*/*/result.json") if (int(p.parent.parent.name.split("_")[1]), p.parent.name) not in expected]
        if extra:
            raise ValueError(f"Có kết quả ngoài cấu hình, không tự trộn: {extra}")
        inventory.append({"profile": profile, "run_folder": root.name, "n_images": manifest["data"]["n"],
                          "expected_runs": len(expected), "completed_runs": len(actual), "expected_seeds": len(cfg["seeds"]),
                          "complete": actual == expected, "missing_pairs": sorted(expected - actual),
                          "pretrain_epochs": cfg["pretrain_epochs"], "cluster_epochs": cfg["cluster_epochs"], "backbone": cfg["backbone"]})
    for profile in {"paper70k", "legacy70k"} & labels_by_profile.keys():
        if "legacy10k" in labels_by_profile and not np.array_equal(labels_by_profile[profile][-10000:], labels_by_profile["legacy10k"]):
            raise ValueError("Thứ tự nhãn test 10k khác phần cuối của bộ 70k")
    raw_labels_verified = False
    if raw_data_dir is not None and Path(raw_data_dir).is_dir():
        raw_data_dir = Path(raw_data_dir)
        values = []
        for name in ("train-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"):
            path = raw_data_dir / name
            if sha256(path) != canonical[name]:
                raise ValueError("Tệp nhãn gốc sai SHA256: " + name)
            data = path.read_bytes()
            magic, n = struct.unpack(">II", data[:8])
            if magic != 2049 or len(data) != n + 8:
                raise ValueError("Header nhãn IDX không đúng")
            values.append(np.frombuffer(data, np.uint8, offset=8))
        for profile, y in labels_by_profile.items():
            expected = values[1] if profile == "legacy10k" else np.concatenate(values)
            if not np.array_equal(y, expected):
                raise ValueError("Nhãn dự đoán không khớp nhãn Fashion-MNIST chính thức")
        raw_labels_verified = True
    frame, epochs = pd.DataFrame(rows), pd.DataFrame(epoch_rows)
    if frame.empty:
        raise ValueError("Chưa có lượt hoàn tất để thống kê")
    frame = frame.sort_values(["profile", "method", "seed"]).reset_index(drop=True)
    # Compare exported summaries without trusting them as the primary source.
    for profile, root in roots.items():
        path = root / "summary.csv"
        if path.exists():
            track(path)
            original = pd.read_csv(path).set_index("method")
            for method, group in frame[frame.profile == profile].groupby("method"):
                for metric in METRICS:
                    if method not in original.index or abs(original.loc[method, metric + "_mean"] - group[metric].mean()) > 1e-9:
                        issues.append({"profile": profile, "method": method, "severity": "warning", "issue": "summary.csv cũ khác điểm tổng hợp lại: " + metric})
    audit = {"analysis_version": ANALYSIS_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
             "total_completed_runs": len(frame), "predictions_recomputed": int(frame.predictions_verified.sum()),
             "all_declared_profiles_complete": all(x["complete"] for x in inventory),
             "all_three_profiles_present": set(roots) == {"paper70k", "legacy10k", "legacy70k"},
             "official_raw_label_bytes_checked": raw_labels_verified, "issues": issues,
             "source_files": sources, "python": platform.python_version()}
    frame.to_csv(output_dir / "per_seed_recomputed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory).to_csv(output_dir / "completion_audit.csv", index=False, encoding="utf-8-sig")
    epochs.to_csv(output_dir / "training_history_all.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "audit.json", audit)
    return {"roots": roots, "frame": frame, "epochs": epochs, "inventory": pd.DataFrame(inventory),
            "audit": audit, "ae": pd.DataFrame(ae_rows), "output": output_dir}


def summarize_results(bundle):
    df, rows = bundle["frame"], []
    for (profile, method), group in df.groupby(["profile", "method"], sort=False):
        expected = int(bundle["inventory"].set_index("profile").loc[profile, "expected_seeds"])
        row = {"profile": profile, "method": method, "method_label": METHOD_LABELS[method], "seeds": len(group),
               "expected_seeds": expected, "complete": len(group) == expected,
               "collapsed_runs": int(group.collapsed.fillna(False).sum()), "mixed_environment_runs": int(group.mixed_environment.sum()),
               "guard_count_total": int(group.guard_count.fillna(0).sum()), "rescue_count_total": int(group.rescue_count.fillna(0).sum()),
               "finetune_minutes_mean": float(group.finetune_seconds.mean() / 60), "peak_gpu_gb_max": float(group.peak_gpu_gb.max())}
        for metric in METRICS:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(bundle["output"] / "summary_recomputed.csv", index=False, encoding="utf-8-sig")
    ae = bundle["ae"]
    if not ae.empty:
        ae.groupby("profile")[list(METRICS)].agg(["mean", "std", "count"]).to_csv(bundle["output"] / "ae_kmeans_summary.csv", encoding="utf-8-sig")
    return summary


def comparisons(bundle):
    df, rows, differences = bundle["frame"], [], []
    for profile, group in df.groupby("profile"):
        expected = int(bundle["inventory"].set_index("profile").loc[profile, "expected_seeds"])
        available = set(group.method)
        contrasts = [("ada_im", b) for b in ("dcss_paper", "dcss_rms", "im_rms") if b in available]
        if {"im_rms", "dcss_rms"} <= available:
            contrasts.append(("im_rms", "dcss_rms"))
        campaign_signatures = set("|".join(group.environment_signatures).split("|")) - {""}
        mixed_campaign = len(campaign_signatures) > 1 or bool(group.mixed_environment.any())
        for target, baseline in contrasts:
            a, b = group[group.method == target].set_index("seed"), group[group.method == baseline].set_index("seed")
            seeds = a.index.intersection(b.index).sort_values()
            for metric in METRICS:
                delta = a.loc[seeds, metric] - b.loc[seeds, metric]
                planned_primary = profile == "paper70k" and target == "ada_im" and baseline == "im_rms" and metric == "ARI"
                rows.append({"profile": profile, "target": target, "baseline": baseline, "metric": metric,
                             "paired_seeds": ",".join(map(str, seeds)), "planned_primary": planned_primary,
                             "complete": len(seeds) == expected, "mixed_campaign": mixed_campaign,
                             "interpretation": "thăm dò: nhiều môi trường" if mixed_campaign else "thăm dò: chỉ 3 seed" if expected < 10 else "đủ seed; cần xem giả định kiểm định",
                             **paired_stats(delta.to_numpy())})
                differences.extend({"profile": profile, "target": target, "baseline": baseline, "metric": metric,
                                    "seed": int(seed), "delta_pp": float(delta.loc[seed] * 100)} for seed in seeds)
    pairs = pd.DataFrame(rows)
    pairs["p_holm_secondary_within_profile"] = np.nan
    for profile in pairs.profile.unique():
        mask = (pairs.profile == profile) & ~pairs.planned_primary & pairs.p_signflip_two_sided.notna()
        pairs.loc[mask, "p_holm_secondary_within_profile"] = holm_adjust(pairs.loc[mask, "p_signflip_two_sided"])
    pairs.to_csv(bundle["output"] / "paired_comparisons_recomputed.csv", index=False, encoding="utf-8-sig")
    deltas = pd.DataFrame(differences)
    deltas.to_csv(bundle["output"] / "paired_deltas_per_seed.csv", index=False, encoding="utf-8-sig")
    # A predefined descriptive sensitivity view: retain only full execution traces
    # that show one identical training environment for both methods of the pair.
    sensitivity = []
    for profile, group in df.groupby("profile"):
        a = group[group.method == "ada_im"].set_index("seed")
        b = group[group.method == "im_rms"].set_index("seed")
        seeds = [s for s in a.index.intersection(b.index) if bool(a.loc[s, "strict_single_environment"]) and bool(b.loc[s, "strict_single_environment"]) and a.loc[s, "environment_signatures"] == b.loc[s, "environment_signatures"]]
        for metric in METRICS:
            sensitivity.append({"profile": profile, "metric": metric, "seeds": ",".join(map(str, seeds)),
                                "rule": "full_trace_single_matching_environment_both_methods", "posthoc_descriptive_only": True,
                                **paired_stats((a.loc[seeds, metric] - b.loc[seeds, metric]).to_numpy())})
    pd.DataFrame(sensitivity).to_csv(bundle["output"] / "environment_sensitivity.csv", index=False, encoding="utf-8-sig")
    return pairs, deltas, pd.DataFrame(sensitivity)


def scale_comparison(bundle):
    df = bundle["frame"]
    small, full = df[df.profile == "legacy10k"], df[df.profile == "legacy70k"]
    if small.empty or full.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Explicitly verify that only data size/profile/split changed.
    c1 = read_json(bundle["roots"]["legacy10k"] / "manifest.json")
    c2 = read_json(bundle["roots"]["legacy70k"] / "manifest.json")
    drop = {"profile", "split", "require_cuda", "cpu_threads"}
    if {k: v for k, v in c1["config"].items() if k not in drop} != {k: v for k, v in c2["config"].items() if k not in drop} or c1["methods"] != c2["methods"]:
        raise ValueError("Cấu hình 10k/70k khác nhau ngoài số ảnh; không thể coi là kiểm tra quy mô đơn thuần")
    joined = small.merge(full, on=["method", "seed"], suffixes=("_10k", "_70k"), validate="one_to_one")
    rows = []
    for metric in METRICS:
        joined[metric + "_delta_pp"] = 100 * (joined[metric + "_70k"] - joined[metric + "_10k"])
    for method, group in joined.groupby("method"):
        for metric in METRICS:
            rows.append({"method": method, "metric": metric,
                         "mean_10k_pct": float(group[metric + "_10k"].mean() * 100), "mean_70k_pct": float(group[metric + "_70k"].mean() * 100),
                         **paired_stats((group[metric + "_70k"] - group[metric + "_10k"]).to_numpy())})
    table = pd.DataFrame(rows)
    joined.to_csv(bundle["output"] / "scale_per_seed.csv", index=False, encoding="utf-8-sig")
    table.to_csv(bundle["output"] / "scale_summary.csv", index=False, encoding="utf-8-sig")
    return table, joined


def prediction_agreement(bundle):
    rows = []
    for profile, root in bundle["roots"].items():
        group = bundle["frame"][bundle["frame"].profile == profile]
        a = group[group.method == "ada_im"].set_index("seed")
        b = group[group.method == "im_rms"].set_index("seed")
        for seed in a.index.intersection(b.index):
            paths = [root / f"seed_{seed:02d}" / method / "predictions.npz" for method in ("ada_im", "im_rms")]
            if not all(p.exists() for p in paths):
                continue
            with np.load(paths[0], allow_pickle=False) as fa, np.load(paths[1], allow_pickle=False) as fb:
                aa, bb = fa["predictions"], fb["predictions"]
                score = scores_from_predictions(aa, bb)
                rows.append({"profile": profile, "seed": seed, "partition_ARI_between_methods": score["ARI"],
                             "agreement_after_hungarian_pct": 100 * score["ACC"],
                             "meaning": "agreement_between_partitions_not_label_accuracy"})
    table = pd.DataFrame(rows)
    table.to_csv(bundle["output"] / "im_vs_ada_partition_agreement.csv", index=False, encoding="utf-8-sig")
    return table


def formatted_summary(summary, profile):
    frame = summary[summary.profile == profile].copy()
    frame["_order"] = frame.method.map({m: i for i, m in enumerate(COLORS)})
    frame = frame.sort_values("_order")
    result = frame[["method_label", "seeds"]].rename(columns={"method_label": "Phương pháp", "seeds": "Số seed"})
    for metric in METRICS:
        result[metric + " ×100 (TB ± ĐLC mẫu)"] = [f"{100*m:.3f} ± {100*s:.3f}" for m, s in zip(frame[metric + "_mean"], frame[metric + "_sd"])]
    return result


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(Path(output) / (name + "." + suffix), dpi=180, bbox_inches="tight")
    return fig


def make_figures(bundle, summary, pairs, scale):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.titleweight": "bold"})
    output, frame = bundle["output"], bundle["frame"]
    figures = []
    order = [m for m in COLORS if m in set(frame.method)]
    for profile in bundle["roots"]:
        group = frame[frame.profile == profile]
        methods = [m for m in order if m in set(group.method)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, metric in zip(axes, METRICS):
            for i, method in enumerate(methods):
                values = group[group.method == method][metric].to_numpy() * 100
                axis.errorbar(i, values.mean(), yerr=values.std(ddof=1) if len(values)>1 else 0, fmt="o", color=COLORS[method], capsize=6, markersize=8)
                axis.scatter(i + np.linspace(-.1, .1, len(values)), values, s=18, color=COLORS[method], alpha=.6)
            axis.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=22, ha="right")
            axis.set_ylabel(metric + " × 100")
            axis.grid(axis="y", alpha=.2)
        fig.suptitle(f"{profile}: tất cả seed; chấm lớn = trung bình, thanh = độ lệch chuẩn mẫu")
        figures.append(save_figure(fig, output, profile + "_scores"))
    primary = frame[frame.profile == "paper70k"]
    if not primary.empty:
        a, b = primary[primary.method == "ada_im"].set_index("seed"), primary[primary.method == "im_rms"].set_index("seed")
        shared = sorted(set(a.index) & set(b.index))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        delta = 100*(a.loc[shared, "ARI"] - b.loc[shared, "ARI"])
        axes[0].bar(shared, delta, color=["#2A9D8F" if x >= 0 else "#D05C4D" for x in delta])
        axes[0].axhline(0, color="black", lw=.8)
        axes[0].set(xlabel="Seed", ylabel="ΔARI (điểm phần trăm)", title="ADA-IM − IM: chênh lệch từng seed", xticks=shared)
        selection = pairs[(pairs.profile == "paper70k") & (pairs.target == "ada_im") & (pairs.metric == "ARI")]
        for i, row in enumerate(selection.itertuples()):
            axes[1].errorbar(row.delta_mean_pp, i, xerr=[[row.delta_mean_pp-row.ci95_low_pp], [row.ci95_high_pp-row.delta_mean_pp]], fmt="o", color="#7A5195", capsize=5)
        axes[1].set_yticks(range(len(selection)), ["ADA-IM − " + METHOD_LABELS[m] for m in selection.baseline])
        axes[1].axvline(0, color="black", ls="--", lw=.8)
        axes[1].set(xlabel="ΔARI và KTC 95% (điểm phần trăm)", title="So sánh theo cặp — mang tính thăm dò")
        figures.append(save_figure(fig, output, "paper70k_paired_ari"))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        history = bundle["epochs"]
        for axis, quantity, label in zip(axes, ["reconstruction", "soft_silhouette", "conditional_entropy"], ["Sai số tái tạo", "Soft silhouette", "Entropy có điều kiện"]):
            for method in order:
                h = history[(history.profile == "paper70k") & (history.method == method)]
                if not h.empty and quantity in h:
                    agg = h.groupby("epoch")[quantity].agg(["mean", "std"])
                    axis.plot(agg.index, agg["mean"], color=COLORS[method], label=METHOD_LABELS[method])
                    axis.fill_between(agg.index, agg["mean"]-agg["std"], agg["mean"]+agg["std"], color=COLORS[method], alpha=.10)
            axis.set(xlabel="Epoch phân cụm", ylabel=label)
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        fig.suptitle("paper70k: lịch sử trung bình ± độ lệch chuẩn trên mọi seed")
        figures.append(save_figure(fig, output, "paper70k_training_curves"))
        fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
        matrix = np.full((len(shared), len(order)), np.nan)
        labels = np.empty_like(matrix, dtype=object)
        for i, seed in enumerate(shared):
            for j, method in enumerate(order):
                matched = primary[(primary.seed == seed) & (primary.method == method)]
                if matched.empty:
                    labels[i,j], matrix[i,j] = "Thiếu", 3
                    continue
                row = matched.iloc[0]
                name = "Pha trộn" if row.mixed_environment else "A100" if "A100" in row.device_history else "T4" if "T4" in row.device_history else "CPU/khác"
                matrix[i,j] = {"T4":0,"A100":1,"Pha trộn":2,"CPU/khác":3}[name]
                labels[i,j] = name + ("*" if row.environment_provenance == "original_manifest_only" else "")
        from matplotlib.colors import ListedColormap
        axis.imshow(matrix, cmap=ListedColormap(["#D7E8F7", "#DBEEDB", "#F8DDC4", "#EEEEEE"]), vmin=0, vmax=3, aspect="auto")
        for (i,j), text in np.ndenumerate(labels):
            axis.text(j,i,text,ha="center",va="center",fontsize=9)
        axis.set_xticks(range(len(order)), [METHOD_LABELS[m] for m in order])
        axis.set_yticks(range(len(shared)), shared)
        axis.set(ylabel="Seed", title="Nguồn gốc môi trường paper70k (* chỉ suy từ manifest V1)")
        figures.append(save_figure(fig, output, "paper70k_environment"))
        # Fixed seed 0 selected before seeing results; never show only the best seed.
        methods = [m for m in order if (bundle["roots"]["paper70k"] / "seed_00" / m / "predictions.npz").exists()]
        if methods:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
            for axis, method in zip(axes.ravel(), methods):
                with np.load(bundle["roots"]["paper70k"] / "seed_00" / method / "predictions.npz", allow_pickle=False) as archive:
                    y, pred = archive["labels"], archive["predictions"]
                    contingency = confusion_matrix(y, pred, labels=np.arange(10))
                    true_ids, cluster_ids = linear_sum_assignment(-contingency)
                    mapping = np.zeros(10, dtype=int)
                    mapping[cluster_ids] = true_ids
                    cm = confusion_matrix(y, mapping[pred], labels=np.arange(10))
                normalized = cm / cm.sum(1, keepdims=True)
                im = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
                axis.set_xticks(range(10), CLASS_NAMES, rotation=55, ha="right", fontsize=8)
                axis.set_yticks(range(10), CLASS_NAMES, fontsize=8)
                axis.set(title=METHOD_LABELS[method], xlabel="Cụm sau ghép nhãn", ylabel="Nhãn thật")
                for i in range(10):
                    for j in range(10):
                        if normalized[i,j] >= .05:
                            axis.text(j,i,f"{normalized[i,j]*100:.0f}",ha="center",va="center",fontsize=7,color="white" if normalized[i,j]>.5 else "black")
            for axis in axes.ravel()[len(methods):]:
                axis.axis("off")
            fig.suptitle("paper70k, seed 0 cố định: ma trận nhầm lẫn (% theo hàng); Hungarian chỉ dùng khi đánh giá")
            fig.colorbar(im, ax=list(axes.ravel()), shrink=.7, label="Tỷ lệ theo hàng")
            figures.append(save_figure(fig, output, "paper70k_confusion_seed0"))
    if not scale.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, metric in zip(axes, METRICS):
            for row in scale[scale.metric == metric].itertuples():
                axis.plot([0,1],[row.mean_10k_pct,row.mean_70k_pct],"o-",color=COLORS[row.method],label=METHOD_LABELS[row.method])
            axis.set_xticks([0,1],["10.000 ảnh","70.000 ảnh"])
            axis.set_ylabel(metric + " × 100")
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        fig.suptitle("Cấu hình legacy: tăng số ảnh, giữ cùng giao thức; 3 seed, hai tập có phần giao")
        figures.append(save_figure(fig, output, "legacy_10k_to_70k"))
    return figures


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in values) + " |")
    return "\n".join(lines)


def write_report(bundle, summary, pairs, sensitivity, scale, agreement):
    out, audit = bundle["output"], bundle["audit"]
    lines = ["# Kết quả kiểm chứng DCSS và ADA-IM-DCSS", "",
             f"Đã đọc {audit['total_completed_runs']} lượt hoàn tất; tính lại ACC/NMI/ARI từ {audit['predictions_recomputed']} tệp dự đoán. Không huấn luyện lại, không chọn seed tốt nhất.", "",
             "Các bảng dùng trung bình ± độ lệch chuẩn mẫu qua seed. Giá trị nhân 100; chênh lệch tính theo điểm phần trăm. ARI không phải tỷ lệ ảnh được phân loại đúng.", ""]
    for profile in bundle["roots"]:
        lines += [f"## {profile}", "", markdown_table(formatted_summary(summary, profile)), ""]
    primary = pairs[pairs.planned_primary]
    if len(primary):
        row = primary.iloc[0]
        lines += ["## Câu hỏi chính: ADA có đem lại lợi ích thêm so với IM?", "",
                  f"Trên paper70k ({row.n_pairs} seed), ΔARI = ADA-IM − IM = **{row.delta_mean_pp:+.4f} điểm phần trăm**. KTC 95%: [{row.ci95_low_pp:+.4f}; {row.ci95_high_pp:+.4f}]; p đổi dấu hai phía = {row.p_signflip_two_sided:.6f}.", ""]
        if row.ci95_low_pp <= 0 <= row.ci95_high_pp:
            lines += ["Khoảng tin cậy chứa 0: **chưa có bằng chứng rõ rằng thêm gói ADA cải thiện ARI so với IM-DCSS**. Không đồng nghĩa với chứng minh hai phương pháp tương đương; cần một biên tương đương xác định trước và kiểm định tương ứng nếu muốn kết luận đó.", ""]
        else:
            direction = "dương" if row.delta_mean_pp > 0 else "âm"
            lines += [f"Chênh lệch quan sát có chiều {direction}, KTC không chứa 0. Vẫn phải xét nguồn gốc môi trường, giả định kiểm định và thiết kế thử nghiệm trước khi kết luận.", ""]
        baseline = pairs[(pairs.profile == "paper70k") & (pairs.target == "ada_im") & (pairs.baseline == "dcss_paper")]
        if len(baseline):
            values = baseline.set_index("metric")
            lines += [f"So với DCSS theo phương trình, ADA-IM thay đổi ARI {values.loc['ARI','delta_mean_pp']:+.4f}, NMI {values.loc['NMI','delta_mean_pp']:+.4f} và ACC {values.loc['ACC','delta_mean_pp']:+.4f} điểm phần trăm. Không có lợi ích đồng đều nếu các chỉ số đi theo chiều khác nhau. Phép so sánh này đồng thời đổi hệ số loss và cơ chế sigma, nên không quy toàn bộ chênh lệch cho ADA.", ""]
    if not scale.empty:
        target = scale[scale.method == "ada_im"].set_index("metric")
        lines += ["## Tăng từ 10.000 lên 70.000 ảnh với cấu hình legacy", "",
                  "Phép này giữ kiến trúc và siêu tham số legacy; không so chéo với backbone Conv/BN của paper70k. Hai bộ dữ liệu có phần giao nhau và chỉ có 3 seed, nên đây là kiểm tra quy mô mang tính thăm dò.", "",
                  "Đối với ADA-IM: " + "; ".join(f"{m}: {target.loc[m,'mean_10k_pct']:.3f} → {target.loc[m,'mean_70k_pct']:.3f}, Δ={target.loc[m,'delta_mean_pp']:+.3f} điểm phần trăm" for m in METRICS) + ".", "",
                  "Đánh giá lợi ích ADA phải tiếp tục đối chiếu với IM-DCSS trên cùng 70k, không chỉ so mức điểm với kết quả 10k cũ.", ""]
    ada = bundle["frame"][bundle["frame"].method == "ada_im"]
    lines += ["## Chẩn đoán và nguồn gốc môi trường", "",
              f"Có {int(bundle['frame'].collapsed.fillna(False).sum())}/{len(bundle['frame'])} lượt bị gắn suy biến ở đánh giá cuối. Bộ phát hiện suy biến có ngưỡng riêng; không suy biến không đồng nghĩa phân cụm tốt theo nhãn.", "",
              f"Ở ADA-IM, tổng guard_count={int(ada.guard_count.fillna(0).sum())}, rescue_count={int(ada.rescue_count.fillna(0).sum())}. Các số này chỉ cho biết nhánh bảo vệ/phục hồi có kích hoạt hay không; không chứng minh EMA vô tác dụng.", "",
              "Các môi trường quan sát: " + "; ".join(sorted(set(bundle["frame"].device_history))) + ". Lượt V1 thiếu execution_trace chỉ được gán môi trường từ manifest, chưa có nhật ký đầy đủ từng đoạn. Bảng environment_sensitivity.csv chỉ giữ cặp có trace đầy đủ, một môi trường trùng nhau, là phân tích nhạy cảm mô tả bổ sung; không thay thế toàn bộ 10 seed.", "",
              "Không kết luận tốc độ thuật toán từ thời gian T4/A100/CPU pha trộn. Peak GPU là bộ nhớ PyTorch cấp phát, không phải toàn bộ VRAM của phiên. Các thống kê không dùng nhãn để chọn checkpoint; nhãn ở đây chỉ phục vụ kiểm tra hậu nghiệm.", "",
              "## Giới hạn diễn giải cho luận văn", "",
              "Bộ 70k gồm train và test chính thức, là phân cụm chuyển dẫn; không phải kiểm tra tổng quát hóa trên ảnh chưa dùng. 10k đã tham gia phát triển phương pháp cũng nằm trong 70k. Khoảng tin cậy qua seed phản ánh ngẫu nhiên của thuật toán trên tập cố định, không phải bất định khi lấy mẫu ảnh mới.", "",
              "DCSS-paper là tái triển khai theo phương trình với các lựa chọn đã công bố trong PROTOCOL.md; không gọi là chạy nguyên trạng mã sinh bảng của bài báo. Không diễn giải ARI, NMI và ACC như cùng một đại lượng. Các p phụ được hiệu chỉnh Holm trong từng profile; giả thuyết chính ARI ADA-IM − IM được giữ riêng theo giao thức.", "",
              "**Kết luận thực dụng:** " + ("Đã đủ các lượt khai báo để báo cáo mức điểm của chiến dịch này. " if audit["all_declared_profiles_complete"] else "Chưa đủ lượt; bảng chỉ là tạm thời. ") + "Nếu chênh lệch ADA so với IM còn nhỏ và KTC chứa 0, chưa nên lấy việc thêm ADA làm bằng chứng cải tiến rõ ràng cho luận văn. Giữ kết quả trung thực; nếu nghiên cứu tiếp, cần đối chứng tách từng thành phần và kiểm chứng trên bộ dữ liệu/điều kiện khác.", ""]
    if audit["issues"]:
        lines += ["## Các điểm cần kiểm tra", "", *["- " + str(x) for x in audit["issues"]], ""]
    text = "\n".join(lines)
    (out / "KET_QUA_VA_NHAN_XET.md").write_text(text, encoding="utf-8")
    return text


def export_bundle(bundle):
    path = bundle["output"] / "ThongKe_DCSS_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(bundle["output"].iterdir()):
            if source.is_file() and source != path:
                archive.write(source, source.name)
    return path
