"""CPU-only result reader. Never imports a training module or opens checkpoints."""
from __future__ import annotations
import base64
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

DATASETS = ("emnist", "har", "pendigits", "waveform", "personas")
FASHION = "fashion_mnist_70k_historical"
METHODS = ("ae_kmeans", "dcss_paper", "dcss_rms", "im_rms", "ada_im")
METHOD_NAMES = {"ae_kmeans":"AE + KMeans", "dcss_paper":"DCSS", "dcss_rms":"DCSS-RMS", "im_rms":"IM-DCSS", "ada_im":"ADA-IM-DCSS"}
DATA_NAMES = {FASHION:"Fashion-MNIST 70k (đối chiếu cũ)", "emnist":"EMNIST Balanced", "har":"HAR",
              "pendigits":"Pendigits", "waveform":"Waveform V1", "personas":"Nemotron 100k"}
METRICS = ("ARI", "NMI", "ACC", "silhouette_common", "DB_common", "CH_common")
METRIC_NAMES = {"silhouette_common":"Silhouette chung", "DB_common":"Davies–Bouldin", "CH_common":"Calinski–Harabasz",
                "ARI":"ARI", "NMI":"NMI", "ACC":"ACC"}
COLORS = ["#64748b", "#c98735", "#8c6bb1", "#167d9a", "#c13e63"]
HISTORY_FIELDS = ("epoch", "loss", "reconstruction", "soft_silhouette", "conditional_entropy", "balance_kl",
                  "active_clusters", "seconds", "seen", "peak_gpu_gb")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unpack_snapshot(encoded, expected_sha):
    packed = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(packed).hexdigest() != expected_sha:
        raise ValueError("Snapshot checksum mismatch")
    return json.loads(gzip.decompress(packed))


def read_campaign(root, fashion_root=None):
    """Read data files only. Retain each raw-result checksum in the snapshot."""
    root = Path(root).resolve()
    payload = dict(schema=1, source=str(root), snapshot_date=None, records=[], histories=[], datasets={}, stability=[])
    for dataset in DATASETS:
        matches = list(root.glob(dataset+"_*/manifest.json"))
        if len(matches) != 1:
            raise ValueError(f"{dataset}: cần đúng một manifest; tìm thấy {len(matches)}")
        manifest_path = matches[0]
        directory = manifest_path.parent
        manifest = read(manifest_path)
        cfg, meta = manifest["config"], manifest["data"]
        payload["datasets"][dataset] = dict(n=meta["n"], k=cfg["n_clusters"], seeds=cfg["seeds"],
            pretrain_epochs=cfg["pretrain_epochs"], cluster_epochs=cfg["cluster_epochs"], identity=manifest["identity"],
            manifest_sha256=sha(manifest_path), raw_folder=directory.name, labeled=cfg["labeled"],
            stored_completion=read(directory/"completeness.json") if (directory/"completeness.json").exists() else None,
            embedding=meta.get("embedding"), configuration=cfg)
        for path in sorted(directory.glob("seed_*/*/result.json")):
            raw = read(path)
            record = {k:v for k,v in raw.items() if k not in {"execution_trace","evaluation_environment"}}
            record["dataset"] = dataset
            record["raw_result_sha256"] = sha(path)
            trace = raw.get("execution_trace", [])
            record["hardware"] = sorted({t["environment"].get("gpu") or t["environment"]["device"] for t in trace})
            record["environment_signatures"] = sorted({hashlib.sha256(json.dumps(t["environment"], sort_keys=True).encode()).hexdigest() for t in trace})
            payload["records"].append(record)
        for path in sorted(directory.glob("seed_*/*/history.json")):
            history = read(path)
            payload["histories"].append(dict(dataset=dataset, seed=int(path.parent.parent.name.split("_")[1]),
                phase=path.parent.name, raw_history_sha256=sha(path),
                rows=[{k:r[k] for k in HISTORY_FIELDS if k in r} for r in history]))
        if (directory/"partition_stability.json").exists():
            payload["stability"] += [dict(dataset=dataset, **row) for row in read(directory/"partition_stability.json")]
    if fashion_root is not None:
        old = list(Path(fashion_root).glob("paper70k_*/manifest.json"))
        if len(old) != 1:
            raise ValueError("Fashion-MNIST: cần đúng một hồ sơ paper70k")
        directory = old[0].parent
        manifest = read(old[0])
        payload["datasets"][FASHION] = dict(n=70000, k=10, seeds=manifest["config"]["seeds"],
            labeled=True, historical=True, manifest_sha256=sha(old[0]), raw_folder=directory.name)
        paths = list(directory.glob("seed_*/*/result.json")) + list(directory.glob("seed_*/ae_kmeans_evaluation.json"))
        for path in sorted(paths):
            raw = read(path)
            method = raw.get("method", "ae_kmeans")
            payload["records"].append(dict(dataset=FASHION, method=method, seed=raw["seed"], n=70000,
                status="complete", historical=True, mixed_environment=True, raw_result_sha256=sha(path),
                **{m:raw[m] for m in ("ARI","NMI","ACC")}))
    return payload


def validate_payload(payload):
    records = payload["records"]
    keys = [(r["dataset"],r["seed"],r["method"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Trùng dataset/seed/phương pháp; không được trộn lượt chạy")
    for r in records:
        d = r["dataset"]
        meta = payload["datasets"][d]
        if r["method"] not in METHODS or r["seed"] not in meta["seeds"] or r["n"] != meta["n"] or r["status"] != "complete":
            raise ValueError("Kết quả không khớp manifest")
        if d == FASHION:
            continue
        if r["identity"] != meta["identity"]+f":{r['seed']}:{r['method']}":
            raise ValueError("Sai danh tính kết quả")
        if len(r["cluster_counts"]) != meta["k"] or sum(r["cluster_counts"]) != r["n"]:
            raise ValueError("Sai số mẫu/cụm trong dự đoán đã lưu")
        if r["hard_active_clusters"] != sum(x>0 for x in r["cluster_counts"]):
            raise ValueError("Sai số cụm hoạt động")
        if not meta["labeled"] and any(r.get(m) is not None for m in ("ARI","NMI","ACC")):
            raise ValueError("Không được tạo nhãn đúng giả cho Nemotron")
        for m in METRICS:
            if r.get(m) is not None and not math.isfinite(r[m]):
                raise ValueError("Chỉ số không hữu hạn; dùng null khi không xác định")
    history_index = {(h["dataset"],h["seed"],h["phase"]):h for h in payload["histories"]}
    if len(history_index) != len(payload["histories"]):
        raise ValueError("Trùng lịch sử huấn luyện")
    for r in records:
        d = r["dataset"]
        if d == FASHION:
            continue
        meta = payload["datasets"][d]
        for phase,total in [("pretrain",meta["pretrain_epochs"])] + ([] if r["method"]=="ae_kmeans" else [(r["method"],meta["cluster_epochs"])]):
            history = history_index.get((d,r["seed"],phase),{}).get("rows",[])
            if [row["epoch"] for row in history] != list(range(1,total+1)):
                raise ValueError(f"{d}/{r['seed']}/{phase}: kết quả cuối thiếu lịch sử epoch")
            if not all(row["seen"]==meta["n"] and math.isfinite(row["loss"]) for row in history):
                raise ValueError("Lịch sử có loss lỗi hoặc sai số mẫu")
    for d in DATASETS:
        rows = [r for r in records if r["dataset"]==d]
        if len({r["common_sample_sha256"] for r in rows})>1:
            raise ValueError("Các phương pháp dùng mẫu đánh giá nội tại khác nhau")
    return dict(records=len(records), new_results=sum(r["dataset"]!=FASHION for r in records),
                history_phases=len(payload["histories"]), epoch_rows=sum(len(h["rows"]) for h in payload["histories"]),
                verification="JSON và lịch sử; không tính lại dự đoán từ model/checkpoint")


def result_frame(payload):
    frame = pd.DataFrame(payload["records"])
    for m in METRICS:
        if m not in frame:
            frame[m] = np.nan
        frame[m] = pd.to_numeric(frame[m],errors="raise")
    return frame


def make_tables(payload):
    validate_payload(payload)
    frame = result_frame(payload)
    progress, summaries, health = [], [], []
    for d,meta in payload["datasets"].items():
        group = frame[frame.dataset==d]
        expected = {(s,m) for s in meta["seeds"] for m in METHODS}
        actual = set(zip(group.seed,group.method))
        signatures = {v for r in payload["records"] if r["dataset"]==d for v in r.get("environment_signatures",[])}
        mixed = len(signatures)>1 or bool(group.mixed_environment.fillna(False).any())
        progress.append(dict(dataset=d,samples=meta["n"],K=meta["k"],completed=len(actual),expected=len(expected),
            complete=actual==expected,seeds=len(meta["seeds"]),mixed_environment=mixed,historical=d==FASHION))
        for method in METHODS:
            g = group[group.method==method]
            for metric in METRICS:
                if d==FASHION and metric not in ("ARI","NMI","ACC") or d=="personas" and metric in ("ARI","NMI","ACC"):
                    continue
                v = g[metric].dropna()
                summaries.append(dict(dataset=d,method=method,metric=metric,completed=len(g),expected=len(meta["seeds"]),
                    valid=len(v),undefined=len(g)-len(v),mean=float(v.mean()) if len(v) else np.nan,
                    sample_sd=float(v.std(ddof=1)) if len(v)>1 else np.nan))
            if d!=FASHION:
                health.append(dict(dataset=d,method=method,completed=len(g),K=meta["k"],
                    flagged_collapsed=int(g.collapsed.fillna(False).sum()) if method!="ae_kmeans" and len(g) else np.nan,
                    single_cluster_runs=int((g.hard_active_clusters==1).sum()),
                    active_min=float(g.hard_active_clusters.min()) if len(g) else np.nan,
                    active_mean=float(g.hard_active_clusters.mean()) if len(g) else np.nan,
                    active_max=float(g.hard_active_clusters.max()) if len(g) else np.nan,
                    largest_cluster_max=float(g.hard_largest_cluster_fraction.max()) if len(g) else np.nan,
                    undefined_internal=int(g.silhouette_common.isna().sum())))
    return frame,pd.DataFrame(progress),pd.DataFrame(summaries),pd.DataFrame(health)


def paired(d):
    a = np.asarray(d,dtype=float)
    n = len(a)
    out = dict(n=n,mean_delta=float(a.mean()) if n else np.nan,wins=int((a>0).sum()),losses=int((a<0).sum()),ties=int((a==0).sum()))
    if n<2:
        return dict(**out,ci95_low=np.nan,ci95_high=np.nan,p_raw=np.nan)
    margin = float(stats.t.ppf(.975,n-1)*a.std(ddof=1)/math.sqrt(n))
    signs = np.asarray(list(itertools.product([-1.,1.],repeat=n))) if n<=16 else None
    return dict(**out,ci95_low=out["mean_delta"]-margin,ci95_high=out["mean_delta"]+margin,
                p_raw=float(np.mean(np.abs((signs*a).mean(1))>=abs(a.mean())-1e-15)) if signs is not None else np.nan)


def holm(values):
    p = np.asarray(values,dtype=float)
    order = np.argsort(p,kind="stable")
    out = np.empty_like(p)
    out[order] = np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out


def comparisons(frame,payload):
    rows = []
    for d in DATASETS:
        for baseline in METHODS[:-1]:
            metrics = METRICS[3:] if d=="personas" else METRICS
            for metric in metrics:
                a = frame[(frame.dataset==d)&(frame.method=="ada_im")].set_index("seed")[metric].dropna()
                b = frame[(frame.dataset==d)&(frame.method==baseline)].set_index("seed")[metric].dropna()
                common = a.index.intersection(b.index).sort_values()
                rows.append(dict(dataset=d,comparison="ada_im - "+baseline,metric=metric,
                    primary=baseline=="im_rms" and metric==("silhouette_common" if d=="personas" else "ARI"),
                    complete=set(common)==set(payload["datasets"][d]["seeds"]),**paired(a.loc[common]-b.loc[common])))
    all_pairs = pd.DataFrame(rows)
    primary = all_pairs[all_pairs.primary].copy()
    if set(primary.dataset)!=set(DATASETS) or len(primary)!=5:
        raise ValueError("Họ kiểm định chính phải có đúng 5 bộ")
    values = [r.p_raw if r.complete and pd.notna(r.p_raw) else 1. for r in primary.itertuples()]
    primary["p_holm_family5"] = [p if done else np.nan for p,done in zip(holm(values),primary.complete)]
    return all_pairs,primary


def display_summary(summary,dataset,metrics):
    rows = []
    for method in METHODS:
        row = {"Phương pháp":METHOD_NAMES[method]}
        for metric in metrics:
            g = summary[(summary.dataset==dataset)&(summary.method==method)&(summary.metric==metric)]
            if g.empty:
                continue
            r = g.iloc[0]
            factor = 100 if metric in ("ARI","NMI","ACC") else 1
            value = "—" if pd.isna(r["mean"]) else f"{factor*r['mean']:.3f} ± {factor*r['sample_sd']:.3f}"
            row[METRIC_NAMES[metric]+(" ×100" if factor==100 else "")] = value
            row[metric+" (hợp lệ/đã xong)"] = f"{int(r.valid)}/{int(r.completed)}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_show(fig,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_label_scores(frame,summary,out):
    datasets = [d for d in (FASHION,*DATASETS) if d!="personas" and d in set(summary.dataset)]
    fig,axes = plt.subplots(len(datasets),3,figsize=(16,3.5*len(datasets)),squeeze=False,layout="constrained")
    for row,d in enumerate(datasets):
        for col,metric in enumerate(("ARI","NMI","ACC")):
            ax = axes[row,col]
            for i,m in enumerate(METHODS):
                s = summary[(summary.dataset==d)&(summary.method==m)&(summary.metric==metric)].iloc[0]
                if pd.notna(s["mean"]):
                    ax.bar(i,100*s["mean"],yerr=100*s.sample_sd if pd.notna(s.sample_sd) else 0,color=COLORS[i],capsize=3,alpha=.8)
                g = frame[(frame.dataset==d)&(frame.method==m)].sort_values("seed")
                ax.scatter(i+np.linspace(-.12,.12,len(g)),100*g[metric],s=13,color="#1f2937",alpha=.55,zorder=3)
            ax.set_xticks(range(5),[METHOD_NAMES[m] for m in METHODS],rotation=25,ha="right",fontsize=8)
            ax.set_title(DATA_NAMES[d]+" · "+metric)
            ax.set_ylabel(metric+" ×100 · TB ± SD; chấm = seed")
            ax.grid(axis="y",alpha=.2)
    save_show(fig,Path(out)/"01_label_scores_all_seeds.png")


def plot_personas(frame,summary,out):
    fig,axes = plt.subplots(1,3,figsize=(17,4.8),layout="constrained")
    for ax,metric in zip(axes,METRICS[3:]):
        for i,m in enumerate(METHODS):
            s=summary[(summary.dataset=="personas")&(summary.method==m)&(summary.metric==metric)].iloc[0]
            if pd.notna(s["mean"]):
                ax.bar(i,s["mean"],yerr=s.sample_sd if pd.notna(s.sample_sd) else 0,color=COLORS[i],capsize=3)
                ax.annotate(f"n={int(s.valid)}/{int(s.completed)}",(i,s["mean"]),xytext=(0,12),textcoords="offset points",ha="center",fontsize=8)
        ax.set_xticks(range(5),[METHOD_NAMES[m] for m in METHODS],rotation=30,ha="right")
        ax.set_title(METRIC_NAMES[metric]+(" ↓" if metric=="DB_common" else " ↑"))
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Nemotron · cùng đặc trưng/mẫu đánh giá; xem số cụm và suy biến trước khi diễn giải")
    save_show(fig,Path(out)/"02_personas_internal_metrics.png")


def plot_primary(primary,out):
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout="constrained",gridspec_kw={"width_ratios":[2,1]})
    for ax,subset,factor in [(axes[0],primary[primary.dataset!="personas"],100),(axes[1],primary[primary.dataset=="personas"],1)]:
        for i,r in enumerate(subset.itertuples()):
            if pd.isna(r.ci95_low):
                continue
            ax.errorbar(r.mean_delta*factor,i,xerr=[[factor*(r.mean_delta-r.ci95_low)],[factor*(r.ci95_high-r.mean_delta)]],
                        fmt="o",color="#c13e63",capsize=5)
            p = f"{r.p_holm_family5:.4f}" if pd.notna(r.p_holm_family5) else "chưa đủ"
            ax.annotate("p Holm="+p,(r.mean_delta*factor,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
        ax.set_yticks(range(len(subset)),[DATA_NAMES[d] for d in subset.dataset])
        ax.axvline(0,color="#475569",linestyle="--")
        ax.margins(y=.3)
        ax.grid(axis="x",alpha=.2)
        ax.set_xlabel("Chênh lệch ARI ×100" if factor==100 else "Chênh lệch silhouette")
    fig.suptitle("ADA-IM − IM: trung bình và CI 95% chưa hiệu chỉnh; p Holm theo họ 5 phép thử")
    save_show(fig,Path(out)/"03_primary_paired_comparisons.png")


def plot_health(health,out):
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained")
    for ax,column,title in [(axes[0],"flagged_collapsed","Tỷ lệ lượt bị đánh dấu suy biến"),(axes[1],"active_mean","Tỷ lệ số cụm hoạt động trung bình / K")]:
        matrix=np.full((5,5),np.nan)
        labels={}
        for i,d in enumerate(DATASETS):
            for j,m in enumerate(METHODS):
                r=health[(health.dataset==d)&(health.method==m)].iloc[0]
                val=r[column]
                if pd.notna(val) and r.completed:
                    matrix[i,j]=val/(r.completed if column=="flagged_collapsed" else r.K)
                    labels[i,j]=f"{int(val)}/{int(r.completed)}" if column=="flagged_collapsed" else f"{val:.1f}/{int(r.K)}"
                else:
                    labels[i,j]="—"
        palette=plt.get_cmap("YlOrRd" if column=="flagged_collapsed" else "YlGnBu").copy()
        palette.set_bad("#e5e7eb")
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap=palette,aspect="auto")
        for (i,j),s in labels.items():
            ax.text(j,i,s,ha="center",va="center",color="white" if np.isfinite(matrix[i,j]) and matrix[i,j]>.6 else "#111827",fontsize=9)
        ax.set_xticks(range(5),[METHOD_NAMES[m] for m in METHODS],rotation=25,ha="right")
        ax.set_yticks(range(5),[DATA_NAMES[d] for d in DATASETS])
        ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.7)
    save_show(fig,Path(out)/"04_cluster_health.png")


def plot_stability(payload,out):
    frame=pd.DataFrame(payload["stability"])
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout="constrained")
    for ax,d in zip(axes.ravel(),DATASETS):
        for i,m in enumerate(METHODS):
            g=frame[(frame.dataset==d)&(frame.method==m)]
            if not g.empty and pd.notna(g.iloc[0].partition_ARI_mean):
                r=g.iloc[0]
                ax.bar(i,r.partition_ARI_mean,color=COLORS[i],alpha=.85)
        ax.set_xticks(range(5),[METHOD_NAMES[m] for m in METHODS],rotation=25,ha="right",fontsize=8)
        ax.set_title(DATA_NAMES[d])
        ax.set_ylabel("ARI trung bình giữa phân hoạch của các seed")
        ax.grid(axis="y",alpha=.2)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Độ ổn định: 45 cặp phụ thuộc nhau; không phải ARI theo nhãn đúng")
    save_show(fig,Path(out)/"05_partition_stability.png")


def history_frame(payload):
    return pd.DataFrame([dict(dataset=h["dataset"],seed=h["seed"],phase=h["phase"],**r) for h in payload["histories"] for r in h["rows"]])


def plot_histories(payload,out):
    frame=history_frame(payload)
    fig,axes=plt.subplots(2,5,figsize=(22,8),layout="constrained")
    for col,d in enumerate(DATASETS):
        for row,metric in enumerate(("reconstruction","soft_silhouette")):
            ax=axes[row,col]
            for i,m in enumerate(METHODS[1:],start=1):
                g=frame[(frame.dataset==d)&(frame.phase==m)].groupby("epoch")[metric].agg(["mean","std"])
                if len(g):
                    ax.plot(g.index,g["mean"],label=METHOD_NAMES[m],color=COLORS[i],linewidth=1.5)
                    ax.fill_between(g.index,g["mean"]-g["std"].fillna(0),g["mean"]+g["std"].fillna(0),color=COLORS[i],alpha=.12)
            ax.set_title(DATA_NAMES[d])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE tái tạo" if row==0 else "Silhouette mềm mini-batch")
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4)
    fig.suptitle("Quá trình tinh chỉnh: trung bình ± SD qua các seed; không chọn epoch theo nhãn")
    save_show(fig,Path(out)/"06_training_histories.png")


def cluster_detail(frame,dataset="personas",method="ada_im",seed=0):
    if dataset==FASHION:
        raise ValueError("Snapshot Fashion chỉ dùng đối chiếu chỉ số")
    g=frame[(frame.dataset==dataset)&(frame.method==method)&(frame.seed==seed)]
    if len(g)!=1:
        raise ValueError("Không có đúng một kết quả đã chọn")
    counts=g.iloc[0].cluster_counts
    return pd.DataFrame({"Cụm":range(len(counts)),"Số mẫu":counts,"Tỷ lệ":np.array(counts)/sum(counts)})


def export_tables(out,payload,frame,progress,summary,health,all_pairs,primary):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    source=payload.get("source")
    if source and Path(source).is_dir() and (out.resolve()==Path(source).resolve() or out.resolve().is_relative_to(Path(source).resolve())):
        raise ValueError("Thư mục xuất phải nằm ngoài thư mục kết quả gốc")
    tables={"per_seed":frame.drop(columns=["environment_signatures","cluster_counts"],errors="ignore"),
        "progress":progress,"summary":summary,"cluster_health":health,"all_paired_comparisons":all_pairs,
        "primary_comparisons_holm":primary,"partition_stability":pd.DataFrame(payload["stability"]),
        "training_history":history_frame(payload)}
    for name,table in tables.items():
        table.to_csv(out/(name+".csv"),index=False,encoding="utf-8-sig")
        (out/(name+".json")).write_text(table.to_json(orient="records",force_ascii=False,indent=2),encoding="utf-8")
    report=dict(source=source,snapshot_date=payload.get("snapshot_date"),validation=validate_payload(payload),
                units="raw fractions in CSV/JSON; SD uses ddof=1",
                limitation="Reads stored scores/histories; does not recompute all metrics from predictions.npz",
                stability_source="stored partition_stability.json; descriptive dependent seed pairs")
    (out/"analysis_provenance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return tables


def export_html_zip(out,tables,primary):
    import zipfile
    out=Path(out)
    html=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Thống kê DCSS nhiều bộ dữ liệu</title>',
          '<style>body{font:15px system-ui;max-width:1450px;margin:auto;padding:24px;color:#17202a}img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd1d1;padding:6px}.scroll{overflow-x:auto}h2{margin-top:36px}</style>',
          '<h1>Thống kê DCSS, IM-DCSS và ADA-IM-DCSS</h1><p>Chỉ đọc kết quả; không huấn luyện. Fashion-MNIST là đối chiếu lịch sử. SD mẫu dùng ddof=1. CSV/JSON giữ đơn vị gốc.</p>',
          '<p>Nemotron không có nhãn chuẩn. Cần đọc chỉ số nội tại cùng số cụm hoạt động và cờ suy biến. ARI giữa seed chỉ mô tả độ ổn định. Không loại lượt xấu; chỉ số không xác định để trống.</p>',
          '<p>So sánh chính ADA-IM − IM: hiệu chỉnh Holm trên 5 bộ bổ sung; CI 95% từng bộ chưa hiệu chỉnh. Thống kê từ điểm đã lưu, chưa tính lại mọi điểm từ predictions.npz.</p>']
    for name in ("progress","primary_comparisons_holm","cluster_health"):
        html.append("<h2>"+name+"</h2><div class=scroll>"+tables[name].to_html(index=False,na_rep="—",float_format=lambda v:f"{v:.6f}")+"</div>")
    for p in sorted(out.glob("*.png")):
        encoded=base64.b64encode(p.read_bytes()).decode()
        html.append("<h2>"+p.stem+"</h2><img src='data:image/png;base64,"+encoded+"'>")
    html.append("<h2>Trung bình và độ lệch chuẩn</h2><div class=scroll>"+tables["summary"].to_html(index=False,na_rep="—")+"</div></html>")
    path=out/"ThongKe_DCSS_MultiDataset.html"
    path.write_text("\n".join(html),encoding="utf-8")
    archive=out/"ThongKe_DCSS_MultiDataset.zip"
    with zipfile.ZipFile(archive.with_suffix(".zip.tmp"),"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in {".csv",".json",".png",".html"}:
                z.write(p,p.name)
    archive.with_suffix(".zip.tmp").replace(archive)
    return path,archive
