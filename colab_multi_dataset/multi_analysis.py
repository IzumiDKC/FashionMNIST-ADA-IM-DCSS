"""Read-only result aggregation; reports are written ONLY into the new campaign."""
from __future__ import annotations
import html
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import dcss_core as core
import multi_runner as runner

DATASETS=("emnist","har","pendigits","waveform","personas")


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p,kind="stable")
    adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1)
    out=np.empty_like(p)
    out[order]=adjusted
    return out.tolist()


def fashion_reference(old_root):
    """No training, migrations, summaries or writes in the historical directory."""
    roots=sorted(Path(old_root).glob("paper70k_*/manifest.json"))
    if len(roots)!=1:
        raise ValueError("Need exactly one historical paper70k profile; preserve and select it explicitly")
    root=roots[0].parent
    manifest=json.loads(roots[0].read_text(encoding="utf-8"))
    rows=[]
    for path in sorted(root.glob("seed_*/*/result.json")):
        r=json.loads(path.read_text(encoding="utf-8"))
        rows.append({k:r[k] for k in ("seed","method","ARI","NMI","ACC")})
    for path in sorted(root.glob("seed_*/ae_kmeans_evaluation.json")):
        r=json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(method="ae_kmeans",**{k:r[k] for k in ("seed","ARI","NMI","ACC")}))
    frame=pd.DataFrame(rows)
    expected={(s,m) for s in manifest["config"]["seeds"] for m in ("ae_kmeans",*manifest["config"]["methods"])}
    if {(r["seed"],r["method"]) for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError("Historical paper70k results are incomplete/duplicated")
    summaries=[]
    for method,g in frame.groupby("method",sort=False):
        row=dict(dataset="fashion_mnist_70k_historical",method=method,completed_seeds=len(g),expected_seeds=len(g),complete=True,
                 historical_read_only=True,campaign_mixed_environment=True)
        for m in ("ARI","NMI","ACC"):
            row[m+"_mean"]=float(g[m].mean())
            row[m+"_sample_sd"]=float(g[m].std(ddof=1))
        summaries.append(row)
    return pd.DataFrame(summaries),dict(root=str(root),manifest_sha256=core.digest(roots[0]),
        result_sha256={str(p.relative_to(root)):core.digest(p) for p in sorted(root.glob("seed_*/*/result.json"))},
        interpretation="Historical reference; not included in the new multiple-testing family. Mixed T4/A100 history retained.")


def campaign_report(output_root, old_root):
    out=runner.protected_output(output_root,old_root)
    report=out/"analysis"
    report.mkdir(parents=True,exist_ok=True)
    old_summary,provenance=fashion_reference(old_root)
    core.atomic_json(report/"fashion_reference_provenance.json",provenance)
    frames=[old_summary]
    inventory,primary,stabilities=[],[],[]
    for key in DATASETS:
        roots=sorted(out.glob(key+"_*/manifest.json"))
        if len(roots)>1:
            raise ValueError("Multiple campaigns for one dataset; do not combine them: "+key)
        if not roots:
            inventory.append(dict(dataset=key,completed=0,expected=50,complete=False))
            primary.append(dict(dataset=key,metric="silhouette_common" if key=="personas" else "ARI",complete=False,n=0,sign_flip_p_two_sided=None))
            continue
        root=roots[0].parent
        manifest=json.loads(roots[0].read_text(encoding="utf-8"))
        cfg=runner.Config(**{**manifest["config"],"seeds":tuple(manifest["config"]["seeds"]),
                             "methods":tuple(manifest["config"]["methods"]),"dense_hidden":tuple(manifest["config"]["dense_hidden"])})
        runner.summarize(root,cfg)
        completion=json.loads((root/"completeness.json").read_text())
        inventory.append(dict(dataset=key,**completion))
        if (root/"summary.csv").exists():
            frames.append(pd.read_csv(root/"summary.csv"))
            stabilities += [dict(dataset=key,**r) for r in runner.stability(root)]
            pairs=json.loads((root/"paired_comparisons.json").read_text())
            p=next(r for r in pairs if r["planned_primary"])
            primary.append(dict(dataset=key,**p))
        else:
            primary.append(dict(dataset=key,metric=cfg.primary_metric,complete=False,n=0,sign_flip_p_two_sided=None))
    # Fixed family of FIVE additional-dataset comparisons; missing tests remain p=1.
    adjusted=holm([r["sign_flip_p_two_sided"] if r.get("complete") and r.get("sign_flip_p_two_sided") is not None else 1. for r in primary])
    for r,p in zip(primary,adjusted):
        r["holm_p_family5"]=p if r["complete"] else None
        r["interpretation"]="planned comparison; inspect CI, valid-seed counts, mixed hardware and representation limits; no automatic thesis claim"
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(report/"all_datasets_summary.csv",index=False)
    pd.DataFrame(inventory).to_csv(report/"progress.csv",index=False)
    pd.DataFrame(stabilities).to_csv(report/"partition_stability.csv",index=False)
    core.atomic_json(report/"primary_comparisons_holm.json",primary)
    core.atomic_json(report/"progress.json",inventory)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    for ax,key in zip(axes.ravel(),("fashion_mnist_70k_historical",*DATASETS)):
        g=combined[combined.dataset==key]
        metric="silhouette_common" if key=="personas" else "ARI"
        if g.empty:
            ax.text(.5,.5,"Chưa có kết quả",ha="center",va="center",transform=ax.transAxes)
        else:
            ax.bar(np.arange(len(g)),g[metric+"_mean"],yerr=g[metric+"_sample_sd"].fillna(0),capsize=3)
            ax.set_xticks(np.arange(len(g)),g.method,rotation=35,ha="right")
            ax.set_ylabel(metric+" · trung bình ± SD")
        ax.set_title(key)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(report/"scores.png",dpi=160)
    plt.close(fig)
    tables=[]
    for key,g in combined.groupby("dataset",sort=False):
        metric_cols=[c for c in g if c.endswith("_mean") or c.endswith("_sample_sd")]
        cols=["method","completed_seeds","expected_seeds","complete"]+[c for c in metric_cols if not g[c].isna().all()]
        tables.append("<h2>"+html.escape(key)+"</h2>"+g[cols].to_html(index=False,float_format=lambda v:f"{v:.6f}"))
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>DCSS — nhiều bộ dữ liệu</title>
<style>body{font:16px system-ui;max-width:1350px;margin:auto;padding:24px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:8px}img{max-width:100%}.scroll{overflow-x:auto}</style>
<h1>DCSS / IM-DCSS / ADA-IM-DCSS — kết quả đang có</h1>
<p>Fashion-MNIST là kết quả cũ được giữ nguyên. Các bộ mới dùng 10 seed, epoch cuối cố định; bảng thiếu seed là tạm thời.
ARI/NMI/ACC là tỷ lệ (0,45 = 45%). Chỉ số nội tại dùng cùng đầu vào và cùng 2.000 mẫu đánh giá cho mọi phương pháp.
Nemotron không có nhãn chuẩn: ARI giữa các seed chỉ phản ánh độ ổn định, không phải độ đúng.</p>
<p>Kiểm định ghép cặp chính: ADA-IM − IM. Hiệu chỉnh Holm cho 5 bộ bổ sung; không gộp Fashion cũ vào họ kiểm định mới.
DB thấp hơn tốt hơn; ARI/NMI/ACC/silhouette/CH cao hơn tốt hơn trong từng bộ và cùng biểu diễn.
Đổi phần cứng được ghi lại; không so sánh tốc độ trực tiếp giữa các GPU.</p><img src="scores.png"><h2>Tiến độ</h2>'''
    page+=pd.DataFrame([{k:v for k,v in r.items() if k!="missing"} for r in inventory]).to_html(index=False)
    page+="<div class=scroll>"+"".join(tables)+"<h2>So sánh chính</h2>"+pd.DataFrame(primary).to_html(index=False)+"</div></html>"
    (report/"ThongKe_DCSS_MultiDataset.html").write_text(page,encoding="utf-8")
    return report,combined


def export_review(output_root,old_root):
    out=runner.protected_output(output_root,old_root)
    target=out/"results_for_review.zip"
    with zipfile.ZipFile(target.with_suffix(".zip.tmp"),"w",zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if not path.is_file() or any(part in {"cache","raw","model"} for part in path.relative_to(out).parts):
                continue
            if path.suffix in {".json",".csv",".png",".py",".md",".html"}:
                archive.write(path,path.relative_to(out))
    target.with_suffix(".zip.tmp").replace(target)
    return target
