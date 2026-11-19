"""New-dataset orchestration, reusing the frozen Fashion-MNIST numerical core."""
from __future__ import annotations
import copy
import json
import os
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from threadpoolctl import threadpool_limits
import dcss_core as core
import multi_data as data


@dataclass(frozen=True)
class Config(core.Config):
    profile: str = "unset"
    split: str = "all_transductive"
    input_dim: int = 0
    dense_hidden: tuple = (500, 500, 2000)
    evaluation_sample_size: int = 2000
    evaluation_seed: int = data.SAMPLE_SEED
    labeled: bool = True


def config_for(meta):
    return Config(profile=meta["dataset"], input_dim=meta["input_dim"], n_clusters=meta["n_clusters"],
                  backbone="author_conv_bn" if meta["dataset"] == "emnist" else "dense_linear_decoder",
                  labeled=meta["labeled"], primary_metric="ARI" if meta["labeled"] else "silhouette_common")


class DenseAutoencoder(nn.Module):
    """Explicit adaptation: d-500-500-2000-10, symmetric linear-output decoder."""
    def __init__(self, input_dim, latent=10, hidden=(500, 500, 2000)):
        super().__init__()
        def layers(dims, encoder):
            result = []
            for i, (a, b) in enumerate(zip(dims, dims[1:])):
                result.append(nn.Linear(a,b))
                if i < len(dims)-2:
                    result.append(nn.ReLU())
                elif encoder:
                    result.append(nn.Tanh())
            return nn.Sequential(*result)
        self.encoder = layers([input_dim, *hidden, latent], True)
        self.decoder = layers([latent, *reversed(hidden), input_dim], False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def make_model(cfg):
    if cfg.backbone == "author_conv_bn":
        return core.Autoencoder(cfg.latent_dim, cfg.backbone)
    if cfg.backbone == "dense_linear_decoder":
        return DenseAutoencoder(cfg.input_dim, cfg.latent_dim, cfg.dense_hidden)
    raise ValueError("Unknown backbone: " + cfg.backbone)


def protected_output(output_root, protected_root, resume_folder=None):
    out, old = Path(output_root).resolve(), Path(protected_root).resolve()
    if out == old or out.is_relative_to(old) or old.is_relative_to(out) or "FashionMNIST" in out.name:
        raise ValueError("New outputs must be disjoint from the protected Fashion-MNIST folder")
    if resume_folder:
        r = Path(resume_folder)
        r = r.resolve() if r.is_absolute() else (out/r).resolve()
        if not r.is_relative_to(out):
            raise ValueError("Resume folder must stay inside the new output root")
    return out


def augmented_meta(meta):
    # Pin ALL adaptation/evaluation code. Runtime hardware is not part of identity.
    return {**meta, "extension": "multi-dataset-2026-11-11-v1",
            "extension_sources": {p.name:core.digest(p) for p in [Path(__file__), Path(data.__file__), Path(core.__file__)]}}


def evaluate(x, labels, pred, cfg):
    if len(pred) != len(x) or (labels is not None and len(labels) != len(x)):
        raise ValueError("Evaluation length mismatch")
    if cfg.labeled != (labels is not None):
        raise ValueError("Ground-truth availability mismatch; no fabricated labels allowed")
    idx = np.sort(np.random.default_rng(cfg.evaluation_seed).choice(len(x), min(cfg.evaluation_sample_size,len(x)), replace=False))
    common = core.batch_tensor(x, idx, torch.device("cpu")).numpy().reshape(len(idx), -1)
    unique = len(np.unique(pred[idx]))
    valid = 1 < unique < len(idx)
    with threadpool_limits(limits=cfg.cpu_threads):
        scores = dict(silhouette_common=float(silhouette_score(common,pred[idx])) if valid else None,
                      DB_common=float(davies_bouldin_score(common,pred[idx])) if valid else None,
                      CH_common=float(calinski_harabasz_score(common,pred[idx])) if valid else None)
    scores.update(core.metrics(labels,pred) if labels is not None else {"ARI":None,"NMI":None,"ACC":None})
    counts = np.bincount(pred,minlength=cfg.n_clusters)
    scores.update(hard_active_clusters=int(np.count_nonzero(counts)), hard_largest_cluster_fraction=float(counts.max()/len(pred)),
                  common_metric_status="valid" if valid else "undefined_singleton_or_degenerate_partition",
                  common_space="fixed_input_features_same_for_every_method_and_seed", common_sample_sha256=data.array_hash(idx),
                  common_sample_n=len(idx), cluster_counts=counts.tolist())
    return scores, idx


def save_predictions(path, pred, labels, z, sample, probabilities=None):
    path = Path(path)
    payload = dict(predictions=pred, embeddings=z, sample_indices=sample)
    if labels is not None:
        payload["labels"] = labels
    if probabilities is not None:
        payload["probabilities"] = probabilities
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp,path)
    return core.digest(path)


def verify_result(path, identity):
    path = Path(path)
    if not path.exists():
        return False
    r = json.loads(path.read_text(encoding="utf-8"))
    pred = path.parent/r["predictions_file"]
    if r["identity"] != identity or r["status"] != "complete" or not pred.exists() or core.digest(pred) != r["predictions_sha256"]:
        raise core.ResumeCompatibilityError("Result or predictions are incompatible/corrupt: " + str(path))
    return True


def run_experiment(x, labels, meta, cfg, output_root, protected_root, runtime=None, deadline=None):
    runtime = runtime or core.RuntimeOptions()
    output_root = protected_output(output_root, protected_root, runtime.resume_folder)
    if len(x) != meta["n"] or cfg.labeled != (labels is not None) or (labels is not None and len(labels) != len(x)):
        raise ValueError("Data/label metadata mismatch")
    if meta["dataset"] != cfg.profile or cfg.input_dim != meta["input_dim"] or cfg.n_clusters != meta["n_clusters"]:
        raise ValueError("Configuration does not match dataset")
    device = core.select_device(runtime)
    torch.set_num_threads(cfg.cpu_threads)
    core.seed_all(cfg.seeds[0])
    root, identity, manifest = core.resolve_run(output_root,cfg,augmented_meta(meta),core.environment(device),runtime)
    session = core.start_session(root,manifest,device,runtime)
    if deadline is not None:
        session["deadline"] = deadline
    for p in (Path(__file__),Path(data.__file__)):
        dest=root/"implementations"/(core.digest(p)+"_"+p.name)
        if not dest.exists():
            dest.write_bytes(p.read_bytes())
    print("RESULT DIRECTORY:",root,flush=True)
    status = dict(state="running",identity=identity,expected_runs=len(cfg.seeds)*len(cfg.methods))
    core.atomic_json(root/"status.json",status)
    try:
        for seed in cfg.seeds:
            core.seed_all(seed)
            seed_dir = root/f"seed_{seed:02d}"
            done = [m for m in cfg.methods if verify_result(seed_dir/m/"result.json",identity+f":{seed}:{m}")]
            baseline_done = verify_result(seed_dir/"ae_kmeans"/"result.json",identity+f":{seed}:ae_kmeans")
            if len(done)==len(cfg.methods) and baseline_done:
                print("Already complete; skip training:",cfg.profile,seed,flush=True)
                continue
            model=make_model(cfg).to(device)
            history_pre = core.train_phase(model,x,cfg,seed,device,seed_dir/"pretrain",identity+f":{seed}:pretrain",runtime=runtime,session=session)
            pretrained=copy.deepcopy(model.state_dict())
            init_path=seed_dir/"initialization.pt"
            init_id=identity+f":{seed}:initialization"
            if init_path.exists():
                init=core.load_checkpoint(init_path,init_id)
                z_init,centers=init["embeddings"],init["centers"]
            else:
                z_init,_=core.infer(model,x,device,cfg.batch_size)
                print("KMeans initialization (CPU), n_init =",cfg.kmeans_n_init,flush=True)
                with threadpool_limits(limits=cfg.cpu_threads):
                    km=KMeans(n_clusters=cfg.n_clusters,n_init=cfg.kmeans_n_init,random_state=seed,algorithm="lloyd").fit(z_init.numpy())
                centers=torch.from_numpy(km.cluster_centers_).float()
                core.atomic_torch(init_path,dict(identity=init_id,embeddings=z_init,centers=centers))
            pre_trace=core.checkpoint_trace(seed_dir/"pretrain",identity+f":{seed}:pretrain",manifest["environment"])
            if not baseline_done:
                pred=core.squared_distances(z_init,centers).argmin(1).numpy()
                metrics,idx=evaluate(x,labels,pred,cfg)
                directory=seed_dir/"ae_kmeans"
                directory.mkdir(exist_ok=True)
                sha=save_predictions(directory/"predictions.npz",pred,labels,z_init.numpy(),idx)
                trace=pre_trace+[{"session_id":session["session_id"],"phase":"evaluation","environment":session["environment"]}]
                core.atomic_json(directory/"result.json",dict(identity=identity+f":{seed}:ae_kmeans",seed=seed,method="ae_kmeans",
                    profile=cfg.profile,n=len(x),status="complete",**metrics,execution_trace=trace,
                    mixed_environment=len({core.hash_json(t["environment"]) for t in trace})>1,
                    pretrain_seconds=sum(h["seconds"] for h in history_pre),finetune_seconds=0.,
                    predictions_file="predictions.npz",predictions_sha256=sha))
            for method_name in cfg.methods:
                if method_name in done:
                    continue
                core.seed_all(seed)
                model.load_state_dict(pretrained)
                head=core.RBFHead(centers,z_init,core.METHODS[method_name]).to(device)
                directory=seed_dir/method_name
                run_id=identity+f":{seed}:{method_name}"
                history=core.train_phase(model,x,cfg,seed,device,directory,run_id,head,runtime=runtime,session=session)
                z,q=core.infer(model,x,device,cfg.batch_size,head)
                pred=q.numpy().argmax(1)
                scores,idx=evaluate(x,labels,pred,cfg)
                trace=pre_trace+core.checkpoint_trace(directory,run_id,manifest["environment"])
                trace+=[{"session_id":session["session_id"],"phase":"evaluation","environment":session["environment"]}]
                sha=save_predictions(directory/"predictions.npz",pred,labels,z.numpy(),idx,q.numpy())
                result=dict(identity=run_id,seed=seed,method=method_name,profile=cfg.profile,n=len(x),status="complete",
                    **{**scores,**core.unsupervised(q.numpy())},epochs=cfg.cluster_epochs,
                    finetune_seconds=sum(h["seconds"] for h in history),pretrain_seconds=sum(h["seconds"] for h in history_pre),
                    peak_gpu_gb=max(h["peak_gpu_gb"] for h in history),guard_count=int(head.guard_count),rescue_count=int(head.rescue_count),
                    execution_trace=trace,mixed_environment=len({core.hash_json(t["environment"]) for t in trace})>1,
                    predictions_file="predictions.npz",predictions_sha256=sha)
                core.atomic_json(directory/"result.json",result)
                print({k:result[k] for k in ("profile","seed","method",cfg.primary_metric,"hard_active_clusters")},flush=True)
                summarize(root,cfg)
                del head
                if device.type=="cuda":
                    torch.cuda.empty_cache()
        status["state"]="complete"
    except BaseException as exc:
        status.update(state="paused" if isinstance(exc,core.PauseRequested) else "interrupted_or_failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        core.atomic_json(root/"status.json",status)
        report={k:v for k,v in session.items() if k not in {"deadline","legacy_environment"}}
        report.update(ended_at=core.utc_now(),state=status["state"])
        core.atomic_json(root/"sessions"/(session["session_id"]+".json"),report)
        summarize(root,cfg)
    return root


def summarize(root,cfg):
    import pandas as pd
    root=Path(root)
    rows=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("seed_*/*/result.json"))]
    methods=("ae_kmeans",*cfg.methods)
    expected={(s,m) for s in cfg.seeds for m in methods}
    actual={(r["seed"],r["method"]) for r in rows}
    if not actual <= expected or len(actual)!=len(rows):
        raise ValueError("Unexpected/duplicate run in result folder")
    envs={core.hash_json(t["environment"]) for r in rows for t in r["execution_trace"]}
    mixed=len(envs)>1
    complete=actual==expected
    core.atomic_json(root/"completeness.json",dict(completed=len(rows),expected=len(expected),complete=complete,
                        missing=[dict(seed=s,method=m) for s,m in sorted(expected-actual)],campaign_mixed_environment=mixed))
    if not rows:
        return
    metrics=("ARI","NMI","ACC","silhouette_common","DB_common","CH_common") if cfg.labeled else ("silhouette_common","DB_common","CH_common")
    frame=pd.DataFrame([{k:v for k,v in r.items() if k not in {"execution_trace","cluster_counts"}} for r in rows])
    frame.to_csv(root/"per_seed.csv",index=False)
    summaries=[]
    for method in methods:
        group=frame[frame.method==method]
        row=dict(dataset=cfg.profile,method=method,completed_seeds=len(group),expected_seeds=len(cfg.seeds),
                 complete=len(group)==len(cfg.seeds),campaign_mixed_environment=mixed)
        for metric in metrics:
            values=pd.to_numeric(group[metric],errors="coerce").dropna()
            row[metric+"_valid_seeds"]=len(values)
            row[metric+"_mean"]=float(values.mean()) if len(values) else None
            row[metric+"_sample_sd"]=float(values.std(ddof=1)) if len(values)>1 else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(root/"summary.csv",index=False)
    comparisons=[]
    for baseline in ("ae_kmeans","dcss_paper","dcss_rms","im_rms"):
        for metric in metrics:
            a=frame[frame.method=="ada_im"].set_index("seed")[metric].dropna()
            b=frame[frame.method==baseline].set_index("seed")[metric].dropna()
            shared=a.index.intersection(b.index).sort_values()
            comparisons.append(dict(comparison="ada_im - "+baseline,metric=metric,paired_seeds=shared.tolist(),
                complete=len(shared)==len(cfg.seeds),planned_primary=baseline=="im_rms" and metric==cfg.primary_metric,
                campaign_mixed_environment=mixed,confirmatory=False,
                **core.paired_statistics((a.loc[shared]-b.loc[shared]).to_numpy(dtype=float))))
    core.atomic_json(root/"paired_comparisons.json",comparisons)


def stability(root):
    """Across-seed ARI between partitions: stability, NOT agreement with true labels."""
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text())
    rows=[]
    for method in ("ae_kmeans",*manifest["config"]["methods"]):
        paths=sorted(root.glob(f"seed_*/{method}/predictions.npz"))
        preds=[np.load(p,allow_pickle=False)["predictions"] for p in paths if (p.parent/"result.json").exists()]
        scores=[float(adjusted_rand_score(a,b)) for i,a in enumerate(preds) for b in preds[i+1:]]
        rows.append(dict(method=method,seeds=len(preds),pairs=len(scores),partition_ARI_mean=float(np.mean(scores)) if scores else None,
                         pairwise_sd_descriptive=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                         interpretation="dependent seed-pair descriptions; not ground-truth accuracy and not independent test samples"))
    core.atomic_json(root/"partition_stability.json",rows)
    return rows


def runtime_smoke(device):
    """GPU/CPU checkpoint roundtrip on this actual runtime, not an accuracy test."""
    import tempfile
    device=torch.device(device)
    result=core.portability_smoke(device)
    cfg=replace(Config(),input_dim=561,backbone="dense_linear_decoder",seeds=(987654,),
                batch_size=16,pretrain_epochs=1,cluster_epochs=1,n_clusters=6)
    core.seed_all(987654)
    x=torch.rand(48,561)
    model=make_model(cfg).to(device)
    z,_=core.infer(model,x,device,16)
    centers=z[:6]
    with tempfile.TemporaryDirectory(prefix="dense_portability_") as folder:
        for target in (device,torch.device("cpu"),device):
            model=make_model(cfg).to(target)
            head=core.RBFHead(centers,z,core.METHODS["ada_im"]).to(target)
            core.train_phase(model,x,cfg,987654,target,folder,"technical_dense",head,stop_after_batches=1)
        state=core.load_training_checkpoint(folder,"technical_dense")
        assert state["next_epoch"]==0 and state["next_batch"]==3
        core.train_phase(model,x,cfg,987654,device,folder,"technical_dense",head)
        assert core.load_training_checkpoint(folder,"technical_dense")["next_epoch"]==1
    result["dense_optimizer_roundtrip_passed"]=True
    return result
