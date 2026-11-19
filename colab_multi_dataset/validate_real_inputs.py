"""Bounded real-data execution and frozen-encoder restart audit, not full experiments."""
from pathlib import Path
from dataclasses import replace
import json
import time
import numpy as np
import torch
import dcss_core as core
import multi_data as data
import multi_runner as runner


def main():
    torch.set_num_threads(2)
    root=Path(__file__).parent
    catalog=json.loads((root/"sources.json").read_text())
    audit={"purpose":"technical_only_not_research_results","device":"cpu","datasets":{}}
    for key in ("emnist","har","pendigits","waveform"):
        x,y,meta=data.load_labeled(key,root/"data/raw"/key,catalog)
        cfg=runner.config_for(meta)
        core.seed_all(321)
        model=runner.make_model(cfg)
        small=x[data.fixed_indices(len(x),max(96,cfg.n_clusters*2))]
        batch=core.batch_tensor(small,slice(None),torch.device("cpu"))
        rec,z=model(batch)
        head=core.RBFHead(z[:cfg.n_clusters].detach(),z.detach(),core.METHODS["ada_im"])
        q,target=head(z)
        loss,_=core.loss_terms(batch,rec,z,q,head.method,99,100)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in list(model.parameters())+list(head.parameters()))
        head.after_step(target)
        audit["datasets"][key]=dict(full_n=len(x),shape=list(x.shape),forward_backward_finite=True,clusters=cfg.n_clusters)
        print(key,"real input forward/backward passed",flush=True)
        del x,y,model,head,batch,rec,z,q
    import pyarrow.parquet as pq
    tables=[pq.read_table(p) for p in sorted((root/"data/personas_sample100k").glob("*.parquet"))]
    import pyarrow as pa
    texts=pa.concat_tables(tables).column("persona").slice(0,520).to_pylist()
    meta=dict(purpose="technical_subset_520_not_research",text_sha256=core.hash_json(texts))
    cache=root/"data/embedding_technical"
    try:
        data.embed_personas(texts,meta,cache,torch.device("cpu"),deadline=time.monotonic()+1)
    except core.PauseRequested:
        pass
    saved={p.name:core.digest(p) for p in cache.glob("chunk_*.npy")}
    features,embedding=data.embed_personas(texts,meta,cache,torch.device("cpu"))
    assert features.shape==(520,384) and np.isfinite(features).all()
    assert np.allclose(np.linalg.norm(features,axis=1),1.,atol=1e-5)
    assert all(core.digest(cache/name)==sha for name,sha in saved.items())
    again,_=data.embed_personas(texts,meta,cache,torch.device("cpu"))
    assert np.array_equal(features,again)
    # Simulate a session ending after the first durable chunk, retaining the
    # actual frozen weights and first 512 vectors. Rebuild only the last 8 rows.
    import shutil
    from unittest.mock import patch
    from transformers import AutoModel, AutoTokenizer
    resume_cache=root/"data/embedding_resume_technical"
    resume_cache.mkdir(exist_ok=True)
    for name in ("embedding_spec.json","chunk_000000.npy","chunk_000000.json"):
        shutil.copyfile(cache/name,resume_cache/name)
    model_loader,tokenizer_loader=AutoModel.from_pretrained,AutoTokenizer.from_pretrained
    def reuse_model(*args,**kwargs):
        kwargs["cache_dir"]=str(cache/"model")
        return model_loader(*args,**kwargs)
    def reuse_tokenizer(*args,**kwargs):
        kwargs["cache_dir"]=str(cache/"model")
        return tokenizer_loader(*args,**kwargs)
    first_sha=core.digest(resume_cache/"chunk_000000.npy")
    with patch.object(AutoModel,"from_pretrained",side_effect=reuse_model),patch.object(AutoTokenizer,"from_pretrained",side_effect=reuse_tokenizer):
        resumed,_=data.embed_personas(texts,meta,resume_cache,torch.device("cpu"))
    assert core.digest(resume_cache/"chunk_000000.npy")==first_sha
    assert np.array_equal(resumed,features)
    cfg=replace(runner.Config(),input_dim=384,backbone="dense_linear_decoder",n_clusters=20,labeled=False)
    model=runner.make_model(cfg)
    rec,z=model(torch.from_numpy(features[:64]))
    head=core.RBFHead(z[:20].detach(),z.detach(),core.METHODS["ada_im"])
    q,target=head(z)
    loss,_=core.loss_terms(torch.from_numpy(features[:64]),rec,z,q,head.method,99,100)
    loss.backward()
    assert torch.isfinite(loss)
    audit["datasets"]["personas"]=dict(sample_n_verified=100000,embedded_technical_rows=520,shape=list(features.shape),
        cached_chunks_retained=1,resumed_from_first_512_vectors_exact=True,reopen_exact=True,forward_backward_finite=True,embedding=embedding)
    core.atomic_json(root/"audit/real_input_execution.json",audit)
    print("Real input checks complete; no full experiment has been run.",flush=True)


if __name__=="__main__":
    main()
