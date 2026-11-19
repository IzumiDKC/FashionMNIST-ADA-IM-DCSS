"""Technical regression tests only; these scores are not research evidence."""
import copy
import gzip
import json
import struct
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
import dcss_core as core
import multi_data as data
import multi_runner as runner
import multi_analysis as analysis

torch.set_num_threads(2)


def fixture(labeled=True):
    core.seed_all(77)
    x=torch.rand(32,6)
    y=np.arange(32)%3 if labeled else None
    meta=dict(dataset="technical",n=len(x),input_dim=6,n_clusters=3,labeled=labeled,features_sha256=data.array_hash(x.numpy()))
    cfg=replace(runner.config_for(meta),seeds=(0,1),dense_hidden=(12,8),latent_dim=4,
        batch_size=8,pretrain_epochs=2,cluster_epochs=2,kmeans_n_init=2,require_cuda=False)
    return x,y,meta,cfg


@pytest.mark.parametrize("cluster",[False,True])
@pytest.mark.parametrize("stop_batch",[1,4])
def test_dense_resume_exactly_matches_uninterrupted_cpu(tmp_path,cluster,stop_batch):
    x,_,_,cfg=fixture()
    model=runner.make_model(cfg)
    state=copy.deepcopy(model.state_dict())
    z,_=core.infer(model,x,torch.device("cpu"),8)
    head=core.RBFHead(z[:3],z,core.METHODS["ada_im"]) if cluster else None
    hstate=copy.deepcopy(head.state_dict()) if head else None
    core.train_phase(model,x,cfg,0,torch.device("cpu"),tmp_path/"full","id",head)
    expected=copy.deepcopy(model.state_dict())
    expected_h=copy.deepcopy(head.state_dict()) if head else None
    model.load_state_dict(state)
    if head:
        head.load_state_dict(hstate)
    core.train_phase(model,x,cfg,0,torch.device("cpu"),tmp_path/"resume","id",head,stop_after_batches=stop_batch)
    saved=core.load_training_checkpoint(tmp_path/"resume","id")
    assert saved["next_batch"]==stop_batch and saved["next_epoch"]==0
    model=runner.make_model(cfg)
    head=core.RBFHead(z[:3],z,core.METHODS["ada_im"]) if cluster else None
    core.train_phase(model,x,cfg,0,torch.device("cpu"),tmp_path/"resume","id",head)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    if head:
        assert all(torch.equal(v,head.state_dict()[k]) for k,v in expected_h.items())


@pytest.mark.parametrize("labeled",[True,False])
def test_all_methods_complete_then_skip_without_training(tmp_path,monkeypatch,labeled):
    x,y,meta,cfg=fixture(labeled)
    dataset="har" if labeled else "personas"
    meta={**meta,"dataset":dataset}
    cfg=replace(cfg,profile=dataset)
    old=tmp_path/"DCSS_FashionMNIST_Full"
    old.mkdir()
    (old/"sentinel").write_bytes(b"protected")
    root=runner.run_experiment(x,y,meta,cfg,tmp_path/"new",old,core.RuntimeOptions(device="cpu"))
    complete=json.loads((root/"completeness.json").read_text())
    assert complete["complete"] and complete["completed"]==10
    results=list(root.glob("seed_*/*/result.json"))
    original={str(p):p.read_bytes() for p in results}
    for p in results:
        r=json.loads(p.read_text())
        assert (r["ARI"] is not None)==labeled
        with np.load(p.parent/"predictions.npz",allow_pickle=False) as a:
            assert ("labels" in a.files)==labeled
    def fail(*a,**k):
        raise AssertionError("Completed run must never train again")
    monkeypatch.setattr(core,"train_phase",fail)
    resumed=runner.run_experiment(x,y,meta,cfg,tmp_path/"new",old,core.RuntimeOptions(device="cpu"))
    assert root==resumed and (old/"sentinel").read_bytes()==b"protected"
    assert all(Path(p).read_bytes()==b for p,b in original.items())
    assert len(runner.stability(root))==5
    import pandas as pd
    monkeypatch.setattr(analysis,"fashion_reference",lambda path:(pd.DataFrame(columns=["dataset"]),{"test_fixture":True}))
    report,frame=analysis.campaign_report(tmp_path/"new",old)
    assert len(frame[frame.dataset==dataset])==5
    tests=json.loads((report/"primary_comparisons_holm.json").read_text())
    assert len(tests)==5
    mine=next(r for r in tests if r["dataset"]==dataset)
    assert mine["complete"] and mine["n"]==2 and mine["holm_p_family5"] is not None
    assert all(r["holm_p_family5"] is None for r in tests if r["dataset"]!=dataset)


def test_new_identity_rejects_changed_features_batch_and_network(tmp_path):
    _,_,meta,cfg=fixture()
    augmented=runner.augmented_meta(meta)
    a=core.make_identity(cfg,augmented,{"gpu":"A100"})[0]
    assert a==core.make_identity(cfg,augmented,{"gpu":"T4"})[0]
    assert a==core.make_identity(replace(cfg,cpu_threads=1),augmented,{"device":"cpu"})[0]
    for changed in (replace(cfg,batch_size=16),replace(cfg,dense_hidden=(20,10)),replace(cfg,evaluation_seed=9)):
        assert a!=core.make_identity(changed,augmented,{})[0]
    assert a!=core.make_identity(cfg,{**augmented,"features_sha256":"changed"},{})[0]


def test_protect_old_and_external_resume(tmp_path):
    old=tmp_path/"old"
    for bad in (old,old/"child",tmp_path):
        with pytest.raises(ValueError):
            runner.protected_output(bad,old)
    for resume in (str(old),"../old"):
        with pytest.raises(ValueError):
            runner.protected_output(tmp_path/"new",old,resume)
    assert runner.protected_output(tmp_path/"new",old,"technical_abc")==tmp_path/"new"


def test_common_space_same_sample_and_no_fake_labels():
    x,y,_,cfg=fixture()
    scores,idx=runner.evaluate(x,y,np.arange(len(x))%3,cfg)
    other,idx2=runner.evaluate(x,y,np.arange(len(x))%2,cfg)
    assert np.array_equal(idx,idx2) and scores["common_sample_sha256"]==other["common_sample_sha256"]
    with pytest.raises(ValueError):
        runner.evaluate(x,None,np.zeros(len(x),dtype=int),cfg)
    missing,_=runner.evaluate(x,None,np.zeros(len(x),dtype=int),replace(cfg,labeled=False))
    assert missing["ARI"] is None and missing["silhouette_common"] is None


def test_idx_strict_header_and_47_labels(tmp_path):
    p=tmp_path/"x.gz"
    a=np.arange(28*28,dtype=np.uint8)
    p.write_bytes(gzip.compress(struct.pack(">IIII",2051,1,28,28)+a.tobytes()))
    assert np.array_equal(data.idx_gz(p).ravel(),a)
    p.write_bytes(gzip.compress(struct.pack(">II",2049,3)+bytes([0,46,10])))
    assert data.idx_gz(p).tolist()==[0,46,10]
    p.write_bytes(gzip.compress(struct.pack(">II",2049,4)+bytes([0,46,10])))
    with pytest.raises(ValueError):
        data.idx_gz(p)


def test_holm_fixed_family_and_missing_tests():
    assert analysis.holm([.01,.04,.03,1.,1.])==pytest.approx([.05,.12,.12,1.,1.])


def test_corrupt_predictions_are_not_skipped(tmp_path):
    p=tmp_path/"predictions.npz"
    p.write_bytes(b"ok")
    r=tmp_path/"result.json"
    core.atomic_json(r,dict(identity="x",status="complete",predictions_file=p.name,predictions_sha256=core.digest(p)))
    assert runner.verify_result(r,"x")
    p.write_bytes(b"changed")
    with pytest.raises(core.ResumeCompatibilityError):
        runner.verify_result(r,"x")


def test_frozen_core_is_identical_to_previous_work():
    old=Path(__file__).parent.parent/"colab_full_fashion_mnist/experiment.py"
    assert core.digest(old)==core.digest(core.__file__)
