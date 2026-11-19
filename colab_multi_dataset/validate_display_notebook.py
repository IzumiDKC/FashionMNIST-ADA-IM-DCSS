"""Golden-result checks and actual execution of the delivered statistics notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
import statistics_only as report

ROOT=Path(__file__).parent.resolve()
PROJECT=ROOT.parent
SOURCE=ROOT/"audit/results_2026_11_15/review"


def main():
    payload=report.read_campaign(SOURCE,PROJECT/"DCSS_FashionMNIST_Full")
    frame,progress,summary,health=report.make_tables(payload)
    _,primary=report.comparisons(frame,payload)
    assert len(frame)==300 and progress.complete.all()
    assert health.flagged_collapsed.sum()==26
    assert len(frame[frame.dataset=="personas"])==50
    assert frame.loc[frame.dataset=="personas",["ARI","NMI","ACC"]].isna().all().all()
    gold=pd.read_csv(SOURCE/"analysis/all_datasets_summary.csv").set_index(["dataset","method"])
    checked=0
    for r in summary.itertuples():
        expected=gold.loc[(r.dataset,r.method)]
        for suffix,value in [("_mean",r.mean),("_sample_sd",r.sample_sd)]:
            assert np.isclose(value,expected[r.metric+suffix],atol=1e-10,rtol=0,equal_nan=True),(r.dataset,r.method,r.metric,suffix)
            checked+=1
        if r.dataset!=report.FASHION:
            assert r.valid==expected[r.metric+"_valid_seeds"]
    gold_primary=report.read(SOURCE/"analysis/primary_comparisons_holm.json")
    for r in primary.itertuples():
        expected=next(x for x in gold_primary if x["dataset"]==r.dataset)
        for field in ("mean_delta","ci95_low","ci95_high"):
            assert np.isclose(getattr(r,field),expected[field],atol=1e-10,rtol=0)
        assert r.p_raw==expected["sign_flip_p_two_sided"]
        assert r.p_holm_family5==expected["holm_p_family5"]
    # A missing seed must remain visibly incomplete and must not shrink family 5.
    shortened={**payload,"records":[r for r in payload["records"] if not (r["dataset"]=="har" and r["method"]=="ada_im" and r["seed"]==9)]}
    f,p,_,_=report.make_tables(shortened)
    _,tests=report.comparisons(f,shortened)
    assert not p[p.dataset=="har"].iloc[0].complete
    assert len(tests)==5 and np.isnan(tests[tests.dataset=="har"].iloc[0].p_holm_family5)
    assert report.paired([1.]*10)["p_raw"]==2/1024
    original={str(p):report.sha(p) for base in [SOURCE,PROJECT/"DCSS_FashionMNIST_Full"] for p in base.rglob("*") if p.is_file()}
    path=ROOT/"ThongKe_DCSS_MultiDataset.ipynb"
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    try:
        NotebookClient(notebook,timeout=240,kernel_name="python3",
            resources={"metadata":{"path":str(PROJECT)}},allow_errors=False).execute()
    except Exception:
        nbformat.write(notebook,ROOT/"audit/statistics_failed_diagnostic.ipynb")
        errors=[(i,o.get("ename"),o.get("evalue")) for i,c in enumerate(notebook.cells)
                for o in c.get("outputs",[]) if o.get("output_type")=="error"]
        print("Execution errors:",errors,flush=True)
        raise RuntimeError("Notebook validation failed; see diagnostic notebook") from None
    nbformat.write(notebook,path)
    assert all(report.sha(p)==value for p,value in original.items()),"A source result changed"
    output=PROJECT/"DCSS_MultiDataset_Statistics"
    assert len(list(output.glob("*.png")))==7
    assert (output/"ThongKe_DCSS_MultiDataset.html").exists()
    assert (output/"ThongKe_DCSS_MultiDataset.zip").exists()
    delivered=report.read(path)
    assert not [o for c in delivered["cells"] for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    result=dict(golden_mean_sd_values_checked=checked,primary_comparisons_checked=5,missing_seed_handling=True,
        all_cells_executed=True,code_cells=sum(c.cell_type=="code" for c in notebook.cells),gpu_required=False,
        readonly_source_files_verified=len(original),record_count=300,new_results=250,epoch_records=25000,
        figures=7,notebook_sha256=report.sha(path),notebook_bytes=path.stat().st_size,output_directory=str(output))
    (ROOT/"audit/statistics_notebook_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
