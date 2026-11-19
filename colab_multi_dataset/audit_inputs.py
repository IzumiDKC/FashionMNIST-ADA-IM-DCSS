"""Local verification against the exact user-provided Drive files (not training)."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import dcss_core as core
import multi_data as data

ROOT=Path(__file__).parent


def main():
    discovery=json.loads((ROOT/"audit/drive_catalog.json").read_text(encoding="utf-8"))
    flat={f["name"]:f for key,files in discovery.items() if key not in {"old_results","destination"} for f in files}
    catalog={}
    for key in ("emnist","har","pendigits","waveform","personas"):
        rows=[]
        for p in sorted((ROOT/"data/raw"/key).iterdir()):
            if p.suffix==".download":
                continue
            f=flat[p.name]
            assert p.stat().st_size==int(f["size"]),(key,p.name)
            rows.append(dict(name=p.name,id=f["id"],size=p.stat().st_size,sha256=core.digest(p)))
        catalog[key]=rows
    core.atomic_json(ROOT/"sources.json",catalog)
    report={"type":"data_integrity_not_experimental_results","datasets":{}}
    for key in ("emnist","har","pendigits","waveform"):
        raw=data.ensure_sources(key,ROOT/"data/raw",catalog)
        x,y,meta=data.load_labeled(key,raw,catalog)
        report["datasets"][key]=meta
        print(key,tuple(x.shape),np.bincount(y).tolist(),flush=True)
    raw=data.ensure_sources("personas",ROOT/"data/raw",catalog)
    texts,meta=data.sample_personas(raw,ROOT/"data/personas_sample100k",catalog)
    report["datasets"]["personas"]={**meta,"empty_texts":sum(not t.strip() for t in texts),
        "schema":pq.ParquetFile(raw/catalog["personas"][0]["name"]).schema.names}
    print("personas",len(texts),meta["global_indices_sha256"],flush=True)
    core.atomic_json(ROOT/"audit/data_verification.json",report)


if __name__=="__main__":
    main()
