"""Verified inputs. Labels are audit/evaluation data, never model inputs."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import os
import struct
from pathlib import Path
import numpy as np
import torch
import dcss_core as core

DATA_VERSION = "multi-data-2026-11-11-v1"
SAMPLE_SEED = 20261111
PERSONA_N = 100_000
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBED_SPEC = dict(model=EMBED_MODEL, revision=EMBED_REVISION, column="persona",
                  max_length=128, pooling="attention_mask_mean", l2_normalized=True,
                  dtype="float32", batch_size=64, chunk_size=512, attention="eager")


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def drive_mounted_path(service, file_id, mount="/content/drive/MyDrive"):
    """Resolve the supplied folder ID, not a guessed title elsewhere in Drive."""
    root_id = service.files().get(fileId="root", fields="id").execute()["id"]
    names, seen = [], set()
    while file_id != root_id:
        if file_id in seen:
            raise ValueError("Drive parent cycle")
        seen.add(file_id)
        m = service.files().get(fileId=file_id, fields="id,name,parents,mimeType").execute()
        names.append(m["name"])
        parents = m.get("parents", [])
        if len(parents) != 1:
            raise ValueError("Folder is not under the authenticated MyDrive; add/mount it explicitly.")
        file_id = parents[0]
    p = Path(mount).joinpath(*reversed(names))
    if not p.is_dir():
        raise FileNotFoundError(f"Verified Drive folder is not mounted: {p}")
    return p


def ensure_sources(key, raw_root, catalog, service=None):
    """Download by authenticated Drive ID; commit only independently pinned bytes."""
    root = Path(raw_root) / key
    root.mkdir(parents=True, exist_ok=True)
    files = catalog[key]
    for f in files:
        path = root / f["name"]
        if not path.exists():
            if service is None:
                raise FileNotFoundError(path)
            from googleapiclient.http import MediaIoBaseDownload
            tmp = path.with_name(path.name + ".download")
            print("Downloading", key, f["name"], flush=True)
            with tmp.open("wb") as dst:
                downloader = MediaIoBaseDownload(dst, service.files().get_media(fileId=f["id"]), chunksize=8*1024**2)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=5)
            if tmp.stat().st_size != f["size"] or core.digest(tmp) != f["sha256"]:
                raise ValueError("Downloaded file differs from audited Drive bytes: " + f["name"])
            os.replace(tmp, path)
        if path.stat().st_size != f["size"] or core.digest(path) != f["sha256"]:
            raise ValueError("Source cache changed/corrupt: " + str(path))
    return root


def idx_gz(path):
    b = gzip.decompress(Path(path).read_bytes())
    if len(b) < 8:
        raise ValueError("Truncated IDX")
    magic, n = struct.unpack(">II", b[:8])
    if magic == 2051 and len(b) >= 16:
        h, w = struct.unpack(">II", b[8:16])
        if (h, w) != (28, 28) or len(b) != 16+n*h*w:
            raise ValueError("Invalid image IDX dimensions/length")
        return np.frombuffer(b, np.uint8, offset=16).reshape(n, 1, h, w).copy()
    if magic == 2049 and len(b) == n+8:
        return np.frombuffer(b, np.uint8, offset=8).copy()
    raise ValueError("Invalid label IDX magic/length")


def fixed_indices(n, count=2000):
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, min(count, n), replace=False))


def load_labeled(key, root, catalog):
    root = Path(root)
    extra = {}
    if key == "emnist":
        prefix = "emnist-balanced-"
        xs = [idx_gz(root / (prefix + split + "-images-idx3-ubyte.gz")) for split in ("train", "test")]
        ys = [idx_gz(root / (prefix + split + "-labels-idx1-ubyte.gz")) for split in ("train", "test")]
        if [len(x) for x in xs] != [112800, 18800] or [len(y) for y in ys] != [112800, 18800]:
            raise ValueError("EMNIST Balanced must have all 112800 + 18800 examples")
        # Stored EMNIST IDX pixels use transposed image axes. Fixed for both splits.
        x, y = np.concatenate(xs).transpose(0, 1, 3, 2).copy(), np.concatenate(ys).astype(np.int64)
        n, d, k = 131600, 784, 47
        if not np.array_equal(np.bincount(y, minlength=k), np.full(k, 2800)):
            raise ValueError("Balanced class counts differ")
        norm = "transpose_image_axes; float32(pixel)/255 per batch"
        extra = {"original_split_sizes": [112800, 18800], "variant": "EMNIST Balanced"}
    elif key == "har":
        xs = [np.loadtxt(root / f"X_{s}.txt", dtype=np.float32) for s in ("train", "test")]
        ys = [np.loadtxt(root / f"y_{s}.txt", dtype=np.int64) for s in ("train", "test")]
        subjects = [np.loadtxt(root / f"subject_{s}.txt", dtype=np.int64) for s in ("train", "test")]
        if [len(a) for a in xs] != [7352, 2947] or any(len(a) != len(b) or len(a) != len(c) for a,b,c in zip(xs,ys,subjects)):
            raise ValueError("HAR original split sizes differ")
        if set(subjects[0]) & set(subjects[1]):
            raise ValueError("HAR source train/test subjects overlap")
        x, y = np.concatenate(xs), np.concatenate(ys)-1
        n, d, k = 10299, 561, 6
        if x.min() < -1.00001 or x.max() > 1.00001:
            raise ValueError("HAR expected precomputed features in [-1,1]")
        x = (x+1)/2
        norm = "fixed (precomputed_feature+1)/2; subject IDs excluded"
        extra = {"original_split_sizes": [7352, 2947], "original_subjects_disjoint": True,
                 "subject_count": len(np.unique(np.concatenate(subjects)))}
    elif key == "pendigits":
        a, b = [np.loadtxt(root / f"pendigits.{s}", delimiter=",", dtype=np.float32) for s in ("tra", "tes")]
        if a.shape != (7494, 17) or b.shape != (3498, 17):
            raise ValueError("Pendigits split sizes/dimensions differ")
        full = np.concatenate([a, b])
        if not np.all(full[:, -1] == full[:, -1].astype(int)):
            raise ValueError("Non-integer labels")
        x, y = full[:, :-1], full[:, -1].astype(np.int64)
        if x.min() < 0 or x.max() > 100:
            raise ValueError("Pendigits coordinates outside [0,100]")
        x = x/100
        n, d, k = 10992, 16, 10
        norm = "fixed coordinate/100"
        extra = {"original_split_sizes": [7494, 3498]}
    elif key == "waveform":
        from unlzw3 import unlzw
        a = np.loadtxt(io.BytesIO(unlzw((root/"waveform.data.Z").read_bytes())), delimiter=",", dtype=np.float32)
        if a.shape != (5000, 22) or not np.all(a[:, -1] == a[:, -1].astype(int)):
            raise ValueError("Expected Waveform V1 5000 x (21 features + label)")
        x, y = a[:, :-1], a[:, -1].astype(np.int64)
        lo, span = x.min(0), np.ptp(x, axis=0)
        x = (x-lo)/np.where(span > 0, span, 1)
        n, d, k = 5000, 21, 3
        norm = "per_feature_minmax fitted to all unlabeled transductive inputs"
        extra = {"min": lo.tolist(), "range": span.tolist(), "variant": "21 features; no extra 19 noise features"}
    else:
        raise ValueError(key)
    x = np.ascontiguousarray(x)
    if x.shape[0] != n or int(np.prod(x.shape[1:])) != d or len(y) != n or not np.isfinite(x).all():
        raise ValueError("Invalid/non-finite data dimensions")
    if not np.array_equal(np.unique(y), np.arange(k)):
        raise ValueError("Unexpected class IDs")
    meta = dict(dataset=key, version=DATA_VERSION, n=n, input_dim=d, n_clusters=k,
                labeled=True, design="transductive_all_inputs; labels_final_evaluation_only",
                order="train_then_test" if key != "waveform" else "source_file_order",
                normalization=norm, features_sha256=array_hash(x), labels_sha256=array_hash(y),
                source_sha256={f["name"]: f["sha256"] for f in catalog[key]},
                class_counts_audit_only=np.bincount(y).tolist(), **extra)
    return torch.from_numpy(x), y, meta


def sample_personas(raw_root, cache_root, catalog, count=PERSONA_N):
    """Uniform sample across ALL shards, row IDs fixed before examining text."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, cache = Path(raw_root), Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(catalog["personas"], key=lambda r:r["name"])
    counts = [pq.ParquetFile(root/f["name"]).metadata.num_rows for f in files]
    if len(files) != 11 or sum(counts) != 1_000_000:
        raise ValueError("Expected all 11 shards and 1,000,000 source personas")
    ids = np.sort(np.random.default_rng(SAMPLE_SEED).choice(sum(counts), count, replace=False))
    spec = dict(version=DATA_VERSION, n=count, seed=SAMPLE_SEED, global_indices_sha256=array_hash(ids),
                shards=[dict(name=f["name"], sha256=f["sha256"], rows=n) for f,n in zip(files,counts)])
    spec_path = cache/"sample_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Sample cache identity mismatch; choose a new cache root")
    core.atomic_json(spec_path, spec)
    offset, selected_paths = 0, []
    for f, n in zip(files, counts):
        local = ids[(ids >= offset) & (ids < offset+n)]-offset
        dest = cache/(f["name"].replace("train-", "sample-"))
        marker = dest.with_suffix(".json")
        if marker.exists():
            saved = json.loads(marker.read_text())
            if not dest.exists() or core.digest(dest) != saved["sha256"] or saved["sample_spec_hash"] != core.hash_json(spec):
                raise ValueError("Corrupt selected-text cache: " + str(dest))
        else:
            parts, cursor = [], 0
            for batch in pq.ParquetFile(root/f["name"]).iter_batches(batch_size=4096, columns=["uuid", "persona"]):
                pick = local[(local >= cursor) & (local < cursor+len(batch))]-cursor
                if len(pick):
                    table = pa.Table.from_batches([batch]).take(pa.array(pick))
                    table = table.append_column("global_row", pa.array(pick+cursor+offset))
                    parts.append(table)
                cursor += len(batch)
            table = pa.concat_tables(parts)
            if table.num_rows != len(local):
                raise ValueError("Sample count mismatch")
            tmp = dest.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, dest)
            core.atomic_json(marker, dict(sha256=core.digest(dest), sample_spec_hash=core.hash_json(spec)))
        selected_paths.append(dest)
        offset += n
    selected = pa.concat_tables([pq.read_table(p) for p in selected_paths]).to_pydict()
    if selected["global_row"] != ids.tolist() or len(set(selected["uuid"])) != count:
        raise ValueError("Sample IDs/order/UUID uniqueness mismatch")
    if any(not isinstance(s, str) or not s.strip() for s in selected["persona"]):
        raise ValueError("Missing text in prespecified sample; do not silently replace samples")
    # Store identifiers for an independent audit without creating artificial labels.
    identity = {"global_rows": selected["global_row"], "uuids": selected["uuid"]}
    core.atomic_json(cache/"sample_ids.json", identity)
    return selected["persona"], {**spec, "identifiers_sha256": core.hash_json(identity),
                                  "texts_sha256": core.hash_json(selected["persona"])}


def embed_personas(texts, sample_meta, cache_root, device, deadline=None):
    """Frozen encoder; verified chunks persist across sessions and CUDA/CPU changes."""
    import importlib.metadata
    import time
    from transformers import AutoTokenizer, AutoModel
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    spec = {"embedding": EMBED_SPEC, "sample": sample_meta}
    spec_hash = core.hash_json(spec)
    path = cache/"embedding_spec.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Embedding cache identity mismatch")
    core.atomic_json(path, spec)
    target, final_marker = cache/"features.npy", cache/"features.json"
    if final_marker.exists():
        m = json.loads(final_marker.read_text())
        if m["spec_hash"] != spec_hash or not target.exists() or core.digest(target) != m["sha256"]:
            raise ValueError("Corrupt final embedding cache")
        return np.load(target), m
    core.seed_all(SAMPLE_SEED)
    tokenizer = model = None
    chunk_paths, provenance, truncated, total_tokens = [], [], 0, 0
    for start in range(0, len(texts), EMBED_SPEC["chunk_size"]):
        if deadline and time.monotonic() >= deadline:
            raise core.PauseRequested("Embedding paused; verified chunks retained")
        stop = min(start+EMBED_SPEC["chunk_size"], len(texts))
        chunk = cache/f"chunk_{start:06d}.npy"
        marker = chunk.with_suffix(".json")
        if marker.exists():
            m = json.loads(marker.read_text())
            if m["spec_hash"] != spec_hash or m["start"] != start or m["stop"] != stop or core.digest(chunk) != m["sha256"]:
                raise ValueError("Corrupt embedding chunk: " + str(chunk))
        else:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION, cache_dir=str(cache/"model"))
                model = AutoModel.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION, cache_dir=str(cache/"model"),
                                                   attn_implementation="eager", use_safetensors=True).to(device).eval()
            vectors, n_trunc, n_tokens = [], 0, 0
            for b in range(start, stop, EMBED_SPEC["batch_size"]):
                batch_text = texts[b:min(b+EMBED_SPEC["batch_size"], stop)]
                lengths = [len(t) for t in tokenizer(batch_text, truncation=False, padding=False)["input_ids"]]
                n_trunc += sum(n > EMBED_SPEC["max_length"] for n in lengths)
                n_tokens += sum(lengths)
                tokens = tokenizer(batch_text, padding=True, truncation=True, max_length=EMBED_SPEC["max_length"], return_tensors="pt").to(device)
                with torch.inference_mode():
                    hidden = model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    z = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                    z = torch.nn.functional.normalize(z, p=2, dim=1)
                vectors.append(z.cpu().float().numpy())
            a = np.concatenate(vectors)
            if a.shape != (stop-start, 384) or not np.isfinite(a).all():
                raise ValueError("Invalid text embeddings")
            tmp = chunk.with_suffix(".npy.tmp")
            with tmp.open("wb") as stream:
                np.save(stream, a, allow_pickle=False)
            os.replace(tmp, chunk)
            m = dict(spec_hash=spec_hash, start=start, stop=stop, sha256=core.digest(chunk),
                     truncated_rows=n_trunc, original_token_count=n_tokens, environment=core.environment(torch.device(device)),
                     transformers=importlib.metadata.version("transformers"), tokenizer=importlib.metadata.version("tokenizers"))
            core.atomic_json(marker, m)
            print(f"Text features committed: {stop}/{len(texts)}", flush=True)
        chunk_paths.append(chunk)
        provenance.append(m)
        truncated += m["truncated_rows"]
        total_tokens += m["original_token_count"]
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tmp = target.with_suffix(".npy.tmp")
    full = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(texts),384))
    for p, m in zip(chunk_paths, provenance):
        full[m["start"]:m["stop"]] = np.load(p, allow_pickle=False)
    full.flush()
    del full
    os.replace(tmp, target)
    core.atomic_json(cache/"embedding_execution.json", provenance)
    result = dict(spec_hash=spec_hash, sha256=core.digest(target), n=len(texts), dimension=384,
                  truncated_rows=truncated, original_token_count=total_tokens, embedding_spec=EMBED_SPEC,
                  mixed_environment=len({core.hash_json({k:m[k] for k in ("environment","transformers","tokenizer")}) for m in provenance}) > 1)
    core.atomic_json(final_marker, result)
    return np.load(target), result


def load_personas(raw_root, cache_root, catalog, device, deadline=None):
    texts, sample_meta = sample_personas(raw_root, Path(cache_root)/"sample100k", catalog)
    features, embedding = embed_personas(texts, sample_meta, Path(cache_root)/"embeddings100k", device, deadline)
    meta = dict(dataset="personas", version=DATA_VERSION, n=len(features), input_dim=384, n_clusters=20,
                labeled=False, design="exploratory_transductive_synthetic_text_no_ground_truth", normalization="frozen_L2_embedding",
                features_sha256=array_hash(features), sample=sample_meta, embedding=embedding,
                k_policy="20 prespecified exploratory groups; not a known true class count")
    return torch.from_numpy(features), None, meta
