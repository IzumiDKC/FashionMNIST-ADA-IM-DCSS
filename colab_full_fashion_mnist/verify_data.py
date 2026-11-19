"""Verify supplied Drive bytes against official Fashion-MNIST release checksums."""
import base64
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from experiment import FILES, digest, load_data, atomic_json

root = Path(__file__).resolve().parent
official = root / "data" / "official"
official.mkdir(parents=True, exist_ok=True)
records = {}
for name, (file_id, md5) in FILES.items():
    url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name + ".gz"
    path = official / (name + ".gz")
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    compressed = path.read_bytes()
    assert hashlib.md5(compressed).hexdigest() == md5, name
    raw_sha = hashlib.sha256(gzip.decompress(compressed)).hexdigest()
    assert digest(root / "data" / name) == raw_sha, name
    records[name] = {"drive_id": file_id, "official_url": url, "official_gz_md5": md5,
                     "raw_sha256": raw_sha, "size": (root / "data" / name).stat().st_size, "drive_equals_official": True}
hashes = {name: r["raw_sha256"] for name, r in records.items()}
x, y, meta = load_data(root / "data", hashes)
atomic_json(root / "canonical_hashes.json", hashes)
atomic_json(root / "audit" / "dataset_verification.json", {"files": records, "data": meta})
drive_manifest = root / "audit" / "drive_files.json"
if drive_manifest.exists():
    info = json.loads(drive_manifest.read_text(encoding="utf-8"))
    (root / "audit" / "original_colab.ipynb").write_bytes(base64.b64decode(info["notebook"]["original_b64"]))
print(json.dumps({"shape": list(x.shape), "classes": meta["class_counts_audit_only"], "all_four_files_equal_official": True}, indent=2))
