"""Build a self-contained Colab without requiring access to the local repository."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat as nb

root = Path(__file__).resolve().parent
source = (root / "experiment.py").read_text(encoding="utf-8")
protocol = (root / "PROTOCOL.md").read_text(encoding="utf-8")
hashes = json.loads((root / "canonical_hashes.json").read_text())
code_sha = hashlib.sha256(source.encode()).hexdigest()
cells = []


def md(text):
    cells.append(nb.v4.new_markdown_cell(text))


def code(text, hidden=False):
    cell = nb.v4.new_code_cell(text)
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


md("""# DCSS và ADA-IM-DCSS — kiểm chứng Fashion-MNIST đầy đủ

**Trạng thái bàn giao: đã kiểm tra chương trình trên CPU; chưa chạy chiến dịch GPU đầy đủ.**

Chọn **T4 GPU → Runtime → Run all**, cho phép gắn Drive khi được hỏi. Notebook
tự tải dữ liệu, kiểm tra tính toàn vẹn, chạy các seed và lưu kết quả. Không cần
clone repo; không cài lại PyTorch/CUDA. Nếu phiên ngắt, Run all lại cùng cấu hình
để tiếp tục từ cuối epoch đã lưu. Các lượt đã hoàn tất không chạy lại.

Mặc định: `paper70k` (10 seed, 100+100 epoch, 4 phương pháp), sau đó
`legacy10k` và `legacy70k` (mỗi bộ 3 seed, 80+40 epoch, 3 phương pháp).
Toàn chiến dịch gồm **58 lượt phân cụm**, có thể cần nhiều phiên Colab.

**Điểm cần hiểu trước khi chạy:** mã tác giả công bố và phương trình DCSS có
bất nhất. Bộ `paper70k` là tái triển khai theo phương trình với lựa chọn được
ghi rõ bên dưới, chưa thể gọi là tái lập nguyên trạng bảng kết quả bài báo.
70.000 ảnh gồm train và test chính thức: đây là phân cụm chuyển dẫn, không phải
đánh giá trên ảnh chưa từng dùng. Các chỉ số sẽ chỉ xuất hiện sau khi chạy thật.
""")
md(protocol)
md("## 1. Kiểm tra GPU và thư viện\nGiữ nguyên PyTorch/CUDA của phiên hiện tại. Chỉ cài thư viện còn thiếu; dừng nếu không có GPU.")
code('''import os, sys, subprocess, importlib.util
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("Chọn Runtime → Change runtime type → T4 GPU rồi Run all lại. Không chạy thay thế bằng CPU.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
required = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "pandas": "pandas", "matplotlib": "matplotlib", "gdown": "gdown", "threadpoolctl": "threadpoolctl"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
print("Dependencies ready. PyTorch was not replaced.")''')
md("## 2. Gắn Drive và chốt cấu hình\nKết quả nằm trong MyDrive của tài khoản bạn gắn. Giữ nguyên cấu hình để khôi phục đúng lượt đang chạy; đổi cấu hình/môi trường sẽ tạo định danh khác.")
code('''from google.colab import drive
from pathlib import Path
import json, hashlib, zipfile
drive.mount("/content/drive")
OUTPUT_ROOT = Path("/content/drive/MyDrive/DCSS_FashionMNIST_Full")
DATA_ROOT = Path("/content/fashion_mnist_verified")
RUN_PROFILES = ["paper70k", "legacy10k", "legacy70k"]
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
# Kiểm tra quyền ghi trước khi chạy dài.
probe = OUTPUT_ROOT / ".write_probe"
probe.write_text("ok", encoding="utf-8")
assert probe.read_text(encoding="utf-8") == "ok"
probe.unlink()
print("Kết quả sẽ được lưu tại:", OUTPUT_ROOT)
print("Các hồ sơ sẽ chạy:", RUN_PROFILES)''')
md("## 3. Mã thực thi được chốt cùng notebook\nÔ dưới ghi module vào /content. Có thể mở rộng ô để kiểm tra. Module không dùng nhãn trong hàm huấn luyện.")
code("%%writefile /content/dcss_full_experiment.py\n" + source, hidden=True)
code(f'''import importlib, sys
if "/content" not in sys.path:
    sys.path.insert(0, "/content")
import dcss_full_experiment as exp
importlib.reload(exp)
EXPECTED_CODE_SHA256 = {code_sha!r}
assert exp.digest(exp.__file__) == EXPECTED_CODE_SHA256, "Mã thực thi bị thay đổi so với notebook đã chốt."
CANONICAL_HASHES = {hashes!r}
PROTOCOL_TEXT = {protocol!r}
(OUTPUT_ROOT / "PROTOCOL.md").write_text(PROTOCOL_TEXT, encoding="utf-8")
exp.atomic_json(OUTPUT_ROOT / "canonical_hashes.json", CANONICAL_HASHES)
print("Code SHA256:", EXPECTED_CODE_SHA256)
exp.technical_smoke("cuda")''')
md("## 4. Tải và xác minh đủ 70.000 ảnh\nƯu tiên bốn tệp trong Drive đã cung cấp. Nguồn thay thế chính thức, nếu cần, phải khớp SHA-256. Sai header, thiếu ảnh hoặc sai checksum đều dừng.")
code('''exp.prepare_data(DATA_ROOT, CANONICAL_HASHES)
images70, labels70, metadata70 = exp.load_data(DATA_ROOT, CANONICAL_HASHES, "all70k")
print("Shape:", tuple(images70.shape), "dtype:", images70.dtype)
print("Số mẫu mỗi lớp (chỉ kiểm tra dữ liệu):", metadata70["class_counts_audit_only"])
assert len(images70) == 70000
exp.atomic_json(OUTPUT_ROOT / "verified_dataset.json", metadata70)
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
sample_ids = np.random.default_rng(20261109).choice(len(images70), 20, replace=False)
for axis, index in zip(axes.ravel(), sample_ids):
    axis.imshow(images70[index, 0], cmap="gray")
    axis.axis("off")
fig.suptitle("Ảnh kiểm tra dữ liệu; nhãn không đưa vào huấn luyện")
plt.show()''')
md("""## 5. Chạy thực nghiệm đã chốt

Chỉ ô này thực hiện huấn luyện đầy đủ. Sau mỗi seed/phương pháp có kết quả
cuối; sau mỗi epoch có checkpoint. Không chọn điểm tốt nhất theo nhãn.
Phải chờ `complete=true` cho từng hồ sơ. Nếu lỗi/NaN/hết VRAM, ô dừng và lưu
trạng thái lỗi; không tự thay đổi batch hay dùng thuật toán thay thế.

Khi tiếp tục, kiểm tra dòng `RESULT DIRECTORY` có trùng thư mục cũ. Nếu môi
trường khác, mã định danh cũng khác; không ghép kết quả của hai môi trường.
""")
code('''completed_roots = {}
for name in RUN_PROFILES:
    cfg = exp.profile_config(name)
    if cfg.split == "all70k":
        images, labels, metadata = images70, labels70, metadata70
    else:
        images, labels, metadata = exp.load_data(DATA_ROOT, CANONICAL_HASHES, cfg.split)
    print("\\nĐANG CHẠY HỒ SƠ:", name, "N=", len(images), "seeds=", cfg.seeds)
    result_root = exp.run_experiment(images, labels, metadata, cfg, OUTPUT_ROOT)
    completed_roots[name] = str(result_root)
    exp.atomic_json(OUTPUT_ROOT / "latest_completed_roots.json", completed_roots)
    from IPython.display import display
    import pandas as pd
    display(pd.read_csv(result_root / "summary.csv"))
    exp.plots(result_root)
print("Đã hoàn thành các hồ sơ đã chọn. Xem ô phân tích và gói kết quả bên dưới.")''')
md("""## 6. Đọc kết quả và kiểm tra tác động tăng dữ liệu

CSV/JSON lưu chỉ số dạng tỷ lệ, ví dụ 0,45 = 45%. Báo cáo đủ mọi seed, gồm cả
lượt suy biến. Kiểm định chính là ARI của ADA-IM trừ IM trên `paper70k`; những
so sánh khác là bổ sung. Khoảng tin cậy chứa 0 nghĩa là chưa có bằng chứng rõ
về chiều cải thiện với phép đo này; không kết luận chỉ từ seed tốt nhất.

Bảng 10k → 70k dưới đây dùng các lượt **chạy mới cùng bộ thực thi**, không
ghép điểm cũ vốn khác lịch ngẫu nhiên. Đây là kiểm tra quy mô dữ liệu trên các
tập có phần giao nhau, không phải đánh giá độc lập hoặc kiểm định chuyển miền.
""")
code('''import pandas as pd
from IPython.display import display
for name, path in completed_roots.items():
    root = Path(path)
    completeness = json.loads((root / "completeness.json").read_text())
    print(name, completeness)
    assert completeness["complete"], "Thiếu lượt chạy: chưa được coi là kết quả cuối."
    pairs = pd.read_json(root / "paired_comparisons.json")
    display(pairs[["comparison", "metric", "n", "mean_delta", "ci95_low", "ci95_high", "sign_flip_p_two_sided", "confirmatory"]])
if "legacy10k" in completed_roots and "legacy70k" in completed_roots:
    small = pd.read_csv(Path(completed_roots["legacy10k"]) / "per_seed.csv")
    full = pd.read_csv(Path(completed_roots["legacy70k"]) / "per_seed.csv")
    scale = small[["seed", "method", "ARI", "NMI", "ACC"]].merge(full[["seed", "method", "ARI", "NMI", "ACC"]], on=["seed", "method"], suffixes=("_10k", "_70k"), validate="one_to_one")
    for metric in ["ARI", "NMI", "ACC"]:
        scale[metric + "_delta_70k_minus_10k"] = scale[metric + "_70k"] - scale[metric + "_10k"]
    scale.to_csv(OUTPUT_ROOT / "scale_comparison_paired_seeds.csv", index=False)
    display(scale)
print("Không tự động kết luận phương pháp tốt hơn. Đọc từng seed, chênh lệch, CI, suy biến và thời gian.")''')
md("## 7. Xuất gói kết quả để gửi kiểm tra\nGói nhẹ chứa cấu hình, mã, CSV/JSON, biểu đồ và lịch sử. Checkpoint/embedding vẫn nằm trong từng thư mục trên Drive để tránh tạo ZIP quá lớn.")
code('''archive = OUTPUT_ROOT / "results_for_review.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    shared_names = ["PROTOCOL.md", "canonical_hashes.json", "verified_dataset.json", "latest_completed_roots.json"]
    if "legacy10k" in completed_roots and "legacy70k" in completed_roots:
        shared_names.append("scale_comparison_paired_seeds.csv")
    for name in shared_names:
        path = OUTPUT_ROOT / name
        if path.exists():
            bundle.write(path, path.relative_to(OUTPUT_ROOT))
    for result_path in completed_roots.values():
        for path in Path(result_path).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv", ".png", ".py"}:
                bundle.write(path, path.relative_to(OUTPUT_ROOT))
print("Gói kết quả:", archive)
print("Kích thước MB:", round(archive.stat().st_size / 1024**2, 2))
print("Bạn có thể lấy ZIP từ Google Drive và gửi lại để phân tích. Không cần gửi toàn bộ checkpoint.")''')

notebook = nb.v4.new_notebook(cells=cells, metadata={
    "colab": {"provenance": [], "gpuType": "T4", "name": "ADA-IM-DCSS"},
    "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "dcss_protocol": {"version": "2026-11-09-v1", "code_sha256": code_sha, "gpu_research_run_performed_before_delivery": False},
})
for i, cell in enumerate(notebook.cells):
    cell.id = f"dcss-{i:02d}"
    if cell.cell_type == "code":
        text = cell.source
        if text.startswith("%%writefile"):
            text = text.split("\n", 1)[1]
        ast.parse(text)
nb.validate(notebook)
path = root / "DCSS_ADA_IM_FashionMNIST_Full.ipynb"
nb.write(notebook, path)
print(json.dumps({"file": str(path), "cells": len(cells), "code_sha256": code_sha, "bytes": path.stat().st_size}, indent=2))
