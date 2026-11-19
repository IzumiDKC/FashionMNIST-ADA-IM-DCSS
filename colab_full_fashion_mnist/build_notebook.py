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

**Bản tiếp tục đa thiết bị — 10/11/2026.** Giữ nguyên giao thức đã chạy trên T4.

Chọn **Python 3 → GPU A100 → phiên bản mới nhất** như ảnh cấu hình của bạn;
RAM cao cấp có thể để tắt. Sau đó **Runtime → Run all**, cho phép gắn Drive.
Nếu A100 không được cấp, chọn L4/T4/G4 hoặc GPU CUDA khác đang có. Notebook
tự tải dữ liệu, kiểm tra tính toàn vẹn, chạy các seed và lưu kết quả. Không cần
clone repo; không cài lại PyTorch/CUDA. Có CPU fallback bằng **cùng mạng PyTorch**,
nhưng sẽ chậm hơn. Nếu phiên ngắt, chọn runtime mới rồi Run all để tiếp tục từ
checkpoint gần nhất. Các lượt đã hoàn tất không chạy lại.

**Tự tìm lại checkpoint T4:** giữ nguyên thư mục `DCSS_FashionMNIST_Full` trên
cùng tài khoản Drive. Mã tự nhận bản V1 tương thích và giữ nguyên thư mục/định
danh cũ. Checkpoint V1 được giữ lại; V2 thêm hai slot dự phòng có SHA-256.
Đổi GPU không đổi tên lượt chạy. Nếu có nhiều lượt tương thích, notebook liệt
kê để bạn chỉ định thư mục, không chọn theo điểm tốt nhất.

**Giới hạn:** notebook không tự xin GPU mới khi Colab thu hồi phiên và không
vượt hạn mức thuê bao. Bạn cần chọn GPU còn được cấp rồi Run all. Chuyển thiết
bị/phần mềm không bảo đảm kết quả trùng từng bit; mọi phiên được ghi lại.
TPU không thuộc bộ thực thi PyTorch CUDA/CPU này. Chỉ mở một phiên ghi cùng
thư mục kết quả tại một thời điểm.

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
md("## 1. Nhận diện A100/GPU khác hoặc CPU\nGiữ PyTorch/CUDA được Colab cài sẵn. Không giới hạn tên GPU. Thiếu GPU được báo rõ, không gây lỗi bắt buộc T4 như V1.")
code('''import os, sys, subprocess, importlib.util
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU được Colab cấp:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME"):
        raise RuntimeError("Đang chọn TPU. Hãy đổi sang GPU A100/L4/T4 hoặc CPU; notebook này không dùng XLA.")
    print("Đang ở CPU. Có thể tiếp tục checkpoint bằng CPU; sẽ chậm. Muốn dùng A100, đổi loại runtime trước khi Run all.")
required = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "pandas": "pandas", "matplotlib": "matplotlib", "gdown": "gdown", "threadpoolctl": "threadpoolctl"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
print("Dependencies ready. PyTorch was not replaced.")''')
md("## 2. Gắn đúng Drive và chính sách tiếp tục\nĐổi thiết bị không đổi cấu hình thí nghiệm. Giữ nguyên OUTPUT_ROOT để tìm checkpoint T4 cũ. Kiểm tra dung lượng Drive; checkpoint còn lưu được mới có thể khôi phục.")
code('''from google.colab import drive
from pathlib import Path
import json, hashlib, zipfile
drive.mount("/content/drive")
OUTPUT_ROOT = Path("/content/drive/MyDrive/DCSS_FashionMNIST_Full")
DATA_ROOT = Path("/content/fashion_mnist_verified")
RUN_PROFILES = ["paper70k", "legacy10k", "legacy70k"]
DEVICE = "auto"               # GPU CUDA nếu có, nếu không thì CPU.
ALLOW_CPU_FALLBACK = True      # Đặt False nếu chỉ muốn chờ GPU.
ALLOW_SOFTWARE_CHANGE = True   # Cho tiếp tục khi Colab đổi phiên bản; luôn ghi nhật ký.
CHECKPOINT_SECONDS = 300       # Ngoài cuối mỗi epoch, lưu khi qua 5 phút sau một bước tối ưu.
MAX_SESSION_HOURS = 0          # 0 = không đặt giới hạn riêng. Ví dụ 4 = tự lưu rồi tạm dừng sau ~4h.
RESUME_FOLDERS = {}            # Ví dụ {"paper70k": "paper70k_2cd93c725947"} nếu có nhiều lượt tương thích.
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
if "OUTPUT_ROOT" not in globals():
    raise RuntimeError("Chưa chạy ô 2 để gắn Drive/cấu hình. Chọn Run all từ đầu; không chạy riêng ô này.")
(OUTPUT_ROOT / "PROTOCOL_portable_v2.md").write_text(PROTOCOL_TEXT, encoding="utf-8")
exp.atomic_json(OUTPUT_ROOT / "canonical_hashes.json", CANONICAL_HASHES)
print("Code SHA256:", EXPECTED_CODE_SHA256)
runtime_policy = exp.RuntimeOptions(device=DEVICE, allow_cpu=ALLOW_CPU_FALLBACK,
    checkpoint_seconds=CHECKPOINT_SECONDS, max_session_hours=MAX_SESSION_HOURS,
    allow_software_change=ALLOW_SOFTWARE_CHANGE)
active_device = exp.select_device(runtime_policy)
print("Thiết bị dùng:", active_device)
from IPython.display import display
import pandas as pd
inventory = exp.progress_inventory(OUTPUT_ROOT)
if inventory:
    print("Tiến độ đã lưu trên Drive (chưa tải lại model):")
    display(pd.DataFrame(inventory))
else:
    print("Chưa tìm thấy tiến độ trong OUTPUT_ROOT. Nếu đã chạy T4, kiểm tra đúng tài khoản Drive/thư mục trước khi chạy ô 5.")
runtime_validation = exp.portability_smoke(active_device)
exp.atomic_json(OUTPUT_ROOT / "runtime_validation_latest.json", runtime_validation)
print("Kiểm tra checkpoint/optimizer trên runtime hiện tại đã qua; đây không phải số liệu luận văn.")''')
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
cuối; sau mỗi epoch và định kỳ giữa epoch có checkpoint. Không chọn điểm tốt nhất theo nhãn.
Phải chờ `complete=true` cho từng hồ sơ. Nếu lỗi/NaN/hết VRAM, ô dừng và lưu
trạng thái lỗi; không tự thay đổi batch hay dùng thuật toán thay thế.

Khi tiếp tục, kiểm tra `RESULT DIRECTORY` và dòng `Resume ... next batch ...`.
Đổi GPU/CPU/phần mềm được ghi trong `sessions/`; vẫn dùng cùng lượt nếu cấu
hình khoa học khớp. Thay batch, hàm mục tiêu hoặc dữ liệu bị từ chối tiếp tục
chung. Nếu A100 ngắt giữa batch, chỉ làm lại phần sau checkpoint đã ghi.
Khi hết giới hạn thời gian tự đặt, ô tạm dừng; có thể xuất kết quả đang có ở ô 7.
""")
code('''if "images70" not in globals() or "runtime_policy" not in globals():
    raise RuntimeError("Thiếu dữ liệu/cấu hình: chạy Run all từ đầu.")
from dataclasses import replace
completed_roots = {}
session_deadline = exp.time.monotonic() + MAX_SESSION_HOURS * 3600 if MAX_SESSION_HOURS > 0 else None
paused = False
for name in RUN_PROFILES:
    cfg = exp.profile_config(name)
    if cfg.split == "all70k":
        images, labels, metadata = images70, labels70, metadata70
    else:
        images, labels, metadata = exp.load_data(DATA_ROOT, CANONICAL_HASHES, cfg.split)
    print("\\nĐANG CHẠY HỒ SƠ:", name, "N=", len(images), "seeds=", cfg.seeds)
    remaining_hours = max(0., (session_deadline - exp.time.monotonic()) / 3600) if session_deadline else 0.
    if session_deadline and remaining_hours <= 0:
        print("Đã hết giới hạn thời gian tự đặt. Run all lần sau để tiếp tục.")
        paused = True
        break
    current_runtime = replace(runtime_policy, resume_folder=RESUME_FOLDERS.get(name), max_session_hours=remaining_hours)
    try:
        result_root = exp.run_experiment(images, labels, metadata, cfg, OUTPUT_ROOT, runtime=current_runtime)
    except exp.PauseRequested as exc:
        print(str(exc))
        paused = True
        break
    completed_roots[name] = str(result_root)
    exp.atomic_json(OUTPUT_ROOT / "latest_completed_roots.json", completed_roots)
    from IPython.display import display
    import pandas as pd
    display(pd.read_csv(result_root / "summary.csv"))
    exp.plots(result_root)
print("Đã tạm dừng có checkpoint." if paused else "Đã hoàn thành các hồ sơ đã chọn.")
print("Có thể xuất gói kết quả đang có ở ô 7.")''')
md("""## 6. Đọc kết quả và kiểm tra tác động tăng dữ liệu

CSV/JSON lưu chỉ số dạng tỷ lệ, ví dụ 0,45 = 45%. Báo cáo đủ mọi seed, gồm cả
lượt suy biến. Kiểm định chính là ARI của ADA-IM trừ IM trên `paper70k`; những
so sánh khác là bổ sung. Nếu đổi GPU/phần mềm, xem cột `mixed_environment` và
`campaign_mixed_environment`; các phép kiểm định được giữ để thăm dò nhưng
không tự gắn kết luận xác nhận cho lượt pha trộn môi trường. Khoảng tin cậy chứa 0 nghĩa là chưa có bằng chứng rõ
về chiều cải thiện với phép đo này; không kết luận chỉ từ seed tốt nhất.

Bảng 10k → 70k dưới đây dùng các lượt **chạy mới cùng bộ thực thi**, không
ghép điểm cũ vốn khác lịch ngẫu nhiên. Đây là kiểm tra quy mô dữ liệu trên các
tập có phần giao nhau, không phải đánh giá độc lập hoặc kiểm định chuyển miền.
""")
code('''import pandas as pd
from IPython.display import display
if "completed_roots" not in globals():
    raise RuntimeError("Chưa chạy ô 5. Nếu chỉ muốn lấy kết quả cũ, chạy ô 7 sau khi gắn Drive và nạp module.")
for name, path in completed_roots.items():
    root = Path(path)
    completeness = json.loads((root / "completeness.json").read_text())
    print(name, completeness)
    assert completeness["complete"], "Thiếu lượt chạy: chưa được coi là kết quả cuối."
    pairs = pd.read_json(root / "paired_comparisons.json")
    display(pairs[["comparison", "metric", "n", "mean_delta", "ci95_low", "ci95_high", "sign_flip_p_two_sided", "planned_primary", "campaign_mixed_environment", "confirmatory"]])
if "legacy10k" in completed_roots and "legacy70k" in completed_roots:
    small = pd.read_csv(Path(completed_roots["legacy10k"]) / "per_seed.csv")
    full = pd.read_csv(Path(completed_roots["legacy70k"]) / "per_seed.csv")
    scale = small[["seed", "method", "ARI", "NMI", "ACC"]].merge(full[["seed", "method", "ARI", "NMI", "ACC"]], on=["seed", "method"], suffixes=("_10k", "_70k"), validate="one_to_one")
    for metric in ["ARI", "NMI", "ACC"]:
        scale[metric + "_delta_70k_minus_10k"] = scale[metric + "_70k"] - scale[metric + "_10k"]
    scale.to_csv(OUTPUT_ROOT / "scale_comparison_paired_seeds.csv", index=False)
    display(scale)
print("Không tự động kết luận phương pháp tốt hơn. Đọc từng seed, chênh lệch, CI, suy biến và thời gian.")''')
md("## 7. Xuất kết quả, kể cả khi chiến dịch chưa xong\nÔ này có thể chạy riêng sau khi đã gắn Drive và nạp module, không yêu cầu GPU hoặc hoàn tất ô 5. Gói nhẹ có nhật ký đổi môi trường. Checkpoint/embedding vẫn giữ trên Drive; bảng thiếu seed chỉ là tạm thời.")
code('''if "OUTPUT_ROOT" not in globals() or "exp" not in globals():
    raise RuntimeError("Chạy ô 1–3 để gắn Drive và nạp module trước khi xuất kết quả.")
display(pd.DataFrame(exp.progress_inventory(OUTPUT_ROOT)))
archive = exp.export_review(OUTPUT_ROOT)
print("Gói kết quả:", archive)
print("Kích thước MB:", round(archive.stat().st_size / 1024**2, 2))
print("Bạn có thể lấy ZIP từ Google Drive và gửi lại để phân tích. Không cần gửi toàn bộ checkpoint.")''')

notebook = nb.v4.new_notebook(cells=cells, metadata={
    "colab": {"provenance": [], "gpuType": "A100", "name": "ADA-IM-DCSS"},
    "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "dcss_protocol": {"version": "2026-11-10-portable-v2", "code_sha256": code_sha, "a100_execution_verified_by_agent": False},
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
