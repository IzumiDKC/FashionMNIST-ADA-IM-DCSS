"""Create a self-contained Colab notebook; no local repository required on Colab."""
import hashlib
import json
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).parent
cells=[]


def md(s):
    cells.append(nb.v4.new_markdown_cell(s))


def code(s,hidden=False):
    c=nb.v4.new_code_cell(s)
    if hidden:
        c.metadata.update(cellView="form",collapsed=True)
    cells.append(c)


def build():
    cells.clear()
    md('''# DCSS / ADA-IM-DCSS — mở rộng nhiều bộ dữ liệu

**Fashion-MNIST đã có: chỉ đọc, không chạy lại.** Năm bộ bổ sung: EMNIST Balanced,
HAR, Pendigits, Waveform V1 và mẫu Nemotron cố định **100.000 hồ sơ trên toàn bộ 11 tệp**.

Chọn **Python 3 → A100** khi còn tài nguyên, sau đó **Run all**. Khi hết phiên,
có thể chuyển sang GPU CUDA khác (T4/L4/G4/H100 nếu có và PyTorch hỗ trợ), hoặc CPU,
rồi **Run all với cùng Drive và thư mục kết quả**. CPU rất chậm; TPU chưa hỗ trợ.
Colab không tự đổi GPU giúp notebook. Không chạy đồng thời hai phiên ghi cùng thư mục.

Checkpoint giữ model, optimizer, tâm/sigma, epoch/batch và trạng thái ngẫu nhiên.
Các bước sau checkpoint hợp lệ gần nhất có thể chạy lại khi phiên bị ngắt.
Không cam kết kết quả giống từng bit khi đổi phần cứng/phần mềm.

**Mặc định:** 10 seed, 100 epoch tiền huấn luyện + 100 epoch tinh chỉnh; 4 biến thể
và cơ sở AE + KMeans. Đây là chiến dịch dài, có thể cần nhiều phiên.
Chưa có kết quả huấn luyện mới trong notebook được bàn giao; các kiểm tra cục bộ là kiểm tra kỹ thuật.

Thứ tự chạy ba bộ nhỏ trước giúp sớm kiểm tra quy trình; không đổi giao thức hay chọn seed theo điểm số.''')
    md("## 1. Chuẩn bị thư viện và nhận diện thiết bị")
    code('''import os, sys, subprocess, importlib.util, importlib.metadata
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
requirements = {"numpy":"numpy", "scipy":"scipy", "sklearn":"scikit-learn", "pandas":"pandas",
    "matplotlib":"matplotlib", "threadpoolctl":"threadpoolctl", "pyarrow":"pyarrow", "unlzw3":"unlzw3"}
install = [pkg for module,pkg in requirements.items() if importlib.util.find_spec(module) is None]
try:
    correct_transformers = importlib.metadata.version("transformers") == "4.57.6"
except importlib.metadata.PackageNotFoundError:
    correct_transformers = False
if not correct_transformers:
    install += ["transformers==4.57.6", "huggingface_hub>=0.34,<1"]
if install:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*install])
import torch
torch.set_num_threads(2)
print("PyTorch:",torch.__version__,"CUDA build:",torch.version.cuda,"CUDA available:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0),"VRAM GiB:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
else:
    print("ĐANG DÙNG CPU: tiếp tục được nhưng rất chậm. Có thể đổi sang GPU rồi Run all.")''')
    md("## 2. Gắn Drive và giữ cấu hình đã chốt\nDùng cùng tài khoản chứa thư mục IM-DCSS. Các ID bên dưới là các thư mục bạn đã cung cấp.")
    code('''from google.colab import drive, auth
drive.mount("/content/drive")
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
drive_service = build("drive","v3",credentials=credentials,cache_discovery=False)
from pathlib import Path
DATASET_ORDER = ["har","pendigits","waveform","emnist","personas"]
WORKSPACE_FOLDER_ID = "106i4XskTRSdgBaZHF3jozGx-JxTMx8ow"
FASHION_RESULTS_FOLDER_ID = "1IqGB9d2gjGIVyK7kYVMbH4XcYFqZ8wSe"
NEW_FOLDER_NAME = "DCSS_MultiDataset_v1"
DEVICE = "auto"                 # auto = CUDA khi có; nếu không dùng CPU
ALLOW_CPU_FALLBACK = True
CHECKPOINT_SECONDS = 300
MAX_SESSION_HOURS = 0            # 0 = không đặt giới hạn riêng; ví dụ 4 để tạm dừng có checkpoint
ALLOW_SOFTWARE_CHANGE = True     # Ghi nhật ký khi PyTorch/Python/GPU thay đổi
RESUME_FOLDERS = {}              # Chỉ đặt nếu có nhiều lượt cùng bộ; không trỏ vào Fashion cũ
RUN_TRAINING = True              # False: chỉ đọc tiến độ và xuất thống kê, không tải dữ liệu/huấn luyện
RUN_DEVICE_CHECK = True          # Kiểm tra thực tế lưu/khôi phục trên thiết bị hiện tại trước chạy dài
assert len(DATASET_ORDER)==len(set(DATASET_ORDER)) and set(DATASET_ORDER)<=set(["har","pendigits","waveform","emnist","personas"])
assert MAX_SESSION_HOURS>=0 and CHECKPOINT_SECONDS>0''')
    md("## 3. Bộ thực thi đóng gói cùng notebook\nKhông cần clone repo. Các ô ẩn dưới đây chép mã đã kiểm tra vào /content. Dữ liệu/tiến độ được lưu trên Drive.")
    files=["dcss_core.py","multi_data.py","multi_runner.py","multi_analysis.py"]
    sources={f:(ROOT/f).read_text(encoding="utf-8") for f in files}
    hashes={f:hashlib.sha256(s.encode()).hexdigest() for f,s in sources.items()}
    for f,s in sources.items():
        code("%%writefile /content/"+f+"\n"+s,True)
    catalog=json.loads((ROOT/"sources.json").read_text(encoding="utf-8"))
    protocol=(ROOT/"PROTOCOL.md").read_text(encoding="utf-8")
    code(f'''import importlib, json, time
if "/content" not in sys.path:
    sys.path.insert(0,"/content")
import dcss_core as core, multi_data as data, multi_runner as runner, multi_analysis as analysis
for module in (core,data,runner,analysis):
    importlib.reload(module)
EXPECTED_CODE_HASHES = {hashes!r}
for name,sha in EXPECTED_CODE_HASHES.items():
    assert core.digest(Path("/content")/name)==sha, "Mã thực thi khác bản notebook đã chốt: "+name
CATALOG = {catalog!r}
PROTOCOL = {protocol!r}
WORKSPACE = data.drive_mounted_path(drive_service,WORKSPACE_FOLDER_ID)
FASHION_ROOT = data.drive_mounted_path(drive_service,FASHION_RESULTS_FOLDER_ID)
OUTPUT_ROOT = runner.protected_output(WORKSPACE/NEW_FOLDER_NAME,FASHION_ROOT)
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
CACHE_ROOT = OUTPUT_ROOT/"cache"
RAW_ROOT = CACHE_ROOT/"raw"
probe=OUTPUT_ROOT/".write_probe"
probe.write_text("ok",encoding="utf-8")
assert probe.read_text()=="ok"
probe.unlink()
core.atomic_json(OUTPUT_ROOT/"sources.json",CATALOG)
core.atomic_json(OUTPUT_ROOT/"implementation_hashes.json",EXPECTED_CODE_HASHES)
(OUTPUT_ROOT/"PROTOCOL.md").write_text(PROTOCOL,encoding="utf-8")
runtime = core.RuntimeOptions(device=DEVICE,allow_cpu=ALLOW_CPU_FALLBACK,checkpoint_seconds=CHECKPOINT_SECONDS,
    allow_software_change=ALLOW_SOFTWARE_CHANGE,max_session_hours=MAX_SESSION_HOURS)
active_device = core.select_device(runtime)
print("Fashion cũ CHỈ ĐỌC:",FASHION_ROOT)
print("Kết quả mới:",OUTPUT_ROOT)
print("Thiết bị:",active_device)
from IPython.display import display, HTML
import pandas as pd
old_summary,old_provenance = analysis.fashion_reference(FASHION_ROOT)
display(old_summary[["method","completed_seeds","ARI_mean","NMI_mean","ACC_mean"]])
inventory = core.progress_inventory(OUTPUT_ROOT)
display(pd.DataFrame(inventory)) if inventory else print("Chưa có checkpoint cho các bộ mới.")
if RUN_TRAINING and RUN_DEVICE_CHECK:
    device_check = runner.runtime_smoke(active_device)
    core.atomic_json(OUTPUT_ROOT/"runtime_validation_latest.json",device_check)
    print("Kiểm tra lưu/khôi phục trên thiết bị hiện tại: ĐẠT (không phải kết quả nghiên cứu).")''')
    md('''## 4. Chạy năm bộ mới và tiếp tục tiến độ

Lần đầu tải đúng các tệp Drive vào cache rồi kiểm tra SHA-256. Nemotron tải khoảng
2,25 GB dữ liệu nguồn và tạo vector cho mẫu 100.000; các phiên sau dùng lại cache.
EMNIST dùng Balanced 131.600 ảnh, 47 lớp. Các bộ có nhãn dùng toàn tập theo thiết kế
transductive, không báo đây là điểm kiểm thử độc lập.

Quan sát `RESULT DIRECTORY`, `Resume ... next batch ...`, `Already complete` và
`Text features committed`. Nếu lỗi/hết bộ nhớ, ô dừng, không tự đổi batch hay dữ liệu.
Chỉ coi hoàn thành khi `complete=true`, đủ 50 kết quả/bộ tính cả AE + KMeans.''')
    code('''from dataclasses import replace
import gc
session_deadline = time.monotonic()+MAX_SESSION_HOURS*3600 if MAX_SESSION_HOURS>0 else None
completed_roots = {}
paused = False
try:
    if RUN_TRAINING:
        for name in DATASET_ORDER:
            if session_deadline and time.monotonic()>=session_deadline:
                raise core.PauseRequested("Đã hết thời gian đặt trước; các kết quả đã ghi được giữ lại.")
            raw = data.ensure_sources(name,RAW_ROOT,CATALOG,drive_service)
            if name == "personas":
                inputs, labels, metadata = data.load_personas(raw,CACHE_ROOT/"personas",CATALOG,active_device,session_deadline)
            else:
                inputs, labels, metadata = data.load_labeled(name,raw,CATALOG)
            cfg = runner.config_for(metadata)
            print("ĐANG CHẠY:",name,"N =",len(inputs),"K =",cfg.n_clusters,"seeds =",cfg.seeds,flush=True)
            root = runner.run_experiment(inputs,labels,metadata,cfg,OUTPUT_ROOT,FASHION_ROOT,
                replace(runtime,resume_folder=RESUME_FOLDERS.get(name)),deadline=session_deadline)
            completed_roots[name] = str(root)
            core.atomic_json(OUTPUT_ROOT/"latest_completed_roots.json",completed_roots)
            display(pd.read_csv(root/"summary.csv"))
            del inputs,labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Chế độ chỉ thống kê: bỏ qua tải dữ liệu và huấn luyện.")
except core.PauseRequested as exc:
    paused=True
    print("ĐÃ TẠM DỪNG:",exc,"Run all ở phiên sau để tiếp tục.")
print("Xem bảng tiến độ ở ô kế tiếp; chưa đủ seed thì chưa phải kết quả cuối.")''')
    md('''## 5. Thống kê và đối chiếu Fashion-MNIST

Ô này có thể chạy riêng sau các ô 1–3, kể cả khi huấn luyện chưa xong. Báo cáo
trung bình ± độ lệch chuẩn mẫu; bảng thiếu seed là tạm thời. Nemotron không có
ARI/NMI/ACC theo nhãn chuẩn; dùng silhouette/DB/CH trên cùng vector đầu vào và
ARI giữa các seed để mô tả độ ổn định.

So sánh chính **ADA-IM − IM**, ghép theo seed; CI, p và hiệu chỉnh Holm cho năm
bộ mới. Fashion-MNIST giữ nguyên đối chiếu lịch sử. Không tự tuyên bố phương pháp
đề xuất tốt hơn nếu chỉ cải thiện một thước đo hoặc chưa đủ bằng chứng.''')
    code('''report_root,combined = analysis.campaign_report(OUTPUT_ROOT,FASHION_ROOT)
display(pd.read_csv(report_root/"progress.csv"))
display(combined)
display(pd.read_json(report_root/"primary_comparisons_holm.json"))
from IPython.display import Image
display(Image(filename=str(report_root/"scores.png")))
print("Báo cáo HTML:",report_root/"ThongKe_DCSS_MultiDataset.html")
print("Bảng CSV:",report_root/"all_datasets_summary.csv")
archive = analysis.export_review(OUTPUT_ROOT,FASHION_ROOT)
print("Gói báo cáo nhẹ:",archive)
print("Checkpoint, vector và dự đoán đầy đủ vẫn nằm trên Drive; gói ZIP không chứa các tệp lớn đó.")''')
    md("## 6. Giao thức đầy đủ và nguồn\n\n"+protocol)
    notebook=nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"colab":{"name":"ADA-IM-DCSS — MultiDataset.ipynb","provenance":[]},"accelerator":"GPU"})
    for i,c in enumerate(notebook.cells):
        c.id=f"multi-dcss-{i:02d}"
    nb.validate(notebook)
    target=ROOT/"DCSS_ADA_IM_MultiDataset.ipynb"
    nb.write(notebook,target)
    print(target,"cells =",len(cells),"bytes =",target.stat().st_size)
    return target


if __name__=="__main__":
    build()
