"""Build a standalone statistics notebook containing the verified result snapshot."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import nbformat as nb
import statistics_only as report

ROOT=Path(__file__).parent.resolve()
PROJECT=ROOT.parent
SNAPSHOT=ROOT/"audit/results_2026_11_15"


def main():
    payload=report.read_campaign(SNAPSHOT/"review",PROJECT/"DCSS_FashionMNIST_Full")
    payload["snapshot_date"]="2026-11-15"
    payload["archive_sha256"]=report.sha(SNAPSHOT/"results_for_review.zip")
    payload["audit"]=report.read(SNAPSHOT/"completion_audit.json")
    validation=report.validate_payload(payload)
    packed=gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8"),mtime=0)
    encoded=base64.b64encode(packed).decode("ascii")
    packed_hash=hashlib.sha256(packed).hexdigest()
    source=(ROOT/"statistics_only.py").read_text(encoding="utf-8")
    cells=[]
    def md(s):
        cells.append(nb.v4.new_markdown_cell(s))
    def code(s,hidden=False):
        ast.parse(s)
        c=nb.v4.new_code_cell(s)
        if hidden:
            c.metadata["jupyter"]={"source_hidden":True}
            c.metadata["cellView"]="form"
        cells.append(c)
    md("""# Thống kê DCSS, IM-DCSS và ADA-IM-DCSS

**Mở file này và chọn Run all. Không cần GPU, PyTorch, dữ liệu ảnh/văn bản hoặc
notebook huấn luyện.** Trên Colab, chọn **Python 3 → CPU**.

Bản kết quả đã kiểm tra ngày **15/11/2026 được nhúng ngay trong file**: năm bộ
mới, 250 bản kết quả, 25.000 bản ghi epoch; thêm 50 kết quả Fashion-MNIST 70k làm
đối chiếu lịch sử. Có thể chuyển sang đọc thư mục Drive hoặc thư mục máy ở ô 1.
Chế độ mặc định không cần gắn Drive. Kết quả gốc luôn chỉ được đọc.

Notebook cung cấp: tiến độ → ARI/NMI/ACC → chỉ số nội tại Nemotron → so sánh ghép
cặp → suy biến/số cụm → độ ổn định → đường học → dữ liệu từng seed → xuất CSV/JSON,
biểu đồ và HTML. Bản đã thực thi có thể xem ngay các bảng và hình khi mở.

**Phạm vi:** tính lại thống kê từ điểm và lịch sử đã lưu; không tải model hoặc tính
lại mọi điểm từ dự đoán. Độ ổn định phân hoạch lấy từ tệp thống kê đã lưu.
Bốn bộ mới có nhãn dùng đánh giá transductive toàn tập, không phải tập kiểm thử
độc lập. Nemotron không có nhãn cụm chuẩn; không báo ARI/NMI/ACC theo nhãn đúng.""")
    md("## 1. Chọn nguồn kết quả và thư mục xuất\nGiữ nguyên mặc định để chạy ngay. Chỉ cần đổi nguồn khi muốn đọc kết quả khác.")
    code('''from pathlib import Path
import sys, importlib.util, subprocess
SOURCE_MODE = "snapshot"       # "snapshot", "drive" hoặc "local"
DATA_ROOT_OVERRIDE = None      # Dùng với local; thư mục chứa har_..., emnist_..., ...
FASHION_ROOT_OVERRIDE = None   # None = DCSS_FashionMNIST_Full bên cạnh thư mục mới
OUTPUT_DIR = Path.cwd() / "DCSS_MultiDataset_Statistics"

requirements = {"numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "matplotlib":"matplotlib"}
missing = [pkg for module,pkg in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
get_ipython().run_line_magic("matplotlib","inline")
assert SOURCE_MODE in {"snapshot","drive","local"}
print("Chế độ:",SOURCE_MODE)
print("Thư mục xuất:",OUTPUT_DIR)''')
    md("## 2. Nạp bộ phân tích và kết quả đã kiểm tra\nHai ô mã thu gọn dưới đây giúp notebook hoạt động độc lập. Không có lệnh huấn luyện.")
    code("import types, sys\nANALYSIS_SOURCE = "+repr(source)+"""
analysis = types.ModuleType("dcss_statistics_only_embedded")
sys.modules[analysis.__name__] = analysis
exec(compile(ANALYSIS_SOURCE,"<statistics_only_embedded>","exec"),analysis.__dict__)
""",True)
    code("SNAPSHOT_B64 = "+repr(encoded)+"\nSNAPSHOT_SHA256 = "+repr(packed_hash),True)
    code('''import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown, FileLink
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.dpi":110})
pd.set_option("display.max_columns",30)
pd.set_option("display.max_rows",60)

if SOURCE_MODE == "snapshot":
    payload = analysis.unpack_snapshot(SNAPSHOT_B64,SNAPSHOT_SHA256)
    print("Đang dùng bản kết quả cố định đã kiểm tra ngày",payload["snapshot_date"])
    print("Không cần truy cập Drive; không đọc trạng thái runtime đang chạy.")
else:
    if SOURCE_MODE == "drive":
        from google.colab import drive
        drive.mount("/content/drive")
        data_root = Path(DATA_ROOT_OVERRIDE) if DATA_ROOT_OVERRIDE else Path("/content/drive/MyDrive/IM-DCSS/DCSS_MultiDataset_v1")
    else:
        if not DATA_ROOT_OVERRIDE:
            raise ValueError("Chế độ local cần DATA_ROOT_OVERRIDE trỏ đến thư mục kết quả.")
        data_root = Path(DATA_ROOT_OVERRIDE).expanduser().resolve()
    fashion_root = Path(FASHION_ROOT_OVERRIDE) if FASHION_ROOT_OVERRIDE else data_root.parent/"DCSS_FashionMNIST_Full"
    if not data_root.is_dir() or not fashion_root.is_dir():
        raise FileNotFoundError("Không tìm thấy thư mục kết quả mới/cũ. Kiểm tra đường dẫn ở ô 1.")
    if OUTPUT_DIR.resolve()==data_root.resolve() or OUTPUT_DIR.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("OUTPUT_DIR phải ở ngoài thư mục kết quả gốc.")
    if OUTPUT_DIR.resolve()==fashion_root.resolve() or OUTPUT_DIR.resolve().is_relative_to(fashion_root.resolve()):
        raise ValueError("OUTPUT_DIR phải ở ngoài thư mục Fashion-MNIST gốc.")
    payload = analysis.read_campaign(data_root,fashion_root)

validation = analysis.validate_payload(payload)
frame,progress,summary,health = analysis.make_tables(payload)
all_pairs,primary = analysis.comparisons(frame,payload)
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
print(json.dumps(validation,ensure_ascii=False,indent=2))
print("Đơn vị CSV/JSON: tỷ lệ gốc. Bảng ARI/NMI/ACC và biểu đồ tương ứng nhân 100 để dễ đọc.")
print("SD là độ lệch chuẩn mẫu (ddof=1); thiếu/không xác định để trống.")''')
    md("""## 3. Quy mô, tiến độ và môi trường

Mỗi bộ mới có **10 seed × 5 phương pháp = 50 kết quả**, tính cả AE + KMeans.
Không gộp Fashion-MNIST cũ vào họ kiểm định của năm bộ mới. Bảng thiếu seed chưa
phải kết quả cuối. Đánh dấu trộn môi trường khi quá trình có thay đổi phần cứng
hoặc phần mềm; không dùng bảng thời gian để so tốc độ thuần túy giữa các GPU.""")
    code('''progress_display = progress.copy()
progress_display["dataset"] = progress_display.dataset.map(analysis.DATA_NAMES)
display(progress_display.rename(columns={"dataset":"Bộ dữ liệu","samples":"Số mẫu","completed":"Đã xong",
    "expected":"Cần có","complete":"Hoàn tất","mixed_environment":"Trộn môi trường","historical":"Đối chiếu cũ"}))
print("Kết quả bộ mới:",int(progress.loc[~progress.historical,"completed"].sum()),"/",
      int(progress.loc[~progress.historical,"expected"].sum()))
if not progress.complete.all():
    display(Markdown("**Có bộ chưa hoàn tất: các bảng hiện tại chỉ là tạm thời.**"))''')
    md("""## 4. ARI, NMI, ACC: trung bình ± SD và từng seed

Các số dưới đây là **chỉ số ×100**. Ví dụ ACC=60 tương ứng tỷ lệ 0,60.
Mỗi cột biểu đồ là trung bình, thanh sai số là SD, mỗi chấm là một seed.
Không so trực tiếp điểm số của hai bộ dữ liệu khác nhau để xếp hạng độ khó.
Nemotron không xuất hiện trong bảng này vì không có nhãn cụm chuẩn.""")
    code('''for dataset in (analysis.FASHION,"emnist","har","pendigits","waveform"):
    if dataset in payload["datasets"]:
        display(Markdown("### "+analysis.DATA_NAMES[dataset]))
        display(analysis.display_summary(summary,dataset,("ARI","NMI","ACC")))
analysis.plot_label_scores(frame,summary,OUTPUT_DIR)''')
    md("""## 5. Nemotron: chỉ số nội tại trên cùng không gian đặc trưng

Silhouette và CH cao hơn, DB thấp hơn là dấu hiệu thuận lợi **trong cùng biểu
diễn và cùng mẫu đánh giá**. Cần xem tiếp số cụm hoạt động: gom thành 1–3 cụm
có thể cho chỉ số nội tại tưởng như tốt dù K đã đặt là 20.
Không thay giá trị không xác định bằng 0. Cột “hợp lệ/đã xong” cho biết trung bình
được tính từ bao nhiêu seed. Văn bản dùng encoder cố định và giới hạn 128 token.""")
    code('''display(analysis.display_summary(summary,"personas",analysis.METRICS[3:]))
embedding = payload["datasets"]["personas"].get("embedding") or {}
if embedding:
    display(pd.DataFrame([{"Số văn bản":embedding.get("n"),"Số chiều":embedding.get("dimension"),
        "Số hồ sơ bị cắt token":embedding.get("truncated_rows"),
        "Tỷ lệ bị cắt":embedding.get("truncated_rows",0)/embedding["n"],
        "Tổng token gốc":embedding.get("original_token_count")}]))
analysis.plot_personas(frame,summary,OUTPUT_DIR)''')
    md("""## 6. ADA-IM có thêm lợi ích so với IM hay không?

So sánh chính: **ADA-IM − IM**, ghép đúng seed. Dùng ARI ở bốn bộ có nhãn và
silhouette chung ở Nemotron. Hiệu chỉnh Holm cho **5 phép thử**, không gồm
Fashion-MNIST cũ. CI 95% theo t trong hình là CI từng bộ chưa hiệu chỉnh; p Holm
mới dùng cho họ kiểm định đã chốt.

Chênh lệch dương có lợi cho ADA theo hai thước đo này. p lớn hơn 0,05 không chứng
minh tương đương hoặc không có tác dụng. Kết quả còn phụ thuộc biểu diễn, suy
biến và môi trường. Các so sánh với DCSS, DCSS-RMS và AE + KMeans được xuất riêng.""")
    code('''primary_display = primary[["dataset","metric","n","complete","mean_delta","ci95_low","ci95_high",
    "wins","losses","ties","p_raw","p_holm_family5"]].copy()
primary_display["dataset"] = primary_display.dataset.map(analysis.DATA_NAMES)
display(primary_display)
analysis.plot_primary(primary,OUTPUT_DIR)
if primary.complete.all() and (primary.p_holm_family5.notna()).all():
    significant = primary[primary.p_holm_family5<.05]
    if significant.empty:
        display(Markdown("**Chưa xác nhận lợi ích bổ sung rõ ràng của ADA so với IM trên năm bộ này sau hiệu chỉnh Holm ở mức 0,05.**"))
    else:
        display(Markdown("**Có chênh lệch đạt ngưỡng kiểm định; cần đọc chiều tác động, CI và hạn chế trước khi kết luận.**"))
        display(significant)
else:
    display(Markdown("**Chưa đủ cặp seed hợp lệ để kết luận cả chiến dịch.**"))''')
    md("""## 7. Suy biến, số cụm hoạt động và các giá trị thiếu

Giữ nguyên mọi lượt chạy, kể cả kết quả bất lợi. Cờ suy biến dùng ngưỡng của bộ
thực thi; không bị gắn cờ vẫn có thể có ít cụm hơn K. AE + KMeans không có cờ
chẩn đoán của đầu phân cụm DCSS, nên ô cờ được để trống thay vì gán “không suy biến”.
Các lượt một cụm vẫn có ARI/NMI/ACC khi có nhãn, nhưng chỉ số nội tại có thể không
xác định. Không diễn giải trung bình chỉ số nội tại như đủ 10 giá trị khi thiếu.""")
    code('''health_display = health.copy()
health_display["dataset"] = health_display.dataset.map(analysis.DATA_NAMES)
health_display["method"] = health_display.method.map(analysis.METHOD_NAMES)
display(health_display)
print("Tổng lượt tinh chỉnh bị gắn cờ suy biến:",int(health.flagged_collapsed.sum()))
analysis.plot_health(health,OUTPUT_DIR)
invalid = frame[(frame.dataset!=analysis.FASHION)&frame.silhouette_common.isna()]
display(invalid[["dataset","seed","method","hard_active_clusters","common_metric_status"]]
        .rename(columns={"dataset":"Bộ dữ liệu","seed":"Seed","method":"Phương pháp"}))''')
    md("""## 8. Độ ổn định giữa các seed

ARI ở đây so sánh **hai phân hoạch**, không so với nhãn đúng. 10 seed tạo 45 cặp
phụ thuộc nhau, vì vậy không coi chúng là 45 quan sát độc lập để kiểm định.
Độ ổn định cao không bảo đảm chất lượng cao, đặc biệt khi các lượt cùng suy biến.
Các giá trị này được đọc từ partition_stability.json đã lưu, không tính lại từ
250 tệp dự đoán trong notebook thống kê.""")
    code('''stability = pd.DataFrame(payload["stability"])
display(stability[["dataset","method","seeds","pairs","partition_ARI_mean","pairwise_sd_descriptive"]])
analysis.plot_stability(payload,OUTPUT_DIR)''')
    md("""## 9. Quá trình học và thời gian

Đường là trung bình, vùng mờ là SD qua các seed. Biểu đồ dùng MSE tái tạo và
silhouette mềm; không so tổng loss giữa các phương pháp có hệ số khác nhau để
khẳng định phương pháp nào tốt hơn. Thời gian dưới đây là thời gian giai đoạn
đã ghi, không bao gồm đầy đủ tải dữ liệu/mã hóa văn bản/KMeans và không phải
tổng thời gian treo phiên. Không cộng lặp phần tiền huấn luyện dùng chung.""")
    code('''analysis.plot_histories(payload,OUTPUT_DIR)
new_runs = frame[frame.dataset!=analysis.FASHION].copy()
new_runs["finetune_minutes"] = new_runs.finetune_seconds/60
timing = new_runs.groupby(["dataset","method"]).agg(
    seeds=("seed","count"),finetune_minutes_mean=("finetune_minutes","mean"),
    finetune_minutes_sd=("finetune_minutes","std"),peak_gpu_gb=("peak_gpu_gb","max"),
    mixed_environment_runs=("mixed_environment","sum")).reset_index()
display(timing)
timing.to_csv(OUTPUT_DIR/"timing.csv",index=False,encoding="utf-8-sig")''')
    md("## 10. Dò từng seed và phân bố cụm\nĐổi ba biến ở ô dưới rồi chạy lại riêng ô để xem một kết quả. Cụm số 0 của hai mô hình không mặc nhiên cùng ý nghĩa.")
    code('''VIEW_DATASET = "personas"
VIEW_METHOD = "ada_im"
VIEW_SEED = 0
selected = frame[(frame.dataset==VIEW_DATASET)&(frame.method==VIEW_METHOD)&(frame.seed==VIEW_SEED)]
display(selected[["dataset","method","seed",*analysis.METRICS,"hard_active_clusters","hard_largest_cluster_fraction"]])
cluster_table = analysis.cluster_detail(frame,VIEW_DATASET,VIEW_METHOD,VIEW_SEED)
display(cluster_table)
fig,ax = plt.subplots(figsize=(11,3.5),layout="constrained")
ax.bar(cluster_table["Cụm"],cluster_table["Số mẫu"],color="#c13e63")
ax.set(xlabel="ID cụm của lượt đã chọn",ylabel="Số mẫu",
       title=analysis.DATA_NAMES[VIEW_DATASET]+" · "+analysis.METHOD_NAMES[VIEW_METHOD]+f" · seed {VIEW_SEED}")
analysis.save_show(fig,OUTPUT_DIR/"07_selected_cluster_distribution.png")
display(frame[frame.dataset==VIEW_DATASET][["dataset","method","seed",*analysis.METRICS,
    "hard_active_clusters","collapsed"]].sort_values(["method","seed"]))''')
    md("""## 11. Xuất toàn bộ bảng và biểu đồ

Thư mục xuất chứa CSV/JSON, ảnh PNG, báo cáo HTML có ảnh nhúng và gói ZIP.
HTML có thể mở độc lập bằng trình duyệt. Notebook đã có sẵn kết quả hiển thị,
nhưng CSV/JSON luôn giữ đơn vị gốc để dùng phân tích tiếp. Không ghi vào thư mục
checkpoint hay dữ liệu gốc.""")
    code('''tables = analysis.export_tables(OUTPUT_DIR,payload,frame,progress,summary,health,all_pairs,primary)
html_path,zip_path = analysis.export_html_zip(OUTPUT_DIR,tables,primary)
print("Đã xuất:",html_path)
print("Gói đầy đủ:",zip_path)
print("Bảng chính:",OUTPUT_DIR/"summary.csv")
if "google.colab" not in sys.modules:
    display(FileLink(str(html_path)))
    display(FileLink(str(zip_path)))
else:
    print("Trên Colab: mở thanh Files và tải ZIP, hoặc đặt DOWNLOAD_ZIP=True ở ô cuối.")''')
    code('''DOWNLOAD_ZIP = False
if DOWNLOAD_ZIP:
    if "google.colab" in sys.modules:
        from google.colab import files
        files.download(str(zip_path))
    else:
        display(FileLink(str(zip_path)))''')
    notebook=nb.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"colab":{"name":"ThongKe_DCSS_MultiDataset.ipynb","provenance":[]},
        "dcss_analysis":{"snapshot_date":"2026-11-15","training":False,"snapshot_sha256":packed_hash}})
    for i,c in enumerate(notebook.cells):
        c.id=f"stats-multi-{i:02d}"
    nb.validate(notebook)
    target=ROOT/"ThongKe_DCSS_MultiDataset.ipynb"
    nb.write(notebook,target)
    (ROOT/"audit/statistics_snapshot_build.json").write_text(json.dumps(dict(snapshot_sha256=packed_hash,
        compressed_bytes=len(packed),notebook=str(target),**validation),indent=2,ensure_ascii=False),encoding="utf-8")
    print(target,"bytes",target.stat().st_size,"snapshot",len(packed),"cells",len(cells))


if __name__=="__main__":
    main()
