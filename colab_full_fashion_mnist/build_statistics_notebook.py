"""Create a CPU-only, self-contained analysis notebook, never a training runner."""
import ast
import hashlib
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
source = (ROOT / "analyze_results.py").read_text(encoding="utf-8")
cells = []


def md(text):
    cells.append(nb.v4.new_markdown_cell(text))


def code(text, collapsed=False):
    ast.parse(text)
    cell = nb.v4.new_code_cell(text)
    if collapsed:
        cell.metadata["jupyter"] = {"source_hidden": True}
    cells.append(cell)


md("""# Thống kê DCSS và ADA-IM-DCSS trên Fashion-MNIST

Notebook **chỉ đọc kết quả, không huấn luyện lại, không cần GPU hoặc PyTorch**.
Mở tại thư mục dự án có `DCSS_FashionMNIST_Full` rồi Run all. Trên Colab, chọn
**CPU**, tải notebook này lên, Run all và gắn đúng Google Drive khi được hỏi.
Notebook tự chứa mã phân tích; không cần clone repo.

Các bước: kiểm tra đủ seed/epoch → tính lại ACC/NMI/ARI từ `predictions.npz`
→ thống kê trung bình, độ lệch chuẩn và chênh lệch theo cặp → xem ảnh hưởng
T4/A100 → đối chiếu 10k/70k → xuất kết quả. Không đọc checkpoint `.pt` và không
chạy mã huấn luyện trong thư mục dữ liệu.

Tệp kết quả gốc được giữ nguyên. Đầu ra mới nằm trong `DCSS_Statistics/` cạnh
thư mục dữ liệu. Bản bàn giao đã thực thi có sẵn bảng và hình ngay trong notebook.
""")
md("## 1. Chọn dữ liệu và thư mục đầu ra\nNếu tự nhận sai vị trí, đặt DATA_ROOT_OVERRIDE thành đường dẫn đến thư mục chứa ba hồ sơ thí nghiệm. Không trỏ vào riêng một seed.")
code('''from pathlib import Path
import sys, importlib.util, subprocess
DATA_ROOT_OVERRIDE = None  # Ví dụ r"F:/ThacSi/Data Mining/final/DCSS_FashionMNIST_Full"
SELECTED_RUNS = {}         # Chỉ cần khi có nhiều thư mục cho cùng profile.
VERIFY_PREDICTIONS = True  # Tính lại toàn bộ 58 tệp, không lấy summary.csv làm nguồn điểm.

required = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn", "matplotlib": "matplotlib"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
get_ipython().run_line_magic("matplotlib", "inline")
if DATA_ROOT_OVERRIDE:
    DATA_ROOT = Path(DATA_ROOT_OVERRIDE).expanduser().resolve()
elif "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_ROOT = Path("/content/drive/MyDrive/DCSS_FashionMNIST_Full")
else:
    candidates = [Path.cwd() / "DCSS_FashionMNIST_Full", Path.cwd().parent / "DCSS_FashionMNIST_Full"]
    DATA_ROOT = next((p.resolve() for p in candidates if p.is_dir()), None)
    if DATA_ROOT is None:
        raise FileNotFoundError("Không tìm thấy DCSS_FashionMNIST_Full; đặt DATA_ROOT_OVERRIDE hoặc mở notebook từ thư mục dự án.")
if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Không thấy thư mục: {DATA_ROOT}")
OUTPUT_DIR = DATA_ROOT.parent / "DCSS_Statistics"
raw_candidate = DATA_ROOT.parent / "colab_full_fashion_mnist" / "data"
RAW_DATA_DIR = raw_candidate if (raw_candidate / "train-labels-idx1-ubyte").exists() else None
print("Nguồn:", DATA_ROOT)
print("Đầu ra:", OUTPUT_DIR)
print("Có thể đối chiếu nhãn IDX gốc:", RAW_DATA_DIR is not None)''')
md("## 2. Mã phân tích và kiểm tra tính toàn vẹn\nMỗi lượt phải khớp manifest, ID, số ảnh và lịch sử epoch. NPZ được đọc với allow_pickle=False. Lượt thiếu được nêu rõ, không ghi thành điểm 0. Các điểm phải khớp phép tính lại trong sai số 1e-9.")
code(source, collapsed=True)
code('''from IPython.display import display, Markdown
bundle = audit_and_recompute(DATA_ROOT, OUTPUT_DIR, selected=SELECTED_RUNS,
                             verify_predictions=VERIFY_PREDICTIONS, raw_data_dir=RAW_DATA_DIR)
display(bundle["inventory"].drop(columns="missing_pairs"))
print("Lượt hoàn tất:", bundle["audit"]["total_completed_runs"])
print("Đã tính lại từ tệp dự đoán:", bundle["audit"]["predictions_recomputed"])
print("Nhãn khớp từng byte với IDX gốc đã xác minh:", bundle["audit"]["official_raw_label_bytes_checked"])
print("Điểm cần kiểm tra:", bundle["audit"]["issues"])
if not bundle["audit"]["all_declared_profiles_complete"]:
    display(Markdown("**CHƯA ĐỦ LƯỢT: các bảng sau chỉ là kết quả tạm thời.**"))''')
md("""## 3. Bảng kết quả chính

Mỗi dòng là trung bình ± độ lệch chuẩn mẫu qua tất cả seed, gồm cả seed kém.
Chỉ số được nhân 100 để dễ đọc; ARI không phải tỷ lệ ảnh phân loại đúng.
`paper70k` dùng Conv/BN, 100+100 epoch, 10 seed; hai bộ `legacy` dùng Conv không
BN, 80+40 epoch, 3 seed. Không dùng khác biệt giữa hai kiến trúc để suy ra ảnh
hưởng riêng của tăng số ảnh.
""")
code('''summary = summarize_results(bundle)
for profile in ["paper70k", "legacy10k", "legacy70k"]:
    if profile in bundle["roots"]:
        display(Markdown("### " + profile))
        display(formatted_summary(summary, profile))
display(summary[["profile", "method", "collapsed_runs", "mixed_environment_runs", "guard_count_total", "rescue_count_total"]])''')
md("""## 4. So sánh theo cặp seed và câu hỏi chính của luận văn

Giả thuyết chính đã chốt: **ARI của ADA-IM − IM-DCSS trên paper70k**. Mọi cặp
dùng cùng seed; không so seed tốt nhất của phương pháp này với seed khác.
Δ dương nghĩa là phương pháp ở cột target cao hơn baseline.

KTC t 95% mô tả biến thiên qua seed trên bộ ảnh cố định. Kiểm định đổi dấu hai
phía xét toàn bộ 2^n cách đổi dấu; cần giả định phù hợp về phân phối chênh lệch
dưới giả thuyết không. KTC chứa 0 không chứng minh tương đương. Các p phụ được
hiệu chỉnh Holm trong từng profile, tách khỏi giả thuyết chính. Thay đổi GPU/
phần mềm và chỉ 3 seed giới hạn mức kết luận; không tự tuyên bố phương pháp thắng.
""")
code('''pairs, deltas, sensitivity = comparisons(bundle)
display(Markdown("### Giả thuyết chính"))
display(pairs[pairs.planned_primary][["profile", "target", "baseline", "metric", "n_pairs", "delta_mean_pp", "ci95_low_pp", "ci95_high_pp", "p_signflip_two_sided", "wins", "ties", "losses", "interpretation"]])
display(Markdown("### Các đối chiếu bổ sung"))
display(pairs[["profile", "target", "baseline", "metric", "delta_mean_pp", "ci95_low_pp", "ci95_high_pp", "p_signflip_two_sided", "p_holm_secondary_within_profile"]].round(6))''')
md("## 5. Ảnh hưởng của tăng số ảnh và mức giống nhau của hai phương pháp\nCặp legacy10k/legacy70k dùng cùng siêu tham số. Hai tập có phần giao nhau; không phải đánh giá độc lập trên ảnh chưa dùng. Độ đồng thuận giữa IM và ADA dưới đây so hai phân hoạch, không phải độ chính xác so với nhãn thật.")
code('''scale, scale_per_seed = scale_comparison(bundle)
display(scale[["method", "metric", "mean_10k_pct", "mean_70k_pct", "delta_mean_pp", "n_pairs"]].round(4))
agreement = prediction_agreement(bundle)
display(agreement.groupby("profile")[["partition_ARI_between_methods", "agreement_after_hungarian_pct"]].agg(["mean", "min", "max"]))''')
md("## 6. Dấu vết môi trường và thời gian\nGiữ mọi seed ở bảng chính. Bảng nhạy cảm chỉ dùng cặp có trace đầy đủ và cùng một môi trường, là mô tả hậu nghiệm. Lượt V1 chỉ có manifest không được coi là có trace đầy đủ. Thời gian huấn luyện pha trộn phần cứng không dùng để kết luận thuật toán nhanh hơn.")
code('''display(bundle["frame"].groupby(["profile", "device_history", "environment_provenance", "mixed_environment"]).size().rename("Số lượt").reset_index())
display(sensitivity[["profile", "metric", "seeds", "n_pairs", "delta_mean_pp", "ci95_low_pp", "ci95_high_pp"]])
display(summary[["profile", "method", "finetune_minutes_mean", "peak_gpu_gb_max"]].round(4))''')
md("## 7. Biểu đồ và chẩn đoán\nThanh sai số của biểu đồ điểm là độ lệch chuẩn mẫu; thanh ở biểu đồ chênh lệch là KTC 95%. Ma trận nhầm lẫn dùng seed 0 cố định, không chọn seed đẹp nhất; Hungarian chỉ ghép tên cụm khi đánh giá. Mỗi hình được lưu PNG và PDF.")
code('''figures = make_figures(bundle, summary, pairs, scale)
for fig in figures:
    display(fig)
    plt.close(fig)
print("Đã xuất", len(figures), "biểu đồ, mỗi biểu đồ có PNG/PDF.")''')
md("## 8. Nhận xét và kết luận theo số liệu thực tế\nNhận xét dưới đây được tạo từ các bảng vừa tính, không điền trước điểm mong muốn.")
code('''report = write_report(bundle, summary, pairs, sensitivity, scale, agreement)
display(Markdown(report))''')
md("## 9. Xuất gói thống kê\nĐầu ra gồm bảng từng seed, thống kê, kiểm định, lịch sử huấn luyện, báo cáo Markdown và hình ảnh. Không sao chép checkpoint hoặc sửa dữ liệu gốc.")
code('''import hashlib
write_json(OUTPUT_DIR / "analysis_execution.json", {
    "analysis_version": ANALYSIS_VERSION,
    "source_root": str(DATA_ROOT),
    "predictions_recomputed": bundle["audit"]["predictions_recomputed"],
    "training_executed": False,
    "gpu_required": False,
    "artifacts": [p.name for p in OUTPUT_DIR.iterdir() if p.is_file()],
})
archive = export_bundle(bundle)
print("Báo cáo:", OUTPUT_DIR / "KET_QUA_VA_NHAN_XET.md")
print("Gói thống kê:", archive)
print("Nguồn kết quả gốc được giữ nguyên.")''')

notebook = nb.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "name": "ThongKe_DCSS_FashionMNIST.ipynb"},
    "dcss_statistics": {"version": "2026-11-10-v1", "analysis_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "training": False},
})
for i, cell in enumerate(notebook.cells):
    cell.id = f"statistics-{i:02d}"
nb.validate(notebook)
path = PROJECT / "ThongKe_DCSS_FashionMNIST.ipynb"
nb.write(notebook, path)
print(path)
