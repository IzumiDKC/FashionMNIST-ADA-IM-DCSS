# Chạy DCSS / ADA-IM-DCSS trên năm bộ dữ liệu bổ sung

**Chỉ xem số liệu đã hoàn tất:** mở
[ThongKe_DCSS_MultiDataset.ipynb](ThongKe_DCSS_MultiDataset.ipynb) và chọn **Run all**.
Hoặc [mở trực tiếp notebook thống kê trên Colab](https://colab.research.google.com/drive/19ylnoBPKHPnkHUjKxmKeWdld0KQgndqq), chọn **CPU → Run all**.
Notebook này đã nhúng kết quả kiểm tra ngày 15/11/2026, có sẵn đầu ra đã thực thi,
không cần GPU, PyTorch hay gắn Drive ở chế độ mặc định. Có thể chạy trong VS Code,
Jupyter hoặc tải riêng file lên Colab và chọn CPU. Đầu ra CSV/JSON, 7 biểu đồ,
HTML và ZIP được tạo trong thư mục `DCSS_MultiDataset_Statistics` tại nơi chạy.
Để đọc kết quả trực tiếp trên Drive thay cho bản nhúng, đổi `SOURCE_MODE="drive"`
trong ô cấu hình; kết quả gốc vẫn chỉ được đọc.

Notebook: [DCSS_ADA_IM_MultiDataset.ipynb](DCSS_ADA_IM_MultiDataset.ipynb).
Mã, cấu hình và dấu vân tay nguồn đã được nhúng trong notebook; không cần tải
repo vào Colab. Đọc [PROTOCOL.md](PROTOCOL.md) trước khi diễn giải kết quả.

## Cách chạy trên Colab

1. Mở notebook đã cập nhật tại https://colab.research.google.com/drive/1im9fG_949mpbZOIxdOp_nQ0M3VfLgenV.
2. Chọn Python 3 và A100 nếu được cấp. T4/L4/G4/H100 hoặc GPU CUDA tương thích
   khác cũng dùng được; không chọn TPU. CPU có thể tiếp tục nhưng sẽ chậm.
3. Chọn **Run all** và cho phép Colab gắn/đọc đúng tài khoản Drive chứa dữ liệu.
4. Kiểm tra dòng `Fashion cũ CHỈ ĐỌC` và `Kết quả mới`. Notebook tự xác định
   thư mục từ các ID đã cung cấp, không yêu cầu sửa đường dẫn khi đổi runtime.
5. Kết quả mới nằm tại `IM-DCSS/DCSS_MultiDataset_v1`. Đợi đủ **50 kết quả/bộ**
   (10 seed × 5 phương pháp, tính cả AE + KMeans) và `complete=true`.

Đã sao lưu notebook Fashion-MNIST trước khi cập nhật:
https://colab.research.google.com/drive/1mfOuj9OxGh8Tubxrb09qVPtNVtc1yXSd.
Bản sao giữ cả nội dung và đầu ra cũ. Thư mục `DCSS_FashionMNIST_Full` không bị
huấn luyện lại hoặc ghi đè; cả kết quả 10k/70k cũ vẫn được giữ.

## Khi hết phiên hoặc hết hạn mức A100

Chọn GPU khác trong Colab → kết nối runtime mới → **Run all**. Giữ nguyên tài
khoản Drive, `NEW_FOLDER_NAME` và cấu hình khoa học. Các lượt đã hoàn thành được
bỏ qua; lượt đang chạy tiếp tục từ checkpoint hợp lệ gần nhất. Các khối vector
Nemotron đã ghi cũng được dùng lại.

Notebook không thể tự cấp GPU mới khi tài khoản hết tài nguyên. Khi không có
CUDA, mặc định `ALLOW_CPU_FALLBACK=True` cho phép dùng CPU và in thông báo rõ.
Không có cam kết giữ phần chưa kịp ghi khi Colab tắt đột ngột. Các batch sau
checkpoint, bước KMeans/đánh giá còn dở hoặc khối văn bản chưa ghi có thể làm lại.

Checkpoint được lưu sau mỗi epoch và định kỳ 300 giây tại ranh giới batch,
có hai bản luân phiên và checksum. Không tự thay đổi batch size, hàm mất mát hay
dữ liệu để né lỗi. Thay các yếu tố này phải mở một thí nghiệm khác. Không chạy
hai phiên ghi cùng thư mục kết quả.

## Dữ liệu và phạm vi

* EMNIST **Balanced**: toàn bộ 131.600 ảnh, 47 lớp; không phải bản ByClass.
* HAR: toàn bộ 10.299 hàng, 561 đặc trưng, 6 hoạt động.
* Pendigits: toàn bộ 10.992 hàng, 16 đặc trưng, 10 lớp.
* Waveform V1: toàn bộ 5.000 hàng, 21 đặc trưng, 3 lớp.
* Nemotron: **100.000 hồ sơ cố định**, chọn ngẫu nhiên trên cả 11 tệp/một triệu
  hàng. Chỉ dùng `persona`, không lấy giới tính/nghề nghiệp làm nhãn đúng.

Bốn bộ có nhãn dùng toàn tập theo thiết kế transductive; ARI/NMI/ACC không phải
điểm trên tập kiểm thử độc lập. Nemotron dùng bộ mã hóa cố định, tối đa 128 token,
K=20 chốt trước cho thăm dò; báo cáo tỷ lệ bị cắt văn bản. Các chỉ số nội tại dùng
cùng vector và cùng mẫu đánh giá cho mọi phương pháp. ARI giữa các seed ở đây
đo độ ổn định phân hoạch, không phải độ chính xác theo nhãn.

Thứ tự mặc định: HAR → Pendigits → Waveform → EMNIST → Nemotron. Mỗi bộ có 10 seed,
100 epoch tiền huấn luyện và 100 epoch tinh chỉnh cho mỗi biến thể. Tất cả biến
thể trong một seed chia sẻ khởi tạo; giữ hệ số như thực nghiệm Fashion trước đó.

## Xem và xuất kết quả

Chạy ô 5 để tạo:

* `analysis/ThongKe_DCSS_MultiDataset.html`: báo cáo và biểu đồ.
* `analysis/all_datasets_summary.csv`: trung bình, SD và số seed hợp lệ.
* `analysis/primary_comparisons_holm.json`: ADA-IM − IM theo seed, CI và hiệu chỉnh
  Holm cho năm bộ mới. Không gộp Fashion cũ vào họ kiểm định mới.
* `analysis/partition_stability.csv`: độ ổn định giữa các seed.
* `analysis/progress.csv`: tiến độ và trạng thái hoàn thành.
* `results_for_review.zip`: gói báo cáo nhẹ, không chứa model/checkpoint/vector lớn.

Nếu chỉ xem kết quả, đặt `RUN_TRAINING=False` rồi Run all: notebook không tải dữ
liệu và không huấn luyện. Không đổi tên thư mục. Bảng thiếu seed chỉ là tạm thời;
chỉ số không xác định được để trống, không thay bằng 0.

CSV/JSON lưu tỷ lệ; SD dùng `ddof=1`. So sánh chính là ADA-IM với IM để kiểm tra
lợi ích riêng của ADA; các đối chứng gồm DCSS, DCSS-RMS và AE + KMeans.

## Kiểm tra đã thực hiện khi bàn giao

`audit/data_verification.json` xác minh đủ dữ liệu và mẫu Nemotron 100k; `sources.json`
ghi SHA-256 của toàn bộ 30 tệp nguồn. `audit/real_input_execution.json` ghi kiểm tra
lan truyền xuôi/ngược bằng mẫu thật của cả năm bộ, mã hóa 520 văn bản bằng đúng
encoder và kiểm tra tiếp tục từ khối 512 vector. Đây **không phải** điểm nghiên cứu.

`test_multi_dataset.py` kiểm tra ngắt/tiếp tục giữa epoch, tính nhất quán dữ liệu,
bỏ qua lượt hoàn tất, thống kê, phát hiện tệp lỗi và bảo vệ thư mục cũ.
`audit/notebook_validation.json` ghi rõ phạm vi thực thi notebook cục bộ và các
thay thế đường dẫn/xác thực Colab. GPU và đăng nhập Colab thật được kiểm tra khi
bạn Run all. Đợt huấn luyện đầy đủ đã hoàn tất và được kiểm tra ngày 15/11/2026:
250 kết quả mới, 250 lịch sử pha huấn luyện, 25.000 bản ghi epoch, đủ 10 seed mỗi bộ.
Xem [biên bản kiểm tra hoàn tất](KIEM_TRA_HOAN_TAT_2026-11-15.md) và
[kết luận tổng hợp](../README.md#5-kết-quả). Phép kiểm tra này đối chiếu JSON,
lịch sử và dấu vân tay đã lưu; không tính lại toàn bộ dự đoán từ trọng số.

Mã phân chia thành `dcss_core.py` (bản sao nguyên vẹn bộ thực thi cũ),
`multi_data.py` (nguồn/tiền xử lý/cache), `multi_runner.py` (mạng dạng bảng và
điều phối) và `multi_analysis.py` (thống kê). Sửa mã nguồn xong phải chạy lại
`build_notebook.py`; không sửa riêng một bản rồi tiếp tục checkpoint có cấu hình khác.
