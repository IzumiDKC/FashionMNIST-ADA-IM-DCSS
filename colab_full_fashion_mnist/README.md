# DCSS / ADA-IM-DCSS: tiếp tục T4 → A100 → GPU khác hoặc CPU

Notebook Fashion-MNIST: [bản tự chứa trong repo](DCSS_ADA_IM_FashionMNIST_Full.ipynb)
hoặc [bản sao trên Colab](https://colab.research.google.com/drive/1mfOuj9OxGh8Tubxrb09qVPtNVtc1yXSd).
Notebook dùng chung trước đây đã chuyển sang năm bộ dữ liệu bổ sung;
xem [hướng dẫn hiện tại](../colab_multi_dataset/README.md).
Bản vận hành V2, cập nhật 10/11/2026; giao thức khoa học giữ nguyên.

Đọc [PROTOCOL.md](PROTOCOL.md) trước khi diễn giải kết quả. Bài báo DCSS và mã
tác giả công bố có bất nhất; notebook công bố rõ lựa chọn triển khai và tách
phép đối chiếu bài báo khỏi phép kiểm tra tăng quy mô của cấu hình cũ.

1. Tải lại trang để nhận bản mới. Chọn **Python 3 → GPU A100 → phiên bản mới
   nhất**. RAM cao cấp có thể để tắt. Bấm Lưu rồi **Runtime → Run all**.
2. Gắn **đúng tài khoản Drive đã lưu kết quả T4**. Kết quả lưu tại
   `MyDrive/DCSS_FashionMNIST_Full/`, theo thư mục cấu hình và mã định danh.
3. Ở ô 3, xem bảng tiến độ cũ; ở ô 5, xem `RESULT DIRECTORY` và
   `Resume ... epochs done, next batch ...`. Giữ nguyên cấu hình. Mặc định chạy 10 seed cho bộ theo bài
   báo, rồi cặp 10k/70k dùng 3 seed để kiểm tra cấu hình cũ. Có thể cần nhiều
   phiên Colab; không có cơ chế giữ phiên trái với giới hạn của Colab.
4. Nếu hết A100/ngắt phiên, chọn L4/T4/G4 hoặc GPU CUDA khác đang được cấp rồi
   Run all. Mặc định `ALLOW_CPU_FALLBACK=True` cũng cho tiếp tục bằng cùng mạng
   PyTorch trên CPU, nhưng chậm hơn. Đặt False nếu chỉ muốn chờ GPU. Các lượt
   hoàn tất được bỏ qua; lượt dở tiếp tục từ batch đã lưu. Đổi GPU/CPU không
   đổi tên lượt chạy. Không tự tăng batch hay bật AMP để nối đúng thí nghiệm.
5. Chỉ dùng bảng có `completeness.json: complete=true` làm bảng cuối. Lấy
   `summary.csv`, `per_seed.csv`, `paired_comparisons.json`, biểu đồ và gói kết
   quả nhẹ do ô cuối tạo. Checkpoint và embedding vẫn nằm riêng trên Drive.
   Ô 7 có thể xuất kết quả tạm sau khi chạy ô 1–3, không cần GPU hoặc hoàn tất
   ô huấn luyện. Chỉ mở một phiên ghi cùng thư mục tại một thời điểm.

Checkpoint V1 tương thích được tự nhận; không cần đổi tên hoặc chép trọng số.
Tệp V1 gốc được giữ lại. V2 thêm hai slot checkpoint luân phiên kèm SHA-256,
lưu cuối mỗi epoch và sau một bước tối ưu khi quá 300 giây từ lần lưu trước.
Ngắt phiên chỉ chạy lại phần chưa lưu. Khi bản mới bị hỏng, thử bản trước;
nếu không còn bản hợp lệ, dừng xử lý thay vì âm thầm bắt đầu lại.

Nếu có nhiều thư mục tương thích, điền `RESUME_FOLDERS` ở ô 2, ví dụ
`{"paper70k": "paper70k_2cd93c725947"}` chỉ khi đúng tên thư mục của bạn.
Chương trình kiểm tra lại cấu hình/dữ liệu, không lựa chọn theo điểm tốt nhất.
`MAX_SESSION_HOURS=0` không đặt giới hạn thêm; giá trị dương cho lưu checkpoint
và tạm dừng sau số giờ chỉ định để phiên sau tiếp tục.

Python 3 là ngôn ngữ/runtime, không phải mẫu GPU. Notebook không tự xin GPU
mới khi Colab thu hồi phiên, không vượt hạn mức thuê bao. TPU chưa được hỗ
trợ. Giữ `OUTPUT_ROOT` và tài khoản Drive để tìm đúng công việc đã làm.

Đổi thiết bị/phần mềm được ghi trong `sessions/`, `execution_trace` và các cột
`mixed_environment`. Tiếp tục checkpoint không bảo đảm trùng từng bit như chạy
liên tục trên một GPU. Kết quả pha trộn được giữ đầy đủ và gắn thăm dò; không
so trực tiếp thời gian T4/A100/CPU để kết luận thuật toán nhanh hơn. Mặc định
cho tiếp tục khi phần mềm Colab thay đổi với nhật ký; đặt
`ALLOW_SOFTWARE_CHANGE=False` nếu cần bắt buộc khớp phiên bản.

Notebook tự chứa mã thực thi, không cần clone hoặc truy cập ổ F: từ Colab.
Không cài lại torch/CUDA. Dữ liệu ưu tiên bốn tệp Drive đã cung cấp; khi tải
Drive không được, dùng bản chính thức có cùng SHA-256 và thông báo nguồn thay
thế. Nếu dữ liệu khác mã kiểm tra, chương trình dừng.

Các tệp dành cho kiểm tra/chỉnh sửa:

- `experiment.py`: thực thi; `build_notebook.py`: đóng gói notebook tự chứa.
- `test_experiment.py`, `test_portable_resume.py`: công thức, gradient, metric,
  tiếp tục giữa epoch, nhập V1, khôi phục khi ghi dở và kiểm soát định danh.
- `run_local_validation.py`: kiểm tra luồng đầy đủ trên 256 ảnh thật, CPU.
- `canonical_hashes.json`, `audit/dataset_verification.json`: nguồn gốc dữ liệu.
- `audit/upstream/`: bản mã tác giả tại commit đã cố định, giữ LICENSE.
- `audit/test_results_portable.xml`, `audit/local_validation_portable.json`: bằng chứng kiểm tra V2.
- `audit/v1/`: nguyên bản mã V1 để đối chiếu tính tương thích.
- `audit/colab_before_portable_update.ipynb`: bản sao notebook và lỗi CPU trước
  khi cập nhật, chỉ lưu cục bộ.
- `audit/original_colab.ipynb`: bản sao notebook trước khi thay thế, chỉ lưu cục bộ.

Các kết quả trong `test_runs/` chỉ phục vụ kiểm tra chương trình, **không phải
kết quả thực nghiệm để đưa vào luận văn**. Không trộn với bảng 10k cũ hoặc với
điểm công bố của bài báo. Thư mục mã gốc `src/` và kết quả cũ được giữ nguyên.
