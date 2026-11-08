# Phân cụm Fashion-MNIST với DCSS và các mở rộng

Tái lập phương pháp DCSS trên một phần dữ liệu Fashion-MNIST gồm **10.000 ảnh**, đồng thời khảo sát ba mở rộng: **Ada-DCSS**, **IM-DCSS** và **Ada-IM-DCSS**.

Phương pháp nền: Vardakas, Papakostas và Likas, [Deep Clustering Using the Soft Silhouette Score: Towards Compact and Well-Separated Clusters](https://doi.org/10.1007/s10994-026-07026-w), *Machine Learning*, 2026.

## Báo cáo và bản trình diễn

- [Báo cáo Word hoàn chỉnh](Final.docx), gồm phương pháp, thực nghiệm, kết quả và hình chụp giao diện.
- [Giao diện trình diễn](web/index.html), gồm đối chứng, độ nhạy, phân tích thành phần, biểu diễn đặc trưng và ảnh mẫu.
- [Bảng kết quả đối chứng](outputs_fashion_mnist/benchmark/summary.csv).

Để xem kết quả đã lưu, tải dự án về rồi mở `web/index.html` bằng trình duyệt. Gói `web/data_bundle.js` chứa dữ liệu cần cho các bảng và hình tương tác.

## Cài đặt và chạy

Môi trường sử dụng Python và các thư viện trong `requirements.txt`. Ví dụ trên Windows với Python 3.12:

```powershell
git clone https://github.com/IzumiDKC/FashionMNIST-ADA-IM-DCSS.git
cd FashionMNIST-ADA-IM-DCSS
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web_app.py
```

Truy cập **http://127.0.0.1:8050/index.html**. Dịch vụ Python cung cấp chức năng chạy trực tiếp trên tập con 3.000 ảnh với 20 vòng tinh chỉnh. PyTorch sử dụng CUDA khi có bộ xử lý đồ họa phù hợp, hoặc bộ xử lý trung tâm.

## Thực nghiệm

Giao thức chính sử dụng 10.000 ảnh xám 28 × 28, mười cụm, ba giá trị khởi tạo 0, 1, 2, 80 vòng tiền huấn luyện và 40 vòng tinh chỉnh. Nhãn tham chiếu được dùng để đánh giá sau huấn luyện.

Ba nhóm thí nghiệm gồm đối chứng mười cấu hình, phân tích năm tổ hợp thành phần và khảo sát năm mức độ rộng khởi tạo. Các mở rộng tập trung vào:

- **Ada-DCSS:** cập nhật độ rộng RBF thích nghi theo từng cụm.
- **IM-DCSS:** cân bằng phân bố cụm và điều chỉnh entropy theo tiến độ huấn luyện.
- **Ada-IM-DCSS:** kết hợp hai hướng trên.

| Cấu hình | ACC (%) | NMI (%) | ARI (%) | ARI giữa các lần chạy |
| --- | ---: | ---: | ---: | ---: |
| DCSS (RMS) | 60,53 ± 1,16 | 63,04 ± 0,98 | 46,79 ± 1,02 | 0,8339 |
| Ada-DCSS | 60,81 ± 1,26 | 63,33 ± 0,88 | 47,24 ± 0,95 | 0,8505 |
| IM-DCSS | 62,49 ± 2,53 | 64,20 ± 1,80 | 49,18 ± 2,48 | 0,8520 |
| Ada-IM-DCSS | 61,57 ± 1,44 | 63,49 ± 1,08 | 47,94 ± 1,28 | 0,8679 |

Các chỉ số chất lượng được trình bày dưới dạng trung bình ± độ lệch chuẩn mẫu trên ba lần khởi tạo. Nguồn: `outputs_fashion_mnist/benchmark/summary.csv`.

Để chạy lại cả ba nhóm và cập nhật dữ liệu giao diện, thực hiện từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m src.run_benchmark --dataset fashion-mnist --backend pytorch --suite all --seeds 0 1 2 --pretrain-epochs 80 --finetune-epochs 40 --batch-size 256 --reuse-checkpoints --output-dir outputs_fashion_mnist
.\.venv\Scripts\python.exe -m src.build_web_data
```

Các trạng thái tiền huấn luyện đã lưu nằm trong `outputs_fashion_mnist/checkpoints/`. Bỏ `--reuse-checkpoints` để tiền huấn luyện lại. Lệnh ghi kết quả vào thư mục đầu ra đã chọn.

## Cấu trúc dự án

| Đường dẫn | Nội dung |
| --- | --- |
| `Final.docx` | Báo cáo Word |
| `web/` | Giao diện và gói dữ liệu trình diễn |
| `web_app.py` | Dịch vụ web và chạy thực nghiệm trực tiếp |
| `src/models/` | Mạng tự mã hóa và tầng phân cụm |
| `src/losses/` | Các hàm mục tiêu |
| `src/engine/` | Tiền huấn luyện, tinh chỉnh và theo dõi trạng thái |
| `src/evaluation/` | Các chỉ số đánh giá |
| `src/config.py` | Cấu hình và danh sách phương pháp |
| `data/fashion/` | 10.000 ảnh Fashion-MNIST cùng nhãn tham chiếu |
| `outputs_fashion_mnist/` | Bảng kết quả, mảng gán cụm, biểu diễn và trạng thái tiền huấn luyện |

## Nguồn dữ liệu

Fashion-MNIST do [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) công bố. Hai tệp `t10k-images-idx3-ubyte.gz` và `t10k-labels-idx1-ubyte.gz` được lưu tại `data/fashion/`, kèm [giấy phép của dữ liệu gốc](data/fashion/LICENSE).
