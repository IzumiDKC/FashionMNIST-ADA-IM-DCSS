# Giao thức kiểm chứng DCSS và ADA-IM-DCSS trên Fashion-MNIST

Chốt ngày 09/11/2026, trước khi chạy thực nghiệm GPU đầy đủ. Tệp này mô tả
những gì chương trình thực hiện và giới hạn của kết luận được phép rút ra.

## 1. Câu hỏi và phạm vi kết luận

Mục tiêu thứ nhất là đối chiếu DCSS trên đủ 70.000 ảnh với bài báo.
Mục tiêu thứ hai là kiểm tra ADA-IM-DCSS có duy trì lợi ích khi tăng quy mô dữ
liệu, đồng thời xác định thành phần ADA có đem lại lợi ích so với chỉ dùng IM.
Hai mục tiêu cần các phép so sánh riêng, vì thay đổi đồng thời kiến trúc, số
epoch và số ảnh sẽ không cho biết yếu tố nào gây ra thay đổi kết quả.

Notebook mặc định chạy ba hồ sơ cấu hình, theo thứ tự:

| Hồ sơ | Ảnh | Kiến trúc | Tiền huấn luyện / phân cụm | Seed | Phương pháp |
|---|---:|---|---:|---|---|
| `paper70k` | 70.000 | Conv/BN theo mã tác giả, khởi tạo He theo bài báo | 100 / 100 | 0–9 | DCSS theo phương trình; DCSS-RMS; IM-DCSS; ADA-IM-DCSS |
| `legacy10k` | 10.000 ảnh test chính thức | Conv không BN, đầu ra sigmoid theo bản cục bộ | 80 / 40 | 0, 1, 2 | DCSS-RMS; IM-DCSS; ADA-IM-DCSS |
| `legacy70k` | 70.000 | Giống `legacy10k` | 80 / 40 | 0, 1, 2 | Giống `legacy10k` |

Tổng cộng 58 lượt phân cụm và 16 lượt tiền huấn luyện dùng chung. Đây là một
chiến dịch nhiều lượt, có thể cần nhiều phiên Colab. Chưa có thời lượng T4 đo
thực tế; chương trình chỉ ước lượng thời gian còn lại của pha hiện tại sau khi
đã đo epoch. Không hạ số epoch/seed âm thầm để hoàn thành trong một phiên.

`legacy10k` được chạy mới bằng cùng bộ thực thi với `legacy70k`. Nó kế thừa
kiến trúc, siêu tham số và hàm mục tiêu cũ, nhưng lịch trộn mẫu, việc bỏ nhánh
residual không dùng và trạng thái ngẫu nhiên khác bộ thực thi cũ. Vì vậy không
đòi hỏi tái tạo đúng từng chữ số của bảng kết quả cũ. Hai lượt chạy mới mới là
cặp phù hợp để đánh giá tác động của tăng quy mô dữ liệu.

Tập 70.000 được nối theo thứ tự **train 60.000 → test 10.000**. Huấn luyện
không dùng nhãn; nhãn chỉ kiểm tra dữ liệu và tính chỉ số sau huấn luyện.
Đây là phân cụm chuyển dẫn (*transductive clustering*), không phải thử nghiệm
khái quát hóa trên tập kiểm tra độc lập. Hơn nữa, 10.000 ảnh đã dùng trước đây
để phát triển phương pháp nằm trong bộ 70.000; không được gọi bộ 70.000 là tập
chưa từng được sử dụng trong quá trình nghiên cứu.

## 2. Đối chiếu bài báo, mã tác giả và lựa chọn triển khai

Nguồn chính:

- [DCSS, Machine Learning 115:81, 2026](https://link.springer.com/article/10.1007/s10994-026-07026-w), phương trình (13)–(22), mục 5.3 và 5.5, bảng 2.
- [Mã tác giả tại commit cố định](https://github.com/gvardakas/Soft-Silhouette/tree/fa3d84c0f3e7c682729cd5ba6f856bb3ef532637).
- [RIM: Discriminative Clustering by Regularized Information Maximization, NeurIPS 2010](https://proceedings.neurips.cc/paper/2010/hash/42998cf32d552343bc8e460416382dca-Abstract.html).
- [SHOT, ICML 2020, phương trình (3)](https://proceedings.mlr.press/v119/liang20a/liang20a.pdf).
- [Fashion-MNIST chính thức](https://github.com/zalandoresearch/fashion-mnist).

Đường dẫn RIM được cung cấp ban đầu có mã băm khác; liên kết phía trên là trang
của đúng bài của Krause, Perona và Gomes. RIM cung cấp cơ sở tối đa hóa thông
tin tương hỗ cho phân cụm; SHOT sử dụng entropy có điều kiện và entropy biên
trong thích nghi miền. Không xem RIM/SHOT là chứng minh cho các hệ số, lịch
warmup hay quy tắc băng thông ADA cụ thể của đề xuất này.

| Nội dung | Bài báo / mã công bố | Quyết định trong `paper70k` |
|---|---|---|
| Soft silhouette | (13) chia khoảng cách có trọng số cho tổng xác suất, loại mẫu đang xét khỏi mẫu số. `Objectives.py` công bố thiếu phép chia này và có cách xếp chỉ số riêng. | Theo (13)–(17), có kiểm tra bằng phép tính vòng lặp độc lập và kiểm tra gradient. |
| Softmax RBF | Bài báo: nhân hệ số T=20. `forward_softMax` công bố mặc định chia 0,25, tương đương T=4. | Chia 0,05, tương đương T=20. |
| Băng thông | Bài báo: giá trị dương nhỏ, tham số được học; không chỉ rõ trị số. Mã công bố khởi tạo log_sigma=0. | `dcss_paper`: sigma=1 rồi học log_sigma. Không gọi đây là sigma cố định. |
| Kiến trúc ảnh | Mã tác giả có BatchNorm sau các khối và sau tanh, kể cả đầu ra giải mã. Bản cục bộ không có BN, đầu ra sigmoid. | Dùng cấu trúc Conv/BN của mã tác giả cho cả bốn phương pháp; không trộn hai kiến trúc trong cùng hồ sơ. |
| Khởi tạo mạng | Bài báo nêu He; mã lớp công bố không có lời gọi He rõ ràng. | He cho Conv/Linear; lựa chọn được ghi trong manifest. |
| MSE | Công thức mô tả tổng sai số trên chiều; mã tác giả dùng `nn.MSELoss` lấy trung bình cả pixel. | Trung bình pixel theo mã, tránh thay tỷ lệ 784 lần mà không công bố. |
| Entropy | Bài báo không nêu cơ số log một cách tường minh; mã dùng cơ số 2. | Log tự nhiên, trung bình mẫu, theo công thức thường dùng cho IM và bản cục bộ. Chưa coi tương đương hệ số với entropy cơ số 2. |
| Tiền xử lý | Không có bộ nạp Fashion-MNIST đầy đủ trong các tệp tác giả đã kiểm tra; loader EMNIST có MinMaxScaler. | Pixel / 255, không tăng cường ảnh, không chuẩn hóa từng ảnh. Đây là lựa chọn triển khai được công bố. |
| Đánh giá và BN | Mã công bố thu dự đoán trong quá trình cập nhật theo batch. | Một lượt `eval()` sau epoch cuối; không cập nhật BN/ADA khi đánh giá. |

Vì các bất nhất trên, tên đúng của kết quả là **tái triển khai DCSS theo phương
trình với các lựa chọn được công bố**, chưa phải tái lập nguyên trạng chương
trình đã sinh bảng 2. Nếu kết quả lệch bảng 2, cần đối chiếu những lựa chọn này
với tác giả trước khi kết luận phương pháp gốc thất bại. Không hứa đạt lại
điểm công bố. Bảng 2 báo cáo Fashion-MNIST NMI 0,63 ± 0,02 và ARI 0,45 ± 0,03;
đây chỉ là mốc tham khảo, không được đưa vào cột kết quả thực nghiệm mới.

Có thêm `author_objective_control` trong mã để chẩn đoán khác biệt: giữ biểu
thức/indexing soft silhouette của mã công bố, entropy cơ số 2 và temperature
0,25. Đây vẫn là phép kiểm soát hàm mục tiêu trên bộ thực thi chung, **không**
được ghi là chạy nguyên xi toàn bộ repository tác giả. Biến thể này tắt mặc
định; nếu bật phải tạo hồ sơ mới, không thay thế kết quả chính sau khi xem điểm.

## 3. Hàm mục tiêu và nội dung đề xuất

Với batch B ảnh, K=10, q là xác suất phân cụm:

```text
a_ik = sum_{j != i} q_jk ||z_i-z_j|| / max(sum_{j != i} q_jk, epsilon)
b_ik = min_{l != k} a_il
S    = mean_i sum_k q_ik (b_ik-a_ik) / max(a_ik,b_ik,epsilon)
q_ik = softmax_k(exp(-||z_i-c_k||² / sigma_k²) / 0.05)
Hc   = -mean_i sum_k q_ik ln(q_ik)
qbar = mean_i q_i
KL   = sum_k qbar_k ln(K*qbar_k)
```

DCSS: `L = MSE + 0.01*(1-S) + 0.01*Hc`.

IM-DCSS và ADA-IM-DCSS:
`L = MSE + 0.012*(1-S) + 0.0025*omega(t)*Hc + 0.004*KL`.
`omega=0` trong 70% đầu số epoch phân cụm, sau đó tăng theo nửa đường cosine
đến 1 ở epoch cuối. KL nhỏ khuyến khích phân bố cụm cân bằng; Hc nhỏ khuyến
khích gán cụm dứt khoát. Với hệ số bằng nhau và không warmup, tổng hai thành
phần entropy tương đương âm thông tin tương hỗ cộng hằng số. Ở cấu hình hiện
tại, cần gọi là **điều chuẩn IM có trọng số và lịch điều chỉnh**, không gọi là
đúng nguyên dạng hàm âm thông tin tương hỗ của RIM/SHOT.

ADA giữ cơ chế đang có: tâm cụm học bằng gradient; sigma khởi tạo theo phân
tán từng cụm, cập nhật bằng EMA hệ số 0,15; mục tiêu sigma kết hợp RMS toàn cục,
RMS nội cụm mềm và hiệu chỉnh khối lượng cụm nhẹ. Ngưỡng bảo vệ trung bình RBF
là 0,08 / 0,78. Sau epoch 1 và 3, chỉ nếu tiêu chí không nhãn phát hiện suy biến
thì đặt lại sigma bằng RMS. Mọi lần bảo vệ/phục hồi đều được đếm và ghi lại.
EMA dùng thống kê trước bước tối ưu nhưng thực hiện cập nhật buffer sau
`backward/step`, để tránh sửa trạng thái đang được autograd sử dụng.

`normalize_latent_scale` trong cấu hình cũ chỉ được lưu mà không có phép chuẩn
hóa thực sự trong forward. Bản này không tuyên bố đã thêm chuẩn hóa latent.
Các ngưỡng, warmup và hệ số ADA được giữ từ bản cũ; chưa có chứng cứ rằng đó là
cấu hình tối ưu hoặc là đóng góp mới đã được xác nhận bởi tổng quan tài liệu.

DCSS-RMS là đối chứng cần thiết: nó cũng khởi tạo sigma theo RMS rồi học bằng
gradient, giới hạn [0,15; 4,5], nhưng không có EMA/guard/rescue. Nhờ đối chứng
này, việc thắng DCSS sigma khởi tạo 1 không bị hiểu nhầm thành bằng chứng toàn
bộ lợi ích đến từ ADA. Đặc biệt, ADA-IM so với IM-DCSS giữ nguyên các hệ số hàm
mục tiêu; cặp này kiểm tra đóng góp của **cả gói ADA** gồm khởi tạo, EMA, guard
và rescue. Chưa cô lập riêng từng thành phần của gói ADA. So DCSS với IM còn
thay đổi hệ số silhouette/entropy, nên chưa phải phép loại bỏ thành phần IM
thuần túy. `ada_only` có sẵn như một đối chứng bổ sung, tắt mặc định.

## 4. Kiểm soát thực nghiệm và tài nguyên

Mỗi seed tiền huấn luyện đúng một lần, lấy K-means với `n_init=100`, rồi chia
cùng trọng số và tâm khởi tạo cho các phương pháp của seed đó. Thứ tự batch
cũng giống nhau giữa các phương pháp. Adam dùng beta=(0,9; 0,999), LR tiền
huấn luyện 1e-3 và weight decay 1e-5; LR phân cụm 5e-4, không weight decay.
Batch 256, latent 10. Hồ sơ paper không cắt gradient; hồ sơ legacy cắt norm
5 ở pha phân cụm theo bản cũ. Không early stopping, không chọn seed tốt nhất,
không chọn checkpoint theo nhãn thật, không điều chỉnh hệ số từ kết quả 70k.

Ảnh lưu uint8 trên RAM máy; chỉ batch hiện tại được đổi float32 và chuyển GPU.
Soft silhouette tính trong batch, không tạo ma trận 70.000 × 70.000. Khoảng
cách N × K chỉ dùng khi khởi tạo/chẩn đoán tâm; đánh giá encoder cũng theo
batch. FP32, AMP/TF32 tắt; bật thuật toán xác định. Nếu CUDA báo thao tác không
hỗ trợ chế độ xác định, chương trình dừng thay vì tự đổi chế độ. Mỗi epoch lưu
thời gian, mức sử dụng VRAM đỉnh của PyTorch và chẩn đoán không nhãn.

Giữ bản torch/CUDA đang cài trên Colab. Không ép hạ cấp hoặc cài lại torch,
torchvision. Lưu phiên bản Python, torch, CUDA, cuDNN, GPU và thư viện tính
toán thực tế. Chỉ được gọi là kiểm chứng trên T4 sau khi phần nhật ký GPU có
thông tin T4 thật; số liệu kiểm tra CPU không được thay thế cho số liệu này.

## 5. Đánh giá đã định trước

Chỉ số chính: **ARI**. Giả thuyết chính của phần cải tiến là chênh lệch ARI
`ADA-IM − IM-DCSS` trên 10 seed của `paper70k`. ACC (ghép nhãn bằng Hungarian
chính xác) và NMI (trung bình số học) là chỉ số phụ. So với DCSS-paper và
DCSS-RMS là đối chiếu bổ sung; không dùng nhiều phép so sánh rồi chỉ báo cáo
phép có lợi. Nếu muốn kết luận xác nhận cho nhiều giả thuyết thì cần hiệu
chỉnh đa kiểm định trong một giao thức bổ sung.

Báo cáo mọi seed gồm cả suy biến: trung bình, độ lệch chuẩn mẫu, chênh lệch
theo cặp seed, khoảng tin cậy t 95%, kiểm định đổi dấu chính xác hai phía.
Khoảng tin cậy mô tả biến thiên do seed trên tập dữ liệu cố định, không phải
độ bất định khi lấy mẫu một quần thể ảnh mới. Với 3 seed legacy, chỉ số thống
kê mang tính thăm dò; không đủ để tuyên bố cải tiến vững chắc.

Không kết luận ADA có lợi chỉ vì trung bình cao hơn một chút. Cần xem khoảng
tin cậy chênh lệch chính, tỷ lệ suy biến, từng seed, thời gian và VRAM. Nếu ADA
không hơn IM, cần báo cáo đúng và xem lại độ cần thiết của thành phần ADA.
Nếu chưa đủ seed, báo cáo là **chưa hoàn tất**. Không loại lượt chạy kém, không
ghi lượt lỗi thành điểm 0 hoặc bỏ qua lỗi mà vẫn coi chiến dịch thành công.

## 6. Dữ liệu và bằng chứng đầu ra

Bốn tệp Drive được so byte bằng SHA-256 với bản giải nén từ kho Fashion-MNIST
chính thức; các bản nén chính thức được kiểm tra MD5 theo bảng phát hành.
`audit/dataset_verification.json` ghi kết quả so khớp, magic/header/chiều dài
IDX và số lượng lớp. Mỗi lần chạy kiểm tra lại SHA-256, số ảnh và thứ tự nối.

Mỗi thư mục kết quả có `manifest.json`, bản mã `experiment.py`, `status.json`,
`completeness.json`, `per_seed.csv`, `summary.csv`, `paired_comparisons.json`.
Trong từng seed có trọng số tiền huấn luyện, tâm K-means dùng chung; mỗi phương
pháp có checkpoint, `history.json`, `result.json`, `predictions.npz` gồm dự
đoán, xác suất, embedding và nhãn phục vụ kiểm tra độc lập. Đơn vị chỉ số trong
CSV/JSON là tỷ lệ 0–1, khi vẽ mới nhân 100. ARI có thể âm.

Checkpoint lưu model, buffer BN/ADA, optimizer, RNG, epoch tiếp theo và lịch sử.
Định danh gắn với mã, cấu hình, dữ liệu và môi trường. Chỉ tiếp tục checkpoint
cùng định danh; cấu hình hoặc phiên bản thay đổi tạo hồ sơ khác, không ghép
kết quả. Ngắt giữa epoch sẽ chạy lại epoch đó từ checkpoint hoàn chỉnh gần
nhất. Nếu GPU hết bộ nhớ/NaN, trạng thái lỗi được ghi rồi dừng; không tự giảm
batch. Người dùng sửa cấu hình phải coi đó là thử nghiệm khác.

## 7. Những việc còn cần trước khi đưa vào luận văn

Chưa có kết quả thực nghiệm GPU đầy đủ ở thời điểm bàn giao notebook. Cần chạy
đủ các hồ sơ, kiểm tra `complete=true`, rồi phân tích chênh lệch. Bất nhất giữa
bài báo và mã tác giả cần được trình bày trung thực trong phần tái lập; có thể
xin tác giả xác nhận tiền xử lý, sigma, cơ số entropy, MSE, BN và mã đã sinh
bảng kết quả. Sau đó mới quyết định có cần một lượt tái lập bổ sung.

Kết quả trên một bộ dữ liệu chưa đủ khẳng định phương pháp tổng quát hay tính
mới. Bài toán mất cân bằng lớp, các bộ ảnh khác và phân tích riêng từng thành
phần ADA là những kiểm chứng tiếp theo, không được coi là đã hoàn thành chỉ
bằng Fashion-MNIST cân bằng.
