# DCSS, IM-DCSS và ADA-IM-DCSS trên các bộ dữ liệu bổ sung

Chốt cấu hình ngày 11/11/2026, trước khi xem kết quả huấn luyện trên năm bộ mới.
Mục tiêu là kiểm tra mức độ khái quát của kết luận Fashion-MNIST: ADA-IM tăng ARI/ACC
so với DCSS tái triển khai nhưng giảm NMI, và chưa có bằng chứng rõ rằng ADA giúp
thêm so với IM. Không chọn bộ dữ liệu, seed hoặc epoch có kết quả đẹp để kết luận.

## 1. Giữ nguyên thực nghiệm Fashion-MNIST

Thư mục có ID `1IqGB9d2gjGIVyK7kYVMbH4XcYFqZ8wSe` chỉ được đọc.
Không huấn luyện lại, sửa checkpoint hoặc ghi lại bảng thống kê trong thư mục này.
Các kết quả `paper70k` là đối chiếu lịch sử; hai hồ sơ `legacy10k` và `legacy70k`
vẫn được giữ nhưng không trộn vào so sánh mới. Kết luận trước đây không bị thay thế
bằng kết quả trên một bộ khác. Các phép thử mới có họ kiểm định riêng.

Toàn bộ dữ liệu đệm, checkpoint và báo cáo mới nằm trong
`IM-DCSS/DCSS_MultiDataset_v1`. Notebook xác định đường dẫn từ ID Drive thật,
không chỉ tìm theo tên. Hàm bảo vệ từ chối ghi vào, bên trong hoặc bên ngoài theo
quan hệ thư mục bao trùm kết quả Fashion-MNIST. Thư mục tiếp tục cũng phải nằm
trong thư mục kết quả mới.

## 2. Dữ liệu và thiết kế đánh giá

| Bộ dữ liệu | Quy mô sử dụng | Đầu vào | K | Đánh giá |
|---|---:|---|---:|---|
| EMNIST Balanced | 131.600 | Ảnh xám 28 × 28 | 47 | ARI/NMI/ACC |
| HAR | 10.299 | 561 đặc trưng đã trích xuất | 6 | ARI/NMI/ACC |
| Pendigits | 10.992 | 16 tọa độ | 10 | ARI/NMI/ACC |
| Waveform V1 | 5.000 | 21 thuộc tính | 3 | ARI/NMI/ACC |
| Nemotron Personas Vietnam | 100.000 từ 1.000.000 | Văn bản `persona` → 384 chiều | 20 | Chỉ số nội tại và độ ổn định |

EMNIST dùng **Balanced**, không gọi chung kết quả này là kết quả của ByClass.
Thư mục được cung cấp có đủ Balanced; ByClass thiếu tệp nhãn huấn luyện tại thời
điểm khảo sát. Balanced gồm 112.800 ảnh train và 18.800 ảnh test, mỗi lớp 2.800 ảnh
sau khi ghép. Ảnh được chuyển vị hai trục không gian và chia 255 khi đưa vào mạng.
Tệp ánh xạ ký tự được giữ để kiểm tra, không làm đặc trưng.

HAR ghép 7.352 và 2.947 hàng theo thứ tự train rồi test. Chỉ dùng 561 đặc trưng
đã có, biến đổi cố định `(x+1)/2`; không đưa mã người tham gia vào mạng. Kiểm tra
hai nhóm người trong phân chia gốc không giao nhau. Tuy nhiên sau khi ghép, đây
**không phải** đánh giá tổng quát hóa sang người chưa thấy. Các cửa sổ cảm biến có
thể tương quan; 10 seed đo dao động thuật toán, không tạo ra 10 quần thể độc lập.

Pendigits ghép 7.494 và 3.498 hàng; chia tọa độ cho 100. Waveform đọc tệp Unix
compress `waveform.data.Z`, đúng bản 21 thuộc tính; không tự thay bằng bản 40
thuộc tính có thêm nhiễu. Waveform dùng min–max từng cột trên toàn bộ đầu vào.
Các giá trị min/range được ghi vào manifest để tái lập.

Cả bốn bộ có nhãn được đánh giá theo thiết kế **phân cụm toàn tập** (transductive):
mọi đầu vào được đưa vào huấn luyện không giám sát; nhãn chỉ dùng kiểm tra dữ liệu
và tính chỉ số ở epoch cuối. Đây không phải điểm số trên tập kiểm thử độc lập.
K được biết từ mô tả bộ dữ liệu; không dò K bằng nhãn hay lựa chọn siêu tham số
theo ARI/NMI/ACC. Không dùng early stopping hoặc chọn checkpoint tốt nhất theo nhãn.

### Nemotron: mẫu cố định và giới hạn biểu diễn

Lấy ngẫu nhiên không hoàn lại 100.000 chỉ số từ **toàn bộ một triệu hàng thuộc
11 tệp**, bằng `numpy.random.default_rng(20261111)`, rồi sắp tăng dần. Lưu chỉ số
toàn cục, UUID và mã kiểm tra. Kiểm tra không thiếu văn bản, không trùng UUID trong
mẫu. Không lọc hoặc đổi mẫu dựa trên nội dung hay kết quả phân cụm. Đây là văn bản
tổng hợp; kết quả không đại diện trực tiếp cho phân bố người Việt ngoài thực tế.

Chỉ cột `persona` làm đầu vào. UUID, giới tính, tuổi, nghề nghiệp và các thuộc tính
khác **không phải nhãn cụm chuẩn** và không dùng để báo cáo ARI/NMI/ACC. K=20 là
lựa chọn thăm dò được chốt trước, không phải khẳng định dữ liệu có 20 nhóm tự nhiên.
Nếu muốn nghiên cứu độ nhạy theo K sau này, phải mở một cấu hình/thư mục riêng.

Bộ mã hóa cố định là `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Dùng mean pooling theo attention
mask, chuẩn hóa L2, float32, attention eager, tối đa 128 token, batch 64.
Giới hạn 128 token có thể bỏ phần cuối hồ sơ; báo cáo số hàng bị cắt và tổng token
gốc, không gọi đây là mã hóa đầy đủ mọi nội dung. Đây là một phép thử trên biểu diễn
văn bản cố định, không phải tái lập trực tiếp kiến trúc ảnh của bài DCSS.

Vector được lưu từng khối 512 hàng. Mọi phương pháp và mọi seed dùng **cùng một
tệp vector đã chốt**, kể cả khi các khối được tạo trên nhiều GPU. Nhật ký ghi lại
thiết bị/thư viện của từng khối; cache hoàn thành có SHA-256 và được dùng lại.

### Xác minh nguồn

`sources.json` ghi ID, kích thước và SHA-256 của đúng tệp Drive được cung cấp, đã
tải và kiểm tra cục bộ. Đây là dấu vân tay của bản dữ liệu người dùng cung cấp;
không tự nhận là xác minh độc lập từng byte với bản tải từ NIST/UCI/NVIDIA.
Đối chiếu kích thước, số hàng, số thuộc tính, nhãn và mô tả với nguồn chính thức.
Sai dấu vân tay, thiếu dữ liệu hoặc cache hỏng phải dừng, không lặng lẽ đổi nguồn.

## 3. Phương pháp và cấu hình so sánh

Mỗi bộ: 10 seed **0–9**, cùng thứ tự mini-batch cho các phương pháp trong mỗi seed.
Tiền huấn luyện autoencoder 100 epoch (Adam lr=0,001, weight decay=0,00001).
Khởi tạo KMeans trong không gian 10 chiều bằng 100 lần khởi tạo, Lloyd, random_state
bằng seed. Lưu cùng trọng số tiền huấn luyện và cùng tâm cho bốn biến thể:

1. **AE + KMeans**: phương pháp cơ sở, không tinh chỉnh bằng hàm mất mát phân cụm.
2. **DCSS tái triển khai**: sigma ban đầu bằng 1 và có thể học.
3. **DCSS-RMS**: đối chứng thay cách khởi tạo sigma theo khoảng cách RMS.
4. **IM-DCSS**: thêm thành phần thông tin tương hỗ, sigma có thể học.
5. **ADA-IM-DCSS**: dùng cập nhật sigma thích nghi cùng thành phần IM.

Tinh chỉnh 100 epoch, batch 256, Adam lr=0,0005, float32, không AMP/TF32, không
tự giảm batch khi hết bộ nhớ. Giữ nguyên hệ số, lịch warmup, quy tắc ADA và công
thức silhouette mềm của bộ thực thi Fashion-MNIST đã kiểm tra.

EMNIST dùng mạng tích chập/chuẩn hóa theo lô như hồ sơ `paper70k`. Ba bộ dạng bảng
và Nemotron dùng mạng d→500→500→2000→10, ReLU ở lớp ẩn, tanh ở mã 10 chiều; giải mã
đối xứng với đầu ra tuyến tính. Đây là cách thích nghi được công bố rõ, không tuyên
bố khớp từng chi tiết mạng dense của tác giả. Mọi biến thể trên cùng một bộ dùng
cùng mạng; không so sánh trực tiếp điểm giữa mạng ảnh và mạng văn bản để xếp hạng.

Với entropy tự nhiên, `Hc = mean_i[-sum_k q_ik ln q_ik]` và
`KL = sum_k qbar_k ln(K*qbar_k)`:

* DCSS/DCSS-RMS: `L = MSE + 0.01*(1-S) + 0.01*Hc`.
* IM/ADA-IM: `L = MSE + 0.012*(1-S) + 0.0025*omega*Hc + 0.004*KL`.
* omega=0 trong 70% epoch đầu, sau đó tăng nửa cosine đến 1.

Vì hệ số hai entropy khác nhau, đây là cách điều chỉnh lấy cảm hứng từ InfoMax,
không phải khẳng định đúng nguyên dạng một công thức cân bằng trọng số của SHOT.
Phép trừ **ADA-IM − IM** là so sánh chính để tách lợi ích bổ sung của ADA; so sánh
với DCSS còn bao gồm thay đổi ở thành phần mất mát và khởi tạo sigma.

## 4. Thống kê và diễn giải

ARI/NMI (arithmetic normalization)/ACC (ghép Hungarian) tính trên tất cả mẫu có
nhãn. Báo cáo trung bình và **độ lệch chuẩn mẫu (ddof=1)** qua đủ 10 seed, giữ cả
lượt suy biến. Mọi số CSV/JSON dạng tỷ lệ; 0,45 tương đương 45% đối với ACC.
Không gán 0 cho chỉ số không xác định hoặc thiếu nhãn.

Chỉ số silhouette cứng, Davies–Bouldin (DB) và Calinski–Harabasz (CH) đo trên cùng
**đầu vào cố định**, cùng tối đa 2.000 hàng chọn trước bằng seed 20261111 cho tất cả
phương pháp/seed. Không so silhouette trên không gian học khác nhau để kết luận
chất lượng tốt hơn. Với Nemotron, đây là không gian vector văn bản cố định 384 chiều.
Silhouette/CH cao hơn và DB thấp hơn là thuận lợi trong cùng bộ/biểu diễn, nhưng
không thay thế đánh giá ngữ nghĩa bởi con người. Trường hợp phân hoạch suy biến:
chỉ số không xác định là null, báo rõ số seed hợp lệ, không tự xóa lượt thất bại.

Độ ổn định được mô tả bằng ARI giữa hai phân hoạch của các seed trên cùng bộ mẫu.
45 cặp từ 10 seed phụ thuộc nhau; chỉ báo cáo mô tả, không coi là 45 quan sát độc
lập để kiểm định. Độ ổn định cao có thể đi cùng chất lượng thấp; cần xem cả số cụm
hoạt động, tỷ lệ cụm lớn nhất và phân bố kích thước cụm.

So sánh ghép cặp ADA-IM − IM dùng ARI ở bốn bộ có nhãn và silhouette chung ở
Nemotron. Báo cáo trung bình chênh lệch, CI 95% theo t (xấp xỉ, n=10) và kiểm định
đổi dấu hai phía chính xác trên 2^10 tổ hợp (giả thiết chênh lệch đối xứng dưới H0).
Hiệu chỉnh Holm trên **5 phép thử đã chốt**; bộ chưa đủ seed không được kết luận.
Các chỉ số/đối chứng khác mang tính bổ sung. Nếu CI chứa 0 hoặc thiếu bằng chứng,
ghi nhận chưa xác nhận cải thiện; không kết luận tương đương hay không có tác dụng.
Các lần chạy trên GPU/phần mềm khác nhau được đánh dấu để thảo luận, đặc biệt khi
so sánh thời gian. Báo cáo không tự động tuyên bố luận văn thành công.

## 5. Tiếp tục qua nhiều phiên và nhiều thiết bị

Chọn Python 3 và A100 khi có. Có thể dùng T4/L4/G4/H100 hoặc CUDA GPU khác mà bản
PyTorch hiện tại hỗ trợ; CPU cũng dùng được nhưng chậm hơn nhiều. TPU chưa được
hỗ trợ. Notebook không tự xin/đổi GPU và không vượt hạn mức Colab.

Sau khi đổi runtime, **Run all**, gắn cùng tài khoản Drive, giữ nguyên thư mục và
cấu hình khoa học. Dùng checkpoint CPU-portable chứa model, optimizer, tâm, sigma,
trạng thái ngẫu nhiên, epoch/batch tiếp theo, lịch sử và nhật ký thiết bị. Ghi sau
mỗi epoch và khi đủ 300 giây tại ranh giới batch, hai phiên bản có checksum để
khôi phục bản trước nếu ghi dở. Lượt hoàn thành được kiểm tra checksum và bỏ qua.

Không có bảo đảm rằng batch đang xử lý được lưu khi Colab tắt đột ngột. Phần sau
checkpoint hợp lệ gần nhất có thể chạy lại; thời gian mất tối đa còn phụ thuộc
batch, bước suy luận cuối epoch và tốc độ ghi Drive. KMeans và đánh giá cuối đang
thực hiện có thể chạy lại nếu chưa hoàn tất; tiền huấn luyện đã lưu không chạy lại.
Khối mã hóa văn bản chưa ghi xong sẽ được tạo lại. Không hứa bitwise giống nhau
giữa GPU/CPU hay phiên bản thư viện, dù cùng quy trình và trạng thái tiếp tục.

Không mở hai phiên huấn luyện cùng ghi một thư mục. Nếu hết dung lượng Drive,
thiếu quyền hoặc checksum không khớp, dừng và giữ file để kiểm tra. Thay dataset,
hàm mất mát, mạng, batch, seed hoặc mã tính toán làm thay đổi danh tính thí nghiệm;
không nối vào checkpoint cũ một cách âm thầm. Muốn thử cấu hình khác dùng thư mục mới.
Không xóa checkpoint/cache trước khi hoàn thành và lưu trữ kết quả.

## 6. Nguồn chính

* DCSS: https://link.springer.com/article/10.1007/s10994-026-07026-w
* Mã tác giả được kiểm tra trước đây: commit `fa3d84c0f3e7c682729cd5ba6f856bb3ef532637`.
* RIM: https://proceedings.neurips.cc/paper/2010/hash/42998cf32d552343bc8e460416382dca-Abstract.html
* SHOT: https://proceedings.mlr.press/v119/liang20a.html
* EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
* HAR: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
* Pendigits: https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
* Waveform: https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1
* Nemotron: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam
* Bộ mã hóa: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
* PyTorch, tính tái lập: https://docs.pytorch.org/docs/stable/notes/randomness.html
