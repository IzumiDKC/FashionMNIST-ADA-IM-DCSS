# Kiểm tra thực nghiệm hoàn tất — 15/11/2026

**Kết luận: cả năm bộ dữ liệu bổ sung đã chạy hoàn tất.** Lượt cuối cùng
(Nemotron) kết thúc lúc **20:39:29 ngày 14/11/2026, giờ Việt Nam**.
Không cần huấn luyện lại để hoàn thành cấu hình đã chốt.

Nguồn kiểm tra: [thư mục kết quả trên Drive](https://drive.google.com/drive/folders/1m6AcyZC228ULO5UejFy8pI9tVMCxsqR9)
và [notebook đã chạy](https://colab.research.google.com/drive/1im9fG_949mpbZOIxdOp_nQ0M3VfLgenV).
Notebook được lưu lần cuối lúc 20:42 ngày 14/11/2026.

## Mức độ hoàn thành

| Bộ dữ liệu | Số mẫu | Seed | Kết quả đã lưu | Hoàn tất lần đầu, giờ Việt Nam |
|---|---:|---|---:|---|
| EMNIST Balanced | 131.600 | 0–9 | 50/50 | 14/11, 15:16 |
| HAR | 10.299 | 0–9 | 50/50 | 11/11, 23:17 |
| Pendigits | 10.992 | 0–9 | 50/50 | 12/11, 00:12 |
| Waveform V1 | 5.000 | 0–9 | 50/50 | 12/11, 00:52 |
| Nemotron | 100.000 cố định | 0–9 | 50/50 | 14/11, 20:39 |

Mỗi bộ có 10 seed × 5 phương pháp: AE + KMeans, DCSS, DCSS-RMS, IM-DCSS và
ADA-IM-DCSS. Tổng cộng **250 bản kết quả**, gồm 50 kết quả cơ sở và 200 lượt
tinh chỉnh. Trường expected_runs=40 trong status.json chỉ đếm bốn biến thể
tinh chỉnh; completeness.json đếm cả AE + KMeans nên có 50. Đây không phải thiếu lượt.

Đã đối chiếu:

* Đủ 250 tệp result.json, không trùng hoặc thiếu cặp seed–phương pháp.
* Đủ 50 giai đoạn tiền huấn luyện và 200 giai đoạn tinh chỉnh; mỗi giai đoạn có
  100 epoch. Tổng cộng **25.000 bản ghi epoch**, loss hữu hạn và số mẫu đúng ở mỗi epoch.
* Các chỉ mục checkpoint kết thúc ở epoch 100, batch tiếp theo bằng 0; danh tính
  thí nghiệm khớp với manifest và từng kết quả.
* Cả năm status.json đều complete; cả năm completeness.json đều 50/50, missing=[].
  Đã tải trực tiếp các tệp trạng thái trên Drive và xác nhận chúng khớp gói ZIP.
* Dấu vân tay đặc trưng/nhãn của bốn bộ có nhãn khớp dữ liệu đã kiểm tra trước.
  Các chỉ số hàng và UUID của Nemotron khớp mẫu 100.000 đã chốt.
* Cùng mẫu đánh giá nội tại trong mọi phương pháp và seed của từng bộ.
* Tính lại trung bình, SD mẫu, chênh lệch ghép cặp, p đổi dấu và hiệu chỉnh Holm
  từ các result.json: khớp bảng đã xuất. Sai khác CI lớn nhất khoảng 1,2 × 10⁻¹²
  khi tính lại giữa hai môi trường, dưới ngưỡng số học đã đặt 10⁻¹⁰.
* Notebook đã lưu không có đầu ra loại error; ô thống kê và xuất ZIP đã chạy.

Một số nhật ký phiên cũ vẫn mang trạng thái running do phiên bị ngắt trước khi
ghi kết thúc. Trạng thái cuối, đủ kết quả và đủ lịch sử epoch xác nhận rằng các
phiên sau đã tiếp tục và hoàn thành; các bản ghi cũ không phải bằng chứng thiếu tiến độ.

Phạm vi kiểm tra này là hồ sơ đã lưu, lịch sử, chỉ mục checkpoint, cấu hình và
thống kê. Chưa tải toàn bộ tensor checkpoint hoặc tính lại mọi chỉ số từ
predictions.npz. Không sửa hoặc huấn luyện lại kết quả trên Drive.

## Những điểm cần giữ lại khi viết luận văn

Hoàn thành đủ lượt chạy không đồng nghĩa mọi lượt đều tạo ra phân hoạch tốt.
Các trường hợp không thuận lợi dưới đây phải được giữ trong báo cáo.

| Bộ / phương pháp | Lượt bị đánh dấu suy biến | Số cụm hoạt động | Lượt có silhouette không xác định |
|---|---:|---|---:|
| Waveform — DCSS | 2/10 | 1–3 | 2 |
| Waveform — DCSS-RMS | 1/10 | 1–3 | 1 |
| Waveform — IM-DCSS | 2/10 | 1–3 | 1 |
| Waveform — ADA-IM-DCSS | 1/10 | 1–3 | 1 |
| Nemotron — DCSS | 10/10 | 1–2 thay vì K=20 | 1 |
| Nemotron — DCSS-RMS | 10/10 | 2–3 thay vì K=20 | 0 |

Tổng cộng 26/200 lượt tinh chỉnh được bộ chẩn đoán hiện tại đánh dấu suy biến.
Cờ này dùng ngưỡng đã chốt, không phải định nghĩa đầy đủ về mọi dạng phân cụm kém.
Ví dụ ADA-IM trên Nemotron không bị gắn cờ, nhưng chỉ có 9–16 cụm hoạt động;
IM có 16–20 cụm. Trên HAR, DCSS chỉ có 3–4 cụm hoạt động khi K=6.

Các chỉ số nội tại không xác định được để trống. Trung bình silhouette của một số
phương pháp vì thế có ít hơn 10 seed hợp lệ; không diễn giải như đủ 10 giá trị,
không thay giá trị thiếu bằng 0 và không loại lượt suy biến khỏi ARI/NMI/ACC.

HAR có lịch sử pha trộn CPU và NVIDIA L4. Bốn bộ còn lại ghi NVIDIA L4 trong
lịch sử thực thi kết quả. Cần giữ thông tin này khi so sánh độ ổn định và thời gian.
Nemotron không có nhãn chuẩn, nên không có ARI/NMI/ACC theo nhãn đúng.

## Đối chiếu sơ bộ lợi ích bổ sung của ADA

So sánh chính đã chốt là ADA-IM trừ IM, ghép theo 10 seed. ARI ở bốn hàng đầu
được biểu diễn bằng điểm phần trăm; silhouette ở hàng cuối giữ đơn vị gốc.

| Bộ | Thước đo | Chênh lệch trung bình ADA-IM − IM | p sau Holm, họ 5 phép thử |
|---|---|---:|---:|
| EMNIST Balanced | ARI | −1,079 điểm phần trăm | 0,0586 |
| HAR | ARI | −1,259 điểm phần trăm | 1,0000 |
| Pendigits | ARI | +0,182 điểm phần trăm | 0,4844 |
| Waveform | ARI | +1,199 điểm phần trăm | 0,7500 |
| Nemotron | Silhouette chung | +0,00155 | 1,0000 |

**Với giao thức và năm bộ này, chưa xác nhận lợi ích bổ sung rõ ràng của ADA so
với IM sau hiệu chỉnh đa kiểm định ở mức 0,05.** EMNIST có chênh lệch trung bình
bất lợi cho ADA; các dấu hiệu thuận lợi ở Pendigits, Waveform và Nemotron chưa
đủ để khẳng định cải thiện. Kết luận này không đồng nghĩa đã chứng minh hai
phương pháp tương đương, cũng không thay thế so sánh riêng với DCSS hoặc AE + KMeans.

Kết quả Fashion-MNIST trước đó tiếp tục được giữ làm đối chiếu lịch sử, không
gộp vào họ kiểm định của năm bộ bổ sung.

## Tệp đối chiếu

* [Gói báo cáo gốc trên Drive](https://drive.google.com/file/d/1S6ya7fK2Y15IW1KfMs5Bj0d9agZbOTwv/view).
* [Bảng tổng hợp trên Drive](https://drive.google.com/file/d/1eYbaeEHJmnswQf10SezgPjcBfHljJZ_Y/view).
* [Biên bản kiểm tra dạng JSON](audit/results_2026_11_15/completion_audit.json).
* [Bảng hoàn thành](audit/results_2026_11_15/completion_audit.csv).
* [Bảng tình trạng cụm từng phương pháp](audit/results_2026_11_15/cluster_health_audit.csv).
* [Bản báo cáo HTML tải về](audit/results_2026_11_15/review/analysis/ThongKe_DCSS_MultiDataset.html).

SHA-256 của ZIP đã kiểm tra:
994674bd13f021f9c266873b5e1f7415ac8dbfac0e5eb60fb2a36e7b2cf6b94b.
