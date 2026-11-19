# Kết quả kiểm chứng DCSS và ADA-IM-DCSS

Đã đọc 58 lượt hoàn tất; tính lại ACC/NMI/ARI từ 58 tệp dự đoán. Không huấn luyện lại, không chọn seed tốt nhất.

Các bảng dùng trung bình ± độ lệch chuẩn mẫu qua seed. Giá trị nhân 100; chênh lệch tính theo điểm phần trăm. ARI không phải tỷ lệ ảnh được phân loại đúng.

## paper70k

| Phương pháp | Số seed | ARI ×100 (TB ± ĐLC mẫu) | NMI ×100 (TB ± ĐLC mẫu) | ACC ×100 (TB ± ĐLC mẫu) |
| --- | --- | --- | --- | --- |
| DCSS theo phương trình | 10 | 45.210 ± 2.100 | 64.804 ± 1.723 | 57.513 ± 2.821 |
| DCSS-RMS | 10 | 45.447 ± 1.841 | 65.089 ± 1.584 | 57.675 ± 2.488 |
| IM-DCSS | 10 | 46.126 ± 2.719 | 63.022 ± 1.743 | 58.096 ± 3.351 |
| ADA-IM-DCSS | 10 | 46.189 ± 2.689 | 63.125 ± 1.752 | 58.125 ± 3.330 |

## legacy10k

| Phương pháp | Số seed | ARI ×100 (TB ± ĐLC mẫu) | NMI ×100 (TB ± ĐLC mẫu) | ACC ×100 (TB ± ĐLC mẫu) |
| --- | --- | --- | --- | --- |
| DCSS-RMS | 3 | 46.711 ± 1.012 | 63.739 ± 0.569 | 59.967 ± 1.789 |
| IM-DCSS | 3 | 47.504 ± 1.134 | 63.907 ± 0.363 | 60.600 ± 2.399 |
| ADA-IM-DCSS | 3 | 47.483 ± 1.084 | 63.945 ± 0.389 | 60.563 ± 2.367 |

## legacy70k

| Phương pháp | Số seed | ARI ×100 (TB ± ĐLC mẫu) | NMI ×100 (TB ± ĐLC mẫu) | ACC ×100 (TB ± ĐLC mẫu) |
| --- | --- | --- | --- | --- |
| DCSS-RMS | 3 | 47.408 ± 0.888 | 65.913 ± 0.930 | 59.163 ± 0.996 |
| IM-DCSS | 3 | 48.643 ± 1.203 | 65.667 ± 1.226 | 61.293 ± 2.542 |
| ADA-IM-DCSS | 3 | 48.592 ± 1.291 | 65.757 ± 1.252 | 61.305 ± 2.708 |

## Câu hỏi chính: ADA có đem lại lợi ích thêm so với IM?

Trên paper70k (10 seed), ΔARI = ADA-IM − IM = **+0.0629 điểm phần trăm**. KTC 95%: [-0.2046; +0.3304]; p đổi dấu hai phía = 0.599609.

Khoảng tin cậy chứa 0: **chưa có bằng chứng rõ rằng thêm gói ADA cải thiện ARI so với IM-DCSS**. Không đồng nghĩa với chứng minh hai phương pháp tương đương; cần một biên tương đương xác định trước và kiểm định tương ứng nếu muốn kết luận đó.

So với DCSS theo phương trình, ADA-IM thay đổi ARI +0.9786, NMI -1.6782 và ACC +0.6121 điểm phần trăm. Không có lợi ích đồng đều nếu các chỉ số đi theo chiều khác nhau. Phép so sánh này đồng thời đổi hệ số loss và cơ chế sigma, nên không quy toàn bộ chênh lệch cho ADA.

## Tăng từ 10.000 lên 70.000 ảnh với cấu hình legacy

Phép này giữ kiến trúc và siêu tham số legacy; không so chéo với backbone Conv/BN của paper70k. Hai bộ dữ liệu có phần giao nhau và chỉ có 3 seed, nên đây là kiểm tra quy mô mang tính thăm dò.

Đối với ADA-IM: ARI: 47.483 → 48.592, Δ=+1.109 điểm phần trăm; NMI: 63.945 → 65.757, Δ=+1.812 điểm phần trăm; ACC: 60.563 → 61.305, Δ=+0.741 điểm phần trăm.

Đánh giá lợi ích ADA phải tiếp tục đối chiếu với IM-DCSS trên cùng 70k, không chỉ so mức điểm với kết quả 10k cũ.

## Chẩn đoán và nguồn gốc môi trường

Có 0/58 lượt bị gắn suy biến ở đánh giá cuối. Bộ phát hiện suy biến có ngưỡng riêng; không suy biến không đồng nghĩa phân cụm tốt theo nhãn.

Ở ADA-IM, tổng guard_count=0, rescue_count=0. Các số này chỉ cho biết nhánh bảo vệ/phục hồi có kích hoạt hay không; không chứng minh EMA vô tác dụng.

Các môi trường quan sát: NVIDIA A100-SXM4-40GB; NVIDIA A100-SXM4-40GB → Tesla T4; Tesla T4. Lượt V1 thiếu execution_trace chỉ được gán môi trường từ manifest, chưa có nhật ký đầy đủ từng đoạn. Bảng environment_sensitivity.csv chỉ giữ cặp có trace đầy đủ, một môi trường trùng nhau, là phân tích nhạy cảm mô tả bổ sung; không thay thế toàn bộ 10 seed.

Không kết luận tốc độ thuật toán từ thời gian T4/A100/CPU pha trộn. Peak GPU là bộ nhớ PyTorch cấp phát, không phải toàn bộ VRAM của phiên. Các thống kê không dùng nhãn để chọn checkpoint; nhãn ở đây chỉ phục vụ kiểm tra hậu nghiệm.

## Giới hạn diễn giải cho luận văn

Bộ 70k gồm train và test chính thức, là phân cụm chuyển dẫn; không phải kiểm tra tổng quát hóa trên ảnh chưa dùng. 10k đã tham gia phát triển phương pháp cũng nằm trong 70k. Khoảng tin cậy qua seed phản ánh ngẫu nhiên của thuật toán trên tập cố định, không phải bất định khi lấy mẫu ảnh mới.

DCSS-paper là tái triển khai theo phương trình với các lựa chọn đã công bố trong PROTOCOL.md; không gọi là chạy nguyên trạng mã sinh bảng của bài báo. Không diễn giải ARI, NMI và ACC như cùng một đại lượng. Các p phụ được hiệu chỉnh Holm trong từng profile; giả thuyết chính ARI ADA-IM − IM được giữ riêng theo giao thức.

**Kết luận thực dụng:** Đã đủ các lượt khai báo để báo cáo mức điểm của chiến dịch này. Nếu chênh lệch ADA so với IM còn nhỏ và KTC chứa 0, chưa nên lấy việc thêm ADA làm bằng chứng cải tiến rõ ràng cho luận văn. Giữ kết quả trung thực; nếu nghiên cứu tiếp, cần đối chứng tách từng thành phần và kiểm chứng trên bộ dữ liệu/điều kiện khác.
