# Kế hoạch 4 tuần: Food-101 Classification + Web Demo

## 1. Mục tiêu và tiêu chí hoàn thành

Xây dựng hệ thống phân loại một món ăn chính trong ảnh vào một trong 101 lớp Food-101, sử dụng duy nhất ảnh trong `food-101/images`.

Sản phẩm cuối gồm:

- Pipeline kiểm tra dữ liệu, huấn luyện, đánh giá và tái lập thí nghiệm.
- Benchmark công bằng giữa ResNet18 và EfficientNet-B0 pretrained ImageNet.
- Model tốt nhất cùng metadata, class mapping và cấu hình preprocessing.
- FastAPI inference service và Streamlit web demo chạy local.
- Báo cáo kết quả, confusion matrix và phân tích các lớp dễ nhầm.

Mục tiêu kỹ thuật:

- EfficientNet-B0 đạt Top-1 test accuracy mục tiêu từ 80%, Top-5 từ 95%.
- Không có ảnh test xuất hiện trong train/validation.
- API trả Top-5, confidence đã hiệu chỉnh và latency.
- Một ảnh hợp lệ được xử lý dưới 1 giây trên GTX 1660 Ti, không tính thời gian upload.
- Toàn bộ quy trình chạy lại được từ command/config mà không chỉnh code.

Ngoài phạm vi: dữ liệu ngoài Food-101, detection nhiều món, segmentation, nutrition, mobile app và cloud deployment.

## 2. Data workflow

Giữ nguyên dataset gốc, không di chuyển hoặc sửa ảnh:

- 101 lớp, 101.000 ảnh.
- Official train: 75.750 ảnh, 750 ảnh/lớp.
- Official test: 25.250 ảnh, 250 ảnh/lớp.

Tạo manifest thay vì sao chép ảnh:

- Dùng seed `42` tách official train theo từng lớp thành:
  - Train: 60.600 ảnh, 600 ảnh/lớp.
  - Validation: 15.150 ảnh, 150 ảnh/lớp.
- Giữ nguyên toàn bộ official test để đánh giá cuối cùng.
- Mỗi record gồm `relative_path`, `class_name`, `class_id`, `split`.
- Class ID lấy theo đúng thứ tự trong `meta/classes.txt`.

Data audit phải kiểm tra:

- File được tham chiếu có tồn tại và giải mã được bằng PIL.
- Ảnh được chuyển về RGB khi đọc; không chỉnh sửa file gốc.
- Không trùng đường dẫn giữa các split.
- Tính SHA-256 để phát hiện exact duplicate xuyên split.
- Thống kê kích thước, aspect ratio, mode ảnh và số mẫu mỗi lớp.
- Ảnh hỏng được ghi vào báo cáo và loại qua manifest, không bị xóa.
- Không làm sạch nhãn test thủ công để tránh làm thay đổi benchmark chuẩn.

Preprocessing:

- Train: `RandomResizedCrop(224)`, horizontal flip, color jitter nhẹ, random rotation tối đa 10°, normalize theo ImageNet.
- Validation/test/inference: resize cạnh ngắn 256, center crop 224, cùng normalization.
- Chỉ train split được augmentation; không tạo thêm ảnh vật lý trên ổ đĩa.

## 3. Model và training workflow

Dùng PyTorch/torchvision trong môi trường Python 3.11, CUDA và automatic mixed precision. Khóa dependency sau khi xác nhận bộ PyTorch/CUDA tương thích với GPU và driver hiện tại.

Protocol chung để so sánh công bằng:

- Models: ResNet18 baseline và EfficientNet-B0 candidate.
- Thay classifier cuối bằng output 101 lớp.
- Loss: cross-entropy với label smoothing `0.1`.
- Optimizer: AdamW, weight decay `1e-4`.
- Batch size `32`; dùng gradient accumulation nếu GPU không đủ VRAM.
- Giai đoạn 1: đóng backbone, train classifier 3 epoch với learning rate `1e-3`.
- Giai đoạn 2: mở toàn bộ model, learning rate backbone `1e-4`, classifier `3e-4`.
- Cosine annealing scheduler, tối đa 20 epoch fine-tuning.
- Early stopping sau 5 epoch không cải thiện validation Top-1.
- AMP, gradient clipping `1.0`, seed `42`, deterministic validation/test.
- Lưu checkpoint có validation Top-1 cao nhất; macro-F1 dùng để phá hòa.

Theo dõi mỗi run:

- Config, seed, git/environment metadata, epoch, learning rate.
- Train/validation loss, Top-1, Top-5, macro precision/recall/F1.
- GPU memory, thời gian mỗi epoch và tổng thời gian train.
- Best checkpoint và lịch sử metrics ở định dạng JSON/CSV.

Evaluation cuối:

- Chỉ chạy official test sau khi đã chọn model/hyperparameter bằng validation.
- Sinh classification report, confusion matrix chuẩn hóa và 20 cặp lớp dễ nhầm nhất.
- Lưu ví dụ false positive/false negative có confidence cao.
- Đo latency sau warm-up với batch size 1 trên GPU và CPU.
- Dùng validation logits để temperature-scale confidence; không dùng test set để calibration.
- Chọn EfficientNet-B0 nếu đạt mục tiêu accuracy và latency; fallback ResNet18 nếu candidate không tạo cải thiện đáng kể hoặc inference vượt giới hạn.

Artifact triển khai gồm:

- `model.pt`: checkpoint tốt nhất.
- `classes.json`: ánh xạ class ID–class name.
- `model_config.json`: architecture, input size và normalization.
- `metrics.json`: validation/test metrics, latency và model version.
- `temperature.json`: tham số confidence calibration.

## 4. FastAPI và Streamlit

FastAPI nạp model một lần khi startup và cung cấp:

- `GET /health`: trạng thái service, device và model version.
- `GET /classes`: danh sách 101 lớp.
- `POST /predict`: nhận một file multipart và trả:
  - `predicted_class`
  - `confidence`
  - `top_k` gồm tối đa 5 `{class_id, class_name, confidence}`
  - `low_confidence`
  - `latency_ms`
  - `model_version`

Quy tắc inference:

- Chỉ nhận JPEG/PNG tối đa 10 MB.
- Kiểm tra MIME và nội dung ảnh; chuyển grayscale/RGBA sang RGB.
- Dùng đúng preprocessing từ artifact, không khai báo transform riêng trong API.
- Confidence lấy từ calibrated softmax.
- Đặt `low_confidence=true` khi confidence Top-1 dưới `0.50`; giao diện phải nói rõ model luôn chọn trong 101 lớp và ảnh ngoài miền có thể không đáng tin.
- Trả HTTP 400/413/415 cho file lỗi, quá lớn hoặc sai định dạng; HTTP 503 nếu model chưa sẵn sàng.

Streamlit:

- Upload hoặc kéo-thả một ảnh.
- Hiển thị ảnh, dự đoán Top-1, confidence, Top-5 dạng bảng/bar chart và latency.
- Hiển thị cảnh báo rõ ràng với kết quả confidence thấp.
- Gọi FastAPI qua HTTP; không nạp model lần thứ hai.
- Có hướng dẫn chạy API và UI local bằng hai command riêng.

## 5. Lộ trình và kiểm thử

### Tuần 1 — Data foundation

- Thiết lập cấu trúc project, dependency và config.
- Sinh manifests cố định, data audit và báo cáo EDA.
- Hoàn thiện Dataset/DataLoader và trực quan hóa augmentation.
- Gate: đủ 101 lớp, đúng số lượng split, không overlap và loader đọc được toàn bộ manifest.

### Tuần 2 — Baseline và candidate

- Chạy smoke training trên tập nhỏ.
- Train đầy đủ ResNet18, sau đó EfficientNet-B0 với cùng protocol.
- Theo dõi VRAM, thời gian và metrics; chỉ điều chỉnh batch/accumulation khi OOM.
- Gate: cả hai run tái lập được, có best checkpoint và validation report.

### Tuần 3 — Evaluation và đóng gói model

- Chọn model bằng validation, chạy test đúng một lần cho cấu hình cuối.
- Calibration, confusion matrix, error analysis và latency benchmark.
- Đóng gói artifact, viết inference module dùng chung.
- Gate: artifact nạp độc lập và cho kết quả nhất quán với evaluation pipeline.

### Tuần 4 — Web demo và nghiệm thu

- Xây FastAPI, Streamlit, validation đầu vào và xử lý lỗi.
- Viết tài liệu setup/train/evaluate/serve và báo cáo kết quả.
- Chạy end-to-end test từ upload đến Top-5 response.
- Gate: demo local hoạt động, API contract đúng, không phụ thuộc notebook hay đường dẫn tuyệt đối.

Test bắt buộc:

- Unit test class mapping, manifest split, transforms và output shape `[batch, 101]`.
- Data leakage test cho path và SHA-256 giữa train/validation/test.
- Smoke train 1 epoch trên subset và resume checkpoint.
- Test cùng một ảnh qua evaluator và API phải có Top-1 giống nhau.
- API tests cho JPEG, PNG, grayscale, RGBA, file hỏng, MIME sai và file quá lớn.
- Kiểm tra tổng confidence Top-5 hợp lệ và thứ tự giảm dần.
- End-to-end test Streamlit → FastAPI → model.
- Reproducibility test: cùng checkpoint và input phải trả cùng kết quả trong sai số số thực cho phép.

## Giả định

- Chỉ ảnh Food-101 trong folder dự án được dùng để train, validation và test.
- Pretrained ImageNet weights được phép dùng làm khởi tạo; không đưa ảnh ImageNet hoặc nguồn ngoài vào pipeline.
- Model và web demo chạy local trên GTX 1660 Ti 6 GB; chưa tối ưu cho production cloud.
- Tên lớp hiển thị được đổi dấu gạch dưới thành khoảng trắng và viết hoa để dễ đọc, nhưng class ID gốc không thay đổi.
- Dataset gốc luôn ở chế độ read-only; mọi manifest, báo cáo, checkpoint và artifact nằm ngoài `food-101`.
