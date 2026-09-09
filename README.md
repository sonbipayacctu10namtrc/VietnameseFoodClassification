# Food-101 Classification

Pipeline huấn luyện và web demo local để phân loại một món ăn trong ảnh thành một trong 101 lớp của Food-101.

## Trạng thái

Đang xây dựng theo từng bước. Dataset gốc nằm tại `food-101/` và luôn được giữ nguyên.

Kế hoạch chi tiết: [PLAN.md](PLAN.md).

## Cấu trúc dự án

```text
food-101/                  # Dataset gốc, không chỉnh sửa
src/food_classifier/       # Mã nguồn Python
configs/                   # Cấu hình train/evaluate/serve
scripts/                   # Entry point cho các workflow
tests/                     # Automated tests
artifacts/                 # Manifest, checkpoint, model export (git-ignored)
reports/                   # EDA và evaluation output (git-ignored)
```

## Yêu cầu môi trường

- Python 3.11 hoặc 3.12
- NVIDIA GPU (khuyến nghị) để training
- Không dùng ảnh ngoài Food-101 cho train, validation hoặc test

Pretrained ImageNet weights chỉ được dùng làm khởi tạo model, theo kế hoạch đã chốt.

## Baseline ResNet18

Kích hoạt môi trường `vision`, sau đó chạy smoke test GPU trước:

```powershell
conda activate vision
python scripts/train.py --smoke-test --epochs 1 --freeze-epochs 0
```

Huấn luyện baseline đầy đủ (12 epoch, 600 train và 150 validation ảnh cho mỗi lớp):

```powershell
python scripts/train.py
```

Huấn luyện EfficientNet-B0 với cosine learning-rate decay:

```powershell
python scripts/train.py --model efficientnet_b0 --epochs 15 --freeze-epochs 2 --output-dir artifacts/efficientnet_b0_full101
```

Checkpoint, manifests và lịch sử metrics được lưu tại `artifacts/resnet18_baseline/`.

Đánh giá checkpoint trên official test split:

```powershell
python scripts/evaluate.py --checkpoint artifacts/resnet18_10class/best.pt --output-dir artifacts/resnet18_10class/evaluation
```

## Training dashboard

Mở dashboard local để xem loss, accuracy, checkpoint và kết quả test:

```powershell
streamlit run apps/training_dashboard.py
```

Trang mặc định tại `http://localhost:8501`. Metrics của các lần train mới được ghi sau mỗi epoch.

## React training dashboard

Chạy backend ở terminal thứ nhất:

```powershell
conda activate vision
uvicorn food_classifier.api:app --reload --port 8000
```

Chạy React frontend ở terminal thứ hai:

```powershell
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173` để xem dashboard.
