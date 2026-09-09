"""Unified local web app: predict a dish from a photo AND browse training runs."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for training_dashboard import

import training_dashboard as dash  # reuse find_runs / show_run
from food_classifier.inference import FoodPredictor

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def runs_with_checkpoint() -> list[Path]:
    return [run for run in dash.find_runs() if (run / "best.pt").exists()]


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_predictor(checkpoint: str) -> FoodPredictor:
    return FoodPredictor(Path(checkpoint))


def predict_tab() -> None:
    st.subheader("🍜 Nhận diện món ăn")
    runs = runs_with_checkpoint()
    if not runs:
        st.warning("Chưa có checkpoint (best.pt) nào trong artifacts/. Hãy huấn luyện một model trước.")
        return
    default = next((i for i, r in enumerate(runs) if r.name == "vietnamese_resnet18"), 0)
    run_name = st.selectbox("Model", [r.name for r in runs], index=default)
    predictor = load_predictor(str(ARTIFACTS_ROOT / run_name / "best.pt"))
    st.caption(f"{len(predictor.classes)} lớp · thiết bị {predictor.device}")

    upload = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.info("Tải lên một ảnh để nhận top-5 dự đoán.")
        return
    image = Image.open(upload).convert("RGB")
    left, right = st.columns([1, 1.3])
    left.image(image, caption=upload.name, use_container_width=True)
    predictions = predictor.predict(image, top_k=5)
    with right:
        st.markdown("#### Kết quả")
        for i, p in enumerate(predictions):
            label = f"**{p['name']}**  ·  `{p['label']}`" + ("  🏆" if i == 0 else "")
            st.write(f"{label} — {p['probability'] * 100:.1f}%")
            st.progress(min(1.0, p["probability"]))
        if predictions[0]["probability"] < 0.4:
            st.warning("Độ tin cậy thấp — ảnh có thể ngoài 17 lớp đã học, hoặc là món dễ nhầm.")


def training_tab() -> None:
    st.subheader("📊 Theo dõi huấn luyện")
    st.button("Refresh")
    runs = dash.find_runs()
    if not runs:
        st.warning("Chưa tìm thấy run nào trong artifacts/.")
        return
    selected = st.selectbox("Training run", [r.name for r in runs])
    dash.show_run(next(r for r in runs if r.name == selected))


def main() -> None:
    st.set_page_config(page_title="Món ăn Việt Nam — Dự đoán & Huấn luyện", page_icon="🍜", layout="wide")
    st.title("🍜 Nhận diện & Theo dõi huấn luyện món ăn Việt Nam")
    predict, training = st.tabs(["🍜 Nhận diện món ăn", "📊 Theo dõi huấn luyện"])
    with predict:
        predict_tab()
    with training:
        training_tab()


if __name__ == "__main__":
    main()
