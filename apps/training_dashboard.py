"""Local Streamlit dashboard for Food-101 experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_runs() -> list[Path]:
    return sorted(
        (path for path in ARTIFACTS_ROOT.iterdir() if path.is_dir() and (path / "manifests").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def show_run(run_dir: Path) -> None:
    history = read_json(run_dir / "history.json") or []
    summary = read_json(run_dir / "summary.json") or {}
    status = read_json(run_dir / "status.json") or {
        "state": "completed" if summary else "unknown",
        "current_epoch": summary.get("epochs", 0),
        "total_epochs": summary.get("epochs", "?"),
        "best_validation_top1": summary.get("best_validation_top1", 0),
    }
    state = str(status["state"]).upper()
    st.subheader(run_dir.name)
    st.caption(f"Artifacts: {run_dir}")

    first, second, third, fourth = st.columns(4)
    first.metric("Status", state)
    second.metric("Epoch", f"{status.get('current_epoch', 0)} / {status.get('total_epochs', '?')}")
    third.metric("Best validation Top-1", f"{status.get('best_validation_top1', summary.get('best_validation_top1', 0)):.2%}")
    fourth.metric("Elapsed", f"{summary.get('elapsed_minutes', 0):.1f} min" if summary else "Running")

    if history:
        frame = pd.DataFrame(history).set_index("epoch")
        st.markdown("#### Learning curves")
        left, right = st.columns(2)
        left.line_chart(frame[["train_loss", "validation_loss"]], y_label="Loss")
        right.line_chart(frame[["train_top1", "validation_top1"]], y_label="Top-1 accuracy")
        st.dataframe(frame, use_container_width=True)
    else:
        st.info("Run has started. Metrics will appear after the first epoch finishes.")

    evaluation_dir = run_dir / "evaluation"
    metrics = read_json(evaluation_dir / "metrics.json")
    if metrics:
        st.markdown("#### Official test evaluation")
        test_one, test_two, test_three = st.columns(3)
        test_one.metric("Top-1", f"{metrics['top1_accuracy']:.2%}")
        test_two.metric("Top-5", f"{metrics['top5_accuracy']:.2%}")
        test_three.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
        matrix_path = evaluation_dir / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption="Official test confusion matrix")

    checkpoint_path = run_dir / "best.pt"
    if checkpoint_path.exists():
        st.success(f"Best checkpoint: {checkpoint_path.name}")


def main() -> None:
    st.set_page_config(page_title="Food-101 Training", page_icon="🍜", layout="wide")
    st.title("🍜 Food-101 Training Dashboard")
    st.caption("Theo dõi metrics được ghi từ các run huấn luyện local.")
    st.button("Refresh", type="primary")

    runs = find_runs()
    if not runs:
        st.warning("Chưa tìm thấy run nào trong artifacts/.")
        return
    selected_name = st.selectbox("Training run", [path.name for path in runs])
    show_run(next(path for path in runs if path.name == selected_name))


if __name__ == "__main__":
    main()
