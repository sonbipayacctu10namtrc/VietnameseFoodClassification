"""FastAPI endpoints that expose local Food-101 training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

app = FastAPI(title="Food-101 Training API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def get_run_dir(run_name: str) -> Path:
    run_dir = ARTIFACTS_ROOT / run_name
    if Path(run_name).name != run_name or not run_dir.is_dir() or not (run_dir / "manifests").exists():
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_name}")
    return run_dir


def serialize_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json", {})
    status = read_json(run_dir / "status.json", {})
    if not status:
        status = {
            "state": "completed" if summary else "unknown",
            "current_epoch": summary.get("epochs", 0),
            "total_epochs": summary.get("epochs", 0),
            "best_validation_top1": summary.get("best_validation_top1", 0),
        }
    evaluation_dir = run_dir / "evaluation"
    evaluation = read_json(evaluation_dir / "metrics.json", None)
    return {
        "name": run_dir.name,
        "status": status,
        "summary": summary,
        "history": read_json(run_dir / "history.json", []),
        "evaluation": evaluation,
        "has_checkpoint": (run_dir / "best.pt").exists(),
        "has_confusion_matrix": (evaluation_dir / "confusion_matrix.png").exists(),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    if not ARTIFACTS_ROOT.exists():
        return []
    runs = [serialize_run(path) for path in ARTIFACTS_ROOT.iterdir() if path.is_dir() and (path / "manifests").exists()]
    return sorted(runs, key=lambda run: run["name"], reverse=True)


@app.get("/api/runs/{run_name}")
def get_run(run_name: str) -> dict[str, Any]:
    return serialize_run(get_run_dir(run_name))


@app.get("/api/runs/{run_name}/confusion-matrix")
def confusion_matrix_image(run_name: str) -> FileResponse:
    matrix_path = get_run_dir(run_name) / "evaluation" / "confusion_matrix.png"
    if not matrix_path.exists():
        raise HTTPException(status_code=404, detail="No confusion matrix exists for this run.")
    return FileResponse(matrix_path, media_type="image/png")
