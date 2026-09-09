from fastapi.testclient import TestClient

from food_classifier.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runs_exposes_completed_ten_class_run() -> None:
    response = client.get("/api/runs/resnet18_10class")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["state"] == "completed"
    assert len(payload["history"]) == 5
    assert payload["evaluation"]["examples"] == 2500
