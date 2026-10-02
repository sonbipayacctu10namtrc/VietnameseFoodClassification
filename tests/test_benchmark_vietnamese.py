import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from benchmark_vietnamese import label_baseline, metrics


def test_metrics_calculates_macro_f1() -> None:
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2, top5=1.0)
    assert result["top1_accuracy"] == 0.75
    assert result["macro_f1"] == 0.7333333333333334


def test_majority_baseline_uses_train_distribution() -> None:
    train = [{"class_id": "0"}, {"class_id": "1"}, {"class_id": "1"}]
    evaluation = [{"class_id": "0"}, {"class_id": "1"}]
    result = label_baseline(train, evaluation, 2)
    assert result["predicted_class_id"] == 1
    assert result["top1_accuracy"] == 0.5
