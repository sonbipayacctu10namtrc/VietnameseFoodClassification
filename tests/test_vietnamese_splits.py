import json

from food_classifier.vietnamese_splits import create_manifests


def test_create_manifests_keeps_same_source_in_one_split(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for label_index, label in enumerate(("banh_mi", "bun_cha")):
        for index in range(10):
            path = image_dir / f"{label}_{index}.jpg"
            path.write_bytes(b"image")
            rows.append({
                "image_path": f"images/{path.name}", "canonical_label": label,
                "source_url": f"https://example.test/{label}/{index // 2}",
                "image_url": f"https://img.test/{label}/{index}.jpg",
                "sha256": f"{label_index:02x}{index:062x}",
                "perceptual_hash": f"{label_index * 1000 + index * 100:064x}",
                "provider": "test", "license": "CC0",
            })
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "manifests"
    summary = create_manifests(metadata, output, min_per_class=10, near_duplicate_distance=0)
    assert summary["classes"] == 2
    assert summary["images"] == 20
    source_splits = {}
    for split in ("train", "validation", "test"):
        text = (output / f"{split}.csv").read_text(encoding="utf-8")
        for row in rows:
            if row["source_url"] in text:
                source_splits.setdefault(row["source_url"], set()).add(split)
    assert all(len(splits) == 1 for splits in source_splits.values())
