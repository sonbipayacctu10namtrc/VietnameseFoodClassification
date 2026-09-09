from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from food_classifier.vietnamese_dishes import DISHES, EXCLUDED_OR_REVIEW
from food_classifier.crawler import _bing_original_images, _normalize_image_url


def test_catalog_has_unique_labels_and_expected_priorities():
    labels = [dish.canonical_label for dish in DISHES]
    assert len(labels) == len(set(labels))
    assert len(DISHES) == 80
    assert {dish.priority for dish in DISHES} == {"P0", "P1", "P2", "P3"}


def test_catalog_has_bilingual_queries_and_excludes_food101_overlap():
    assert all(len(dish.vietnamese_queries) >= 2 for dish in DISHES)
    assert all(dish.english_queries for dish in DISHES)
    assert "pho" not in {dish.canonical_label for dish in DISHES}
    assert any(label == "pho" and food101_class == "pho" for label, _, food101_class in EXCLUDED_OR_REVIEW)


def test_cookpad_image_variants_normalize_to_one_jpeg_url():
    base = "https://img-global.cpcdn.com/recipes/abc123/600x440cq80/photo"
    assert _normalize_image_url(base + ".webp") == _normalize_image_url(base + ".jpg")
    assert _normalize_image_url(base + ".webp").endswith("/680x482cq70/photo.jpg")


def test_bing_parser_extracts_original_image_and_page_urls():
    payload = '{"murl":"https://img.test/food.jpg","purl":"https://page.test/recipe","t":"Food"}'
    html = f'<a class="iusc result" m="{payload.replace(chr(34), "&quot;")}"></a>'
    assert _bing_original_images(html) == [
        ("https://img.test/food.jpg", "https://page.test/recipe", "Food")
    ]
