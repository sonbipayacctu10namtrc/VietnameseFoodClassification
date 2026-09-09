# Vietnamese food image crawler

The catalog is defined in `src/food_classifier/vietnamese_dishes.py`. It contains 79 unique candidates in priorities P0-P3 and an explicit review/exclusion list for Food-101 overlaps. The original `food-101/` directory is never modified.

Install the optional crawler dependencies:

```powershell
pip install -r requirements-crawler.txt
playwright install chromium
```

Create a UTF-8 seed file with one page per line:

```text
banh_mi<TAB>https://example.com/vietnamese-banh-mi
bun_bo_hue<TAB>https://example.com/bun-bo-hue
```

Run catalog export only:

```powershell
$env:PYTHONPATH = "src"
python scripts/crawl_vietnamese_food.py --export-catalog
```

Run crawling and image validation:

```powershell
$env:PYTHONPATH = "src"
python scripts/crawl_vietnamese_food.py --seed-file seeds.txt --output-dir artifacts/vietnamese_food
```

The output contains `images/` and append-only `metadata.jsonl`. Each accepted record includes the canonical label, page URL, image URL, dimensions, format, SHA-256, perceptual hash, and any image license metadata exposed by the page. The crawler only processes user-provided seed pages; it does not bypass search-engine protections or site access controls.
