# Hybrid Vietnamese-food crawler

The crawler writes all providers into one `metadata.jsonl` file and saves accepted images under `images/`. It never changes `food-101/`.

## Providers

- `commons`: queries the Wikimedia Commons MediaWiki API directly. No API key is required; license and attribution metadata are saved.
- `web`: uses Crawl4AI for the user-provided `label<TAB>URL` lines in a seed file. Use actual recipe, restaurant, or editorial pages; do not use Wikimedia search result pages as web seeds.
- `pexels`: requires `PEXELS_API_KEY` and saves the Pexels source page and photographer.
- `flickr`: requires `FLICKR_API_KEY`; requests the Creative Commons-compatible license IDs and saves the Flickr photo page.

## Commands

Start with Commons only, for P0 classes:

```powershell
$env:PYTHONPATH = "src"
python scripts/crawl_vietnamese_food.py --providers commons --priorities P0 --per-label 25
```

Add Crawl4AI pages from `seeds.txt`:

```powershell
python scripts/crawl_vietnamese_food.py --providers web,commons --seed-file seeds.txt --priorities P0
```

Add an API provider after setting its key in the current PowerShell session:

```powershell
$env:PEXELS_API_KEY = "your-key"
python scripts/crawl_vietnamese_food.py --providers commons,pexels --priorities P0 --per-label 25
```

Inspect `rejection_reasons` in the final JSON before expanding the crawl. Review labels and licenses before including any crawled image in training.
