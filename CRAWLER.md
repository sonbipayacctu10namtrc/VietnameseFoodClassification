# Vietnamese food image crawler

The class catalog is defined in `src/food_classifier/vietnamese_dishes.py`: 80 unique dishes in priorities P0–P3, plus an explicit review/exclusion list for dishes that overlap Food-101 classes. The crawler writes every provider into one `metadata.jsonl` file and saves accepted images under `images/`. The original `food-101/` directory is never modified.

## Setup

Install the optional crawler dependencies:

```powershell
pip install -r requirements-crawler.txt
playwright install chromium
```

API keys and the Wikimedia contact are read from environment variables. Copy `.env.crawler.example` to `.env.crawler` (gitignored, never commit real keys), fill it in, then load it into the current shell:

```powershell
# PowerShell
Get-Content .env.crawler | Where-Object { $_ -match '^\w+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
```

```bash
# bash
set -a; source .env.crawler; set +a
```

| Variable | Needed for |
| --- | --- |
| `WIKIMEDIA_CONTACT` | provider `commons` (required) |
| `PEXELS_API_KEY` | provider `pexels` |
| `FLICKR_API_KEY` | provider `flickr` |

### Wikimedia Commons contact

Wikimedia requires a descriptive User-Agent with a way to contact the crawler operator. Set `WIKIMEDIA_CONTACT` to your email address or a public project URL; the `commons` provider refuses to run without it. Do not use a generic browser User-Agent: the crawler identifies itself as `FoodClassificationCrawler/0.2` and includes the configured contact in Commons API requests.

## Providers

- `commons`: queries the Wikimedia Commons MediaWiki API directly. No API key is required; license and attribution metadata are saved.
- `web`: uses Crawl4AI for the `label<TAB>URL` lines of a seed file. Use actual recipe, restaurant or editorial pages; do not use Wikimedia search result pages as web seeds.
- `pexels`: requires `PEXELS_API_KEY`; saves the Pexels source page and photographer.
- `flickr`: requires `FLICKR_API_KEY`; requests only Creative Commons-compatible license IDs and saves the Flickr photo page.

## Seed files

A seed file is UTF-8 text with one page per line, `canonical_label<TAB>URL`; lines starting with `#` are comments:

```text
banh_mi	https://example.com/vietnamese-banh-mi
bun_bo_hue	https://example.com/bun-bo-hue
```

| File | Content |
| --- | --- |
| `seeds.txt` | Hand-picked starter pages for the 30 P0 classes |
| `seeds_cookpad_50classes.txt` | Cookpad searches, five query variants per class |
| `seeds_bing_50classes.txt` | Bing image searches, five query variants per class |

Generated seed files come from `scripts/generate_cookpad_seeds.py` (`--provider cookpad|bing`, `--priorities`, and `--metadata`/`--target` to emit only classes that still have fewer images than the target).

## Commands

```powershell
$env:PYTHONPATH = "src"

# Export the catalog only
python scripts/crawl_vietnamese_food.py --export-catalog

# Start with Commons only, for P0 classes
python scripts/crawl_vietnamese_food.py --providers commons --priorities P0 --per-label 25

# Add Crawl4AI pages from a seed file
python scripts/crawl_vietnamese_food.py --providers web,commons --seed-file seeds.txt --priorities P0

# Add an API provider (key loaded as above)
python scripts/crawl_vietnamese_food.py --providers commons,pexels --priorities P0 --per-label 25
```

On Linux/macOS replace the first line with `export PYTHONPATH=src`. Useful options: `--output-dir` (default `artifacts/vietnamese_food`), `--target-per-class`, `--min-width`/`--min-height` (default 256), `--max-mb` (default 10), `--request-delay` (default 1.5 s) and `--download-concurrency`.

## Output

The output directory contains `images/<label>/` and an append-only `metadata.jsonl`. Each accepted record includes the canonical label, page URL, image URL, dimensions, format, SHA-256, perceptual hash and any license/attribution metadata exposed by the source. Images whose SHA-256 or perceptual hash is already present are skipped.

Inspect `rejection_reasons` in the final JSON summary before expanding the crawl, and review labels and licenses before using any crawled image for training.

## Rules

The crawler only processes the seed pages you provide and the official provider APIs. It does not bypass search-engine protections or site access controls.
