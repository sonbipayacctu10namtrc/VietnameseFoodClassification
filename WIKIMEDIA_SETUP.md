# Wikimedia Commons setup

Wikimedia requires a descriptive User-Agent with a way to contact the crawler operator. Before enabling the `commons` provider, set `WIKIMEDIA_CONTACT` to your email address or a public project URL:

```powershell
$env:WIKIMEDIA_CONTACT = "your-email@example.com"
$env:PYTHONPATH = "src"
python scripts/crawl_vietnamese_food.py --providers commons --priorities P0 --per-label 25
```

Do not use a generic browser User-Agent. The crawler identifies itself as `FoodClassificationCrawler/0.2` and includes this configured contact value for Commons API requests.
