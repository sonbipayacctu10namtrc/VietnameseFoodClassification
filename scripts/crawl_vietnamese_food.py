"""Acquire Vietnamese-food image candidates from web pages and image-library APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from food_classifier.crawler import run_hybrid
from food_classifier.vietnamese_dishes import export_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, help="Optional label<TAB>URL web seed file for Crawl4AI.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vietnamese_food"))
    parser.add_argument("--providers", default="web,commons", help="Comma-separated: web,commons,pexels,flickr")
    parser.add_argument("--priorities", default="P0", help="Comma-separated catalog priorities, e.g. P0,P1")
    parser.add_argument("--per-label", type=int, default=25, help="Maximum API candidates per class and provider")
    parser.add_argument("--target-per-class", type=int, help="Stop saving a class when this total is reached")
    parser.add_argument("--export-catalog", action="store_true", help="Write the complete catalog JSON")
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--max-mb", type=float, default=10)
    parser.add_argument("--request-delay", type=float, default=1.5, help="Seconds between image downloads")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries for HTTP 429/5xx responses")
    parser.add_argument("--http-timeout", type=float, default=30, help="Per-image HTTP timeout in seconds")
    parser.add_argument("--download-concurrency", type=int, default=1, help="Concurrent image downloads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_catalog:
        export_catalog(args.output_dir / "vietnamese_dishes.json")
        print(f"Catalog written to {args.output_dir / 'vietnamese_dishes.json'}")
    if args.export_catalog and not args.seed_file and args.providers == "web,commons":
        return
    stats = asyncio.run(run_hybrid(
        args.output_dir,
        providers=args.providers.split(","),
        priorities=args.priorities.split(","),
        seed_file=args.seed_file,
        per_label=args.per_label,
        target_per_class=args.target_per_class,
        min_width=args.min_width,
        min_height=args.min_height,
        max_bytes=int(args.max_mb * 1024 * 1024),
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        http_timeout=args.http_timeout,
        download_concurrency=args.download_concurrency,
    ))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
