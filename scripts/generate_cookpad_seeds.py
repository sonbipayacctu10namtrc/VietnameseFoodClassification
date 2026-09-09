"""Generate accent-aware web-search seeds for Vietnamese food classes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import quote, quote_plus

from food_classifier.vietnamese_dishes import DISHES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priorities", default="P0,P1")
    parser.add_argument("--provider", choices=("cookpad", "bing"), default="cookpad")
    parser.add_argument("--query-style", choices=("generic", "targeted"), default="generic")
    parser.add_argument("--metadata", type=Path, help="Only emit classes below --target in this JSONL file")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("seeds_cookpad_p0_p1.txt"))
    args = parser.parse_args()

    priorities = {value.strip().upper() for value in args.priorities.split(",")}
    counts: Counter[str] = Counter()
    if args.metadata:
        counts.update(
            json.loads(line)["canonical_label"]
            for line in args.metadata.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    lines = [f"# Generated {args.provider} searches: five query variants per class."]
    for dish in DISHES:
        if dish.priority not in priorities:
            continue
        if args.metadata and counts[dish.canonical_label] >= args.target:
            continue
        region_query = "Việt Nam" if dish.region == "nationwide" else dish.region
        if args.query_style == "targeted":
            queries = (
                f'"{dish.vietnamese_name}"',
                f'"{dish.vietnamese_name}" đặc sản {region_query}',
                f'hình ảnh "{dish.vietnamese_name}"',
                dish.english_queries[0],
                f'"{dish.vietnamese_name}" công thức món Việt',
            )
        else:
            queries = (
                dish.vietnamese_name,
                f"{dish.vietnamese_name} cách làm",
                f"{dish.vietnamese_name} ngon",
                f"{dish.vietnamese_name} truyền thống",
                f"{dish.vietnamese_name} {region_query}",
            )
        for query in dict.fromkeys(queries):
            if args.provider == "cookpad":
                url = f"https://cookpad.com/vn/tim-kiem/{quote(query, safe='')}"
            else:
                suffix = " món ăn Việt Nam" if args.query_style == "generic" else ""
                url = f"https://www.bing.com/images/search?q={quote_plus(query + suffix)}"
            lines.append(f"{dish.canonical_label}\t{url}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines) - 1} seeds to {args.output}")


if __name__ == "__main__":
    main()
