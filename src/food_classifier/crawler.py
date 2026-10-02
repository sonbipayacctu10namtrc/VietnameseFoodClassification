"""Hybrid image acquisition for Vietnamese-food dataset candidates."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from food_classifier.vietnamese_dishes import DISHES, Dish

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
PEXELS_API = "https://api.pexels.com/v1/search"
FLICKR_API = "https://www.flickr.com/services/rest/"


@dataclass(frozen=True)
class Seed:
    label: str
    url: str


@dataclass(frozen=True)
class SourceImage:
    label: str
    image_url: str
    source_url: str
    provider: str
    alt: str = ""
    license: str | None = None
    attribution: str | None = None


class _BingImageParser(HTMLParser):
    """Extract original image/page URLs embedded in Bing image result cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        payload = values.get("m")
        if "iusc" not in classes or not payload:
            return
        try:
            item = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return
        image_url = item.get("murl")
        if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
            source_url = item.get("purl") if isinstance(item.get("purl"), str) else ""
            alt = item.get("t") if isinstance(item.get("t"), str) else ""
            self.results.append((image_url, source_url, alt))


def _bing_original_images(page_html: str) -> list[tuple[str, str, str]]:
    parser = _BingImageParser()
    parser.feed(page_html)
    return parser.results


def read_seeds(path: Path) -> list[Seed]:
    seeds: list[Seed] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].startswith(("http://", "https://")):
            raise ValueError(f"Invalid seed at {path}:{number}; expected label<TAB>https-url")
        seeds.append(Seed(*fields))
    return seeds


def _normalize_image_url(image_url: str) -> str:
    image_url = image_url.replace("https://thumb.wikimedia.org/", "https://upload.wikimedia.org/")
    if "img-global.cpcdn.com/recipes/" in image_url:
        image_url = re.sub(r"(/recipes/[^/]+/)[^/]+/", r"\g<1>680x482cq70/", image_url)
        # Cookpad exposes the same recipe photo as both WebP and JPEG. Request
        # one canonical representation so URL-level dedup catches the pair.
        image_url = re.sub(r"\.(?:webp|jpe?g)(?=$|\?)", ".jpg", image_url, flags=re.IGNORECASE)
    return image_url


def _perceptual_hash(data: bytes) -> str:
    image = Image.open(io.BytesIO(data)).convert("L").resize((16, 16))
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    return f"{int(''.join('1' if pixel >= mean else '0' for pixel in pixels), 2):064x}"


def _extension(content_type: str, image_url: str) -> str:
    suffix = Path(image_url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else mimetypes.guess_extension(content_type.split(";", 1)[0]) or ".jpg"


def _validate_image(data: bytes, min_width: int, min_height: int) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height, image_format = image.width, image.height, image.format or "unknown"
            image.verify()
        if width < min_width or height < min_height:
            raise ValueError(f"image too small: {width}x{height}")
        return width, height, image_format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image") from exc


def _jpeg_bytes(data: bytes) -> bytes:
    """Return canonical RGB JPEG bytes for consistent dataset storage."""
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


def _dishes(priorities: Iterable[str]) -> list[Dish]:
    wanted = {priority.upper() for priority in priorities}
    selected = [dish for dish in DISHES if dish.priority in wanted]
    if not selected:
        raise ValueError("No catalog classes match --priorities")
    return selected


async def _get_with_retry(
    client: Any,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    max_retries: int = 4,
) -> Any:
    """Retry transient CDN failures while honoring Retry-After when provided."""
    import httpx

    response = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, headers=headers, params=params)
        except httpx.RequestError:
            # Transient network/timeout error (no HTTP response): back off and retry.
            if attempt == max_retries:
                raise
            await asyncio.sleep(min(5 * (2**attempt), 40))
            continue
        if response.status_code not in {429, 502, 503, 504}:
            return response
        if attempt == max_retries:
            break
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(5 * (2**attempt), 40)
        except ValueError:
            delay = min(5 * (2**attempt), 40)
        await asyncio.sleep(min(max(delay, 1), 60))
    return response


async def _web_candidates(seeds: list[Seed], stats: Counter[str]) -> list[SourceImage]:
    if not seeds:
        return []
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError as exc:
        raise RuntimeError("Install Crawl4AI: pip install -r requirements-crawler.txt") from exc
    candidates: list[SourceImage] = []
    seen_urls_by_label: dict[str, set[str]] = {}
    async with AsyncWebCrawler() as crawler:
        for seed in seeds:
            seen_urls = seen_urls_by_label.setdefault(seed.label, set())
            result = await crawler.arun(seed.url)
            stats["web_pages"] += 1
            if not result.success:
                stats["web_failed_pages"] += 1
                continue
            if "bing.com/images/search" in seed.url:
                for image_url, source_url, alt in _bing_original_images(getattr(result, "html", "") or ""):
                    image_url = _normalize_image_url(image_url)
                    if image_url in seen_urls:
                        stats["web_variant_duplicates"] += 1
                        continue
                    seen_urls.add(image_url)
                    candidates.append(SourceImage(seed.label, image_url, source_url or seed.url, "web", alt))
                stats["bing_original_candidates"] += 1
            media = getattr(result, "media", {}) or {}
            items = media.get("images", []) if isinstance(media, dict) else []
            for item in sorted(items, key=lambda value: value.get("score", 0), reverse=True):
                url = item.get("src") if isinstance(item, dict) else None
                if url and "img-global.cpcdn.com" in url:
                    if "/recipes/" not in url or "recipe-main-photo" not in url:
                        stats["web_non_recipe_skips"] += 1
                        continue
                    url = _normalize_image_url(url)
                if url in seen_urls:
                    stats["web_variant_duplicates"] += 1
                    continue
                if url and url.startswith(("http://", "https://")):
                    seen_urls.add(url)
                    candidates.append(SourceImage(seed.label, url, seed.url, "web", item.get("alt", ""), item.get("license")))
    return candidates


async def _commons_candidates(client: Any, dishes: list[Dish], per_label: int, stats: Counter[str], request_delay: float, max_retries: int) -> list[SourceImage]:
    candidates: list[SourceImage] = []
    for dish in dishes:
        label_candidates: list[SourceImage] = []
        seen_urls: set[str] = set()
        for query in dish.queries:
            offset: int | None = None
            while len(label_candidates) < per_label:
                params: dict[str, Any] = {
                    "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                    "gsrsearch": query, "gsrlimit": min(50, per_label - len(label_candidates)),
                    "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1024, "maxlag": 5,
                }
                if offset is not None:
                    params["gsroffset"] = offset
                response = await _get_with_retry(client, COMMONS_API, params=params, max_retries=max_retries)
                response.raise_for_status()
                payload = response.json()
                stats["commons_queries"] += 1
                for page in payload.get("query", {}).get("pages", {}).values():
                    info = (page.get("imageinfo") or [{}])[0]
                    url = info.get("thumburl") or info.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    metadata = info.get("extmetadata") or {}
                    label_candidates.append(SourceImage(
                        dish.canonical_label, url,
                        info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{page['title']}",
                        "wikimedia_commons", page.get("title", ""),
                        (metadata.get("LicenseShortName") or {}).get("value"),
                        (metadata.get("Artist") or {}).get("value"),
                    ))
                offset = payload.get("continue", {}).get("gsroffset")
                if offset is None:
                    break
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            if len(label_candidates) >= per_label:
                break
        candidates.extend(label_candidates[:per_label])
    return candidates


async def _pexels_candidates(client: Any, dishes: list[Dish], per_label: int, api_key: str, stats: Counter[str], request_delay: float, max_retries: int) -> list[SourceImage]:
    candidates: list[SourceImage] = []
    for dish in dishes:
        label_candidates: list[SourceImage] = []
        seen_ids: set[int] = set()
        for query in dish.english_queries:
            page = 1
            while len(label_candidates) < per_label:
                response = await _get_with_retry(client, PEXELS_API, params={"query": query, "per_page": min(80, per_label - len(label_candidates)), "page": page, "locale": "vi-VN"}, headers={"Authorization": api_key}, max_retries=max_retries)
                response.raise_for_status()
                stats["pexels_queries"] += 1
                photos = response.json().get("photos", [])
                if not photos:
                    break
                for photo in photos:
                    if photo["id"] in seen_ids:
                        continue
                    seen_ids.add(photo["id"])
                    label_candidates.append(SourceImage(dish.canonical_label, photo["src"].get("large2x") or photo["src"]["large"], photo["url"], "pexels", photo.get("alt", ""), "Pexels License", photo.get("photographer")))
                page += 1
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            if len(label_candidates) >= per_label:
                break
        candidates.extend(label_candidates[:per_label])
    return candidates


async def _flickr_candidates(client: Any, dishes: list[Dish], per_label: int, api_key: str, stats: Counter[str], request_delay: float, max_retries: int) -> list[SourceImage]:
    candidates: list[SourceImage] = []
    for dish in dishes:
        label_candidates: list[SourceImage] = []
        seen_ids: set[str] = set()
        for query in dish.english_queries:
            page = 1
            while len(label_candidates) < per_label:
                response = await _get_with_retry(client, FLICKR_API, params={
                    "method": "flickr.photos.search", "api_key": api_key, "text": query,
                    "license": "1,2,3,4,5,6,9,10", "content_type": 1, "media": "photos",
                    "extras": "license,owner_name,url_l,url_o", "per_page": min(100, per_label - len(label_candidates)),
                    "page": page, "format": "json", "nojsoncallback": 1,
                }, max_retries=max_retries)
                response.raise_for_status()
                payload = response.json()
                if payload.get("stat") != "ok":
                    raise RuntimeError(payload.get("message", "Flickr request failed"))
                stats["flickr_queries"] += 1
                photos = payload.get("photos", {}).get("photo", [])
                if not photos:
                    break
                for photo in photos:
                    if photo["id"] in seen_ids:
                        continue
                    seen_ids.add(photo["id"])
                    url = photo.get("url_l") or photo.get("url_o")
                    if url:
                        label_candidates.append(SourceImage(dish.canonical_label, url, f"https://www.flickr.com/photos/{photo['owner']}/{photo['id']}/", "flickr", photo.get("title", ""), f"Flickr license id {photo.get('license', 'unknown')}", photo.get("ownername")))
                page += 1
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            if len(label_candidates) >= per_label:
                break
        candidates.extend(label_candidates[:per_label])
    return candidates


async def _download(client: Any, candidates: Iterable[SourceImage], output_dir: Path, dishes: dict[str, Dish], stats: Counter[str], *, min_width: int, min_height: int, max_bytes: int, request_delay: float, max_retries: int, target_per_class: int | None, download_concurrency: int) -> Counter[str]:
    root = output_dir / "images"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "metadata.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line] if manifest_path.exists() else []
    # Records may come from another pipeline (e.g. the CLIP-expanded set) whose
    # schema omits image_url/alt/etc.; dedup only needs sha256 + perceptual_hash.
    hashes = {record["sha256"] for record in records if record.get("sha256")}
    perceptual_hashes = {record["perceptual_hash"] for record in records if record.get("perceptual_hash")}
    known_urls = {record["image_url"] for record in records if record.get("image_url")}
    class_counts: Counter[str] = Counter(record["canonical_label"] for record in records if record.get("canonical_label"))
    async def fetch(candidate: SourceImage) -> tuple[bytes, int, int]:
        url = _normalize_image_url(candidate.image_url)
        headers = {
            "User-Agent": client.headers["User-Agent"],
            "Api-User-Agent": client.headers.get("Api-User-Agent", client.headers["User-Agent"]),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if "wikimedia.org" in url:
            headers["Referer"] = "https://commons.wikimedia.org/"
            headers["Origin"] = "https://commons.wikimedia.org"
        try:
            response = await _get_with_retry(client, url, headers=headers, max_retries=max_retries)
            if response.status_code == 403 and candidate.provider == "wikimedia_commons":
                headers["Referer"] = candidate.source_url
                response = await _get_with_retry(client, url, headers=headers, max_retries=max_retries)
            response.raise_for_status()
            if len(response.content) > max_bytes:
                raise ValueError("image exceeds max_bytes")
            width, height, _ = _validate_image(response.content, min_width, min_height)
            return _jpeg_bytes(response.content), width, height
        finally:
            if request_delay > 0:
                await asyncio.sleep(request_delay)

    async def save_batch(manifest: Any, batch: list[SourceImage]) -> None:
        results = await asyncio.gather(*(fetch(candidate) for candidate in batch), return_exceptions=True)
        for candidate, result in zip(batch, results):
            if isinstance(result, BaseException):
                stats["rejected"] += 1
                stats[f"rejected:{type(result).__name__}: {str(result)[:160]}"] += 1
                continue
            if target_per_class is not None and class_counts[candidate.label] >= target_per_class:
                stats["target_reached_skips"] += 1
                continue
            jpeg_data, width, height = result
            sha256 = hashlib.sha256(jpeg_data).hexdigest()
            if sha256 in hashes:
                stats["duplicates"] += 1
                continue
            perceptual_hash = _perceptual_hash(jpeg_data)
            if perceptual_hash in perceptual_hashes:
                stats["perceptual_duplicates"] += 1
                continue
            class_dir = root / candidate.label
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / f"{candidate.label}_{sha256[:16]}.jpg"
            target.write_bytes(jpeg_data)
            manifest.write(json.dumps({
                "image_path": str(target.relative_to(output_dir)).replace("\\", "/"), "canonical_label": candidate.label,
                "vietnamese_name": dishes[candidate.label].vietnamese_name, "provider": candidate.provider,
                "source_url": candidate.source_url, "image_url": candidate.image_url, "alt": candidate.alt,
                "license": candidate.license, "attribution": candidate.attribution, "sha256": sha256,
                "perceptual_hash": perceptual_hash, "width": width, "height": height,
                "format": "JPEG", "content_type": "image/jpeg",
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            hashes.add(sha256)
            perceptual_hashes.add(perceptual_hash)
            known_urls.add(candidate.image_url)
            class_counts[candidate.label] += 1
            stats["downloaded"] += 1

    with manifest_path.open("a", encoding="utf-8") as manifest:
        batch: list[SourceImage] = []
        for candidate in candidates:
            stats["candidates"] += 1
            if target_per_class is not None and class_counts[candidate.label] >= target_per_class:
                stats["target_reached_skips"] += 1
                continue
            if candidate.image_url in known_urls:
                stats["already_downloaded"] += 1
                continue
            batch.append(candidate)
            if len(batch) >= download_concurrency:
                await save_batch(manifest, batch)
                batch = []
        if batch:
            await save_batch(manifest, batch)
    return class_counts


async def run_hybrid(output_dir: Path, *, providers: Iterable[str] = ("web", "commons"), priorities: Iterable[str] = ("P0",), seed_file: Path | None = None, per_label: int = 25, target_per_class: int | None = None, min_width: int = 256, min_height: int = 256, max_bytes: int = 10 * 1024 * 1024, request_delay: float = 1.5, max_retries: int = 4, http_timeout: float = 30, download_concurrency: int = 1) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install crawler dependencies: pip install -r requirements-crawler.txt") from exc
    if request_delay < 0 or max_retries < 0 or per_label < 1 or http_timeout <= 0 or download_concurrency < 1:
        raise ValueError("request_delay/max_retries must be non-negative and per_label must be positive")
    if target_per_class is not None and target_per_class < 1:
        raise ValueError("target_per_class must be positive")
    enabled = {value.strip().lower() for value in providers}
    invalid = enabled - {"web", "commons", "pexels", "flickr"}
    if invalid:
        raise ValueError(f"Unknown providers: {', '.join(sorted(invalid))}")
    contact = os.getenv("WIKIMEDIA_CONTACT")
    if "commons" in enabled and not contact:
        raise RuntimeError(
            "WIKIMEDIA_CONTACT is required for provider commons. "
            "Set it to a contact email or public project URL required by Wikimedia's User-Agent policy."
        )
    selected = _dishes(priorities)
    dish_by_label = {dish.canonical_label: dish for dish in DISHES}
    seeds = read_seeds(seed_file) if seed_file else []
    unknown = sorted({seed.label for seed in seeds} - dish_by_label.keys())
    if unknown:
        raise ValueError(f"Unknown seed labels: {', '.join(unknown)}")
    stats: Counter[str] = Counter()
    agent = (
        f"FoodClassificationCrawler/0.2 ({contact}) httpx"
        if contact
        else "FoodClassificationCrawler/0.2 (web-crawl; respect robots and site terms)"
    )
    headers = {"User-Agent": agent, "Api-User-Agent": agent, "Accept-Encoding": "gzip"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=http_timeout, headers=headers) as client:
        candidates: list[SourceImage] = []
        if "web" in enabled:
            candidates += await _web_candidates(seeds, stats)
        if "commons" in enabled:
            candidates += await _commons_candidates(client, selected, per_label, stats, request_delay, max_retries)
        if "pexels" in enabled:
            key = os.getenv("PEXELS_API_KEY") or ""
            if not key:
                raise RuntimeError("PEXELS_API_KEY is required when using provider pexels")
            candidates += await _pexels_candidates(client, selected, per_label, key, stats, request_delay, max_retries)
        if "flickr" in enabled:
            key = os.getenv("FLICKR_API_KEY") or ""
            if not key:
                raise RuntimeError("FLICKR_API_KEY is required when using provider flickr")
            candidates += await _flickr_candidates(client, selected, per_label, key, stats, request_delay, max_retries)
        class_counts = await _download(client, candidates, output_dir, dish_by_label, stats, min_width=min_width, min_height=min_height, max_bytes=max_bytes, request_delay=request_delay, max_retries=max_retries, target_per_class=target_per_class, download_concurrency=download_concurrency)
    summary = {key: value for key, value in stats.items() if not key.startswith("rejected:")}
    summary["rejection_reasons"] = {key.removeprefix("rejected:"): value for key, value in stats.most_common(20) if key.startswith("rejected:")}
    summary["providers"] = sorted(enabled)
    summary["priorities"] = sorted({dish.priority for dish in selected})
    summary["class_counts"] = {dish.canonical_label: class_counts[dish.canonical_label] for dish in selected}
    if target_per_class is not None:
        summary["target_per_class"] = target_per_class
        summary["classes_below_target"] = {
            dish.canonical_label: target_per_class - class_counts[dish.canonical_label]
            for dish in selected if class_counts[dish.canonical_label] < target_per_class
        }
    return summary
