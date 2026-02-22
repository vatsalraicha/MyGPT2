#!/usr/bin/env python3
"""Script to crawl and download math books from Project Gutenberg, OpenStax, and arXiv."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.crawl.gutenberg import crawl_gutenberg
from bookgpt.crawl.openstax import crawl_openstax
from bookgpt.crawl.arxiv import crawl_arxiv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/crawl.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Crawl math books for BookGPT")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/books/raw",
        help="Directory to save cleaned book texts",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to save the manifest file",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=15,
        help="Maximum books to download per source",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Maximum arXiv papers to download",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["gutenberg", "openstax", "arxiv"],
        choices=["gutenberg", "openstax", "arxiv"],
        help="Sources to crawl",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing manifest instead of overwriting",
    )
    args = parser.parse_args()

    # Ensure directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Load existing manifest if appending
    all_metadata = []
    existing_ids = set()
    if args.append and Path(args.manifest).exists():
        with open(args.manifest) as f:
            all_metadata = json.load(f)
            existing_ids = {e["book_id"] for e in all_metadata}
        logger.info(f"Loaded existing manifest with {len(all_metadata)} entries")

    if "gutenberg" in args.sources:
        logger.info("=== Crawling Project Gutenberg ===")
        meta = crawl_gutenberg(
            output_dir=args.output_dir,
            max_books=args.max_books,
            delay_seconds=args.delay,
        )
        # Only add new entries
        for entry in meta:
            if entry["book_id"] not in existing_ids:
                all_metadata.append(entry)
                existing_ids.add(entry["book_id"])

    if "openstax" in args.sources:
        logger.info("=== Crawling OpenStax ===")
        meta = crawl_openstax(
            output_dir=args.output_dir,
            max_books=args.max_books,
            delay_seconds=args.delay,
        )
        for entry in meta:
            if entry["book_id"] not in existing_ids:
                all_metadata.append(entry)
                existing_ids.add(entry["book_id"])

    if "arxiv" in args.sources:
        logger.info("=== Crawling arXiv ===")
        meta = crawl_arxiv(
            output_dir=args.output_dir,
            max_papers=args.max_papers,
            delay_seconds=max(args.delay, 3.0),  # arXiv requires 3s minimum
        )
        for entry in meta:
            if entry["book_id"] not in existing_ids:
                all_metadata.append(entry)
                existing_ids.add(entry["book_id"])

    # Filter out very small entries (< 10K chars)
    before = len(all_metadata)
    all_metadata = [e for e in all_metadata if e["token_count"] >= 10000 or e["source"] == "arxiv"]
    # For arXiv, allow smaller (papers are shorter than books), but still filter tiny ones
    all_metadata = [e for e in all_metadata if e["token_count"] >= 2000]
    if len(all_metadata) < before:
        logger.info(f"Filtered out {before - len(all_metadata)} entries with insufficient text")

    # Save manifest
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest saved with {len(all_metadata)} entries -> {args.manifest}")

    # Print summary
    total_chars = sum(e["token_count"] for e in all_metadata)
    print(f"\n{'='*70}")
    print(f"Crawling complete: {len(all_metadata)} items ({total_chars / 1024 / 1024:.1f} MB total)")
    print(f"{'='*70}")

    for source in ["gutenberg", "openstax", "arxiv"]:
        items = [e for e in all_metadata if e["source"] == source]
        if items:
            src_chars = sum(e["token_count"] for e in items)
            print(f"\n  [{source}] {len(items)} items ({src_chars / 1024 / 1024:.1f} MB)")
            for entry in items:
                print(f"    • {entry['title'][:70]} ({entry['token_count']:,} chars)")

    print(f"\nManifest: {args.manifest}")
    print(f"Book files: {args.output_dir}/")


if __name__ == "__main__":
    main()
