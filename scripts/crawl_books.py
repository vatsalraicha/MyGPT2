#!/usr/bin/env python3
"""Script to crawl and download math books from Project Gutenberg and OpenStax."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.crawl.gutenberg import crawl_gutenberg
from bookgpt.crawl.openstax import crawl_openstax

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
        default=10,
        help="Maximum books to download per source",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["gutenberg", "openstax"],
        choices=["gutenberg", "openstax"],
        help="Sources to crawl",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds",
    )
    args = parser.parse_args()

    # Ensure directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    all_metadata = []

    if "gutenberg" in args.sources:
        logger.info("=== Crawling Project Gutenberg ===")
        meta = crawl_gutenberg(
            output_dir=args.output_dir,
            max_books=args.max_books,
            delay_seconds=args.delay,
        )
        all_metadata.extend(meta)

    if "openstax" in args.sources:
        logger.info("=== Crawling OpenStax ===")
        meta = crawl_openstax(
            output_dir=args.output_dir,
            max_books=args.max_books,
            delay_seconds=args.delay,
        )
        all_metadata.extend(meta)

    # Save manifest
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest saved with {len(all_metadata)} books -> {args.manifest}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Crawling complete: {len(all_metadata)} books downloaded")
    print(f"{'='*60}")
    for entry in all_metadata:
        print(f"  [{entry['source']}] {entry['title']} ({entry['token_count']:,} chars)")
    print(f"\nManifest: {args.manifest}")
    print(f"Book files: {args.output_dir}/")


if __name__ == "__main__":
    main()
