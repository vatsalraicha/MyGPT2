#!/usr/bin/env python3
"""Script to build and test the query router."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.router.router import BookRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/router.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the BookGPT query router")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/router",
        help="Directory to save router artifacts",
    )
    parser.add_argument(
        "--test-queries",
        nargs="*",
        default=[
            "What is the derivative of sin(x)?",
            "How do you solve a quadratic equation?",
            "What is the central limit theorem?",
            "Define a prime number",
            "Explain Euclidean geometry",
            "What is an integral?",
        ],
        help="Test queries to validate the router",
    )
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    # Build router
    logger.info("Building router index...")
    router = BookRouter()
    router.build_index(args.manifest)

    # Save
    router.save(args.output_dir)
    logger.info(f"Router saved to {args.output_dir}")

    # Test
    print(f"\n{'='*60}")
    print("Router Test Results")
    print(f"{'='*60}")

    for query in args.test_queries:
        results = router.route(query, top_k=3)
        print(f"\nQuery: \"{query}\"")
        for r in results:
            print(f"  → {r['book_id']} (score: {r['score']:.4f}) - {r.get('title', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"Router indexed {len(router.book_ids)} books")


if __name__ == "__main__":
    main()
