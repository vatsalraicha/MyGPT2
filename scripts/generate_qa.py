#!/usr/bin/env python3
"""Standalone Q&A generation script (decoupled from finetune_model.py).

Generates Q&A pairs for books using the v2 qa_generate module.
Supports manual review mode (--sample), single book, or full corpus.

Usage:
    # Generate for 3 sample books (1 small + 1 medium + 1 large) for manual review
    python scripts/generate_qa.py --sample 3 --version v2

    # Generate for a specific book
    python scripts/generate_qa.py --book-id calculus_made_easy --version v2

    # Generate for all books
    python scripts/generate_qa.py --version v2

    # Dry run (generate + print stats, don't write files)
    python scripts/generate_qa.py --sample 3 --dry-run

    # Use cleaned text directory
    python scripts/generate_qa.py --version v2 --books-dir data/books/cleaned
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.data.qa_generate import generate_qa_pairs
from bookgpt.utils.paths import versioned_paths, add_version_arg, ensure_dirs

logger = logging.getLogger(__name__)


def select_sample_books(manifest: list[dict], n: int = 3) -> list[dict]:
    """Select a diverse sample: 1 small + 1 medium + 1 large book.

    For n=3, picks from the 25th, 50th, and 75th percentile by size.
    For other n, distributes evenly across the size range.
    """
    # Sort by char count
    sorted_books = sorted(manifest, key=lambda e: e.get("token_count", 0))
    total = len(sorted_books)

    if total <= n:
        return sorted_books

    # Pick evenly spaced indices
    indices = []
    for i in range(n):
        idx = int((i + 0.5) * total / n)
        indices.append(min(idx, total - 1))

    selected = [sorted_books[i] for i in indices]
    return selected


def print_qa_examples(qa_pairs: list[dict], book_id: str, examples_per_type: int = 3):
    """Print sample Q&A pairs grouped by type."""
    by_type = {}
    for p in qa_pairs:
        t = p.get("type", "unknown")
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(p)

    print(f"\n{'━' * 70}")
    print(f"  📖 {book_id}  —  {len(qa_pairs)} total pairs")
    print(f"{'━' * 70}")

    for qa_type, pairs in sorted(by_type.items()):
        print(f"\n  ── {qa_type.upper()} ({len(pairs)} pairs) ──")
        for j, p in enumerate(pairs[:examples_per_type]):
            print(f"    Q: {p['question'][:100]}")
            print(f"    A: {p['answer'][:120]}")
            if j < min(examples_per_type, len(pairs)) - 1:
                print()


def print_stats_table(all_stats: list[dict]):
    """Print summary stats table across all processed books."""
    print(f"\n{'━' * 110}")
    print(f"  Q&A GENERATION STATS")
    print(f"{'━' * 110}")
    print(f"  {'Book':<40} {'Total':>6} {'Def':>5} {'Thm':>5} {'Form':>5} "
          f"{'KW':>5} {'Sem':>5} {'Cloze':>6} {'Comp':>5} {'Sum':>5} {'AvgA':>5}")
    print(f"  {'─' * 107}")

    for s in all_stats:
        types = s["type_counts"]
        print(f"  {s['book_id']:<40} {s['total']:>6} "
              f"{types.get('definition', 0):>5} "
              f"{types.get('theorem', 0):>5} "
              f"{types.get('formula', 0):>5} "
              f"{types.get('keyword', 0):>5} "
              f"{types.get('semantic', 0):>5} "
              f"{types.get('cloze', 0):>6} "
              f"{types.get('comprehension', 0):>5} "
              f"{types.get('summary', 0):>5} "
              f"{s['avg_answer_len']:>5.1f}")

    print(f"  {'─' * 107}")
    total = sum(s["total"] for s in all_stats)
    avg_a = sum(s["avg_answer_len"] * s["total"] for s in all_stats) / max(total, 1)
    print(f"  {'TOTAL':<40} {total:>6} {'':>5} {'':>5} {'':>5} {'':>5} {'':>5} {'':>6} {'':>5} {'':>5} {avg_a:>5.1f}")
    print(f"{'━' * 110}")


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs for books")
    parser.add_argument(
        "--manifest", type=str, default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--tokenizer-dir", type=str, default="data/tokenizers/shared",
        help="Path to shared tokenizer directory",
    )
    parser.add_argument(
        "--books-dir", type=str, default=None,
        help="Override books directory (default: data/books/cleaned, falls back to raw)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    add_version_arg(parser)
    parser.add_argument("--book-id", type=str, default=None, help="Process specific book only")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N books (1 small + 1 medium + 1 large) for manual review",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate and print stats but don't write files")
    parser.add_argument(
        "--examples-per-type", type=int, default=3,
        help="Number of examples to print per Q&A type in sample mode",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = versioned_paths(config, args.version)
    ensure_dirs(paths)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{paths['logs_dir']}/generate_qa.log"),
        ],
    )
    logger.info(f"Run version: {paths['version']}")

    # Determine books directory
    if args.books_dir:
        books_dir = Path(args.books_dir)
    else:
        # Prefer cleaned, fall back to raw
        cleaned_dir = Path("data/books/cleaned")
        if cleaned_dir.exists() and any(cleaned_dir.glob("*.txt")):
            books_dir = cleaned_dir
            logger.info("Using cleaned books from data/books/cleaned/")
        else:
            books_dir = Path(paths["books_dir"])
            logger.info("Using raw books (no cleaned directory found)")

    # Verify tokenizer
    tokenizer_dir = Path(args.tokenizer_dir)
    if not (tokenizer_dir / "tokenizer.json").exists():
        logger.error(f"Shared tokenizer not found at {tokenizer_dir}")
        logger.error("Run: python scripts/train_tokenizer.py first")
        return

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter books
    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            logger.error(f"Book ID '{args.book_id}' not found in manifest")
            return
    elif args.sample:
        manifest = select_sample_books(manifest, args.sample)
        book_ids = [e["book_id"] for e in manifest]
        logger.info(f"Sample mode: selected {len(manifest)} books: {book_ids}")

    # Q&A config from YAML
    qa_config = config.get("qa_generate", {})

    qa_dir = Path(paths["qa_dir"])
    all_stats = []

    for i, entry in enumerate(manifest):
        book_id = entry["book_id"]
        book_path = books_dir / f"{book_id}.txt"
        qa_path = qa_dir / f"{book_id}.jsonl"

        if not book_path.exists():
            # Try the file_path from manifest as fallback
            book_path = Path(entry["file_path"])
            if not book_path.exists():
                logger.warning(f"Book file not found for {book_id}, skipping")
                continue

        logger.info(f"[{i+1}/{len(manifest)}] Generating Q&A for: {book_id}")

        # Generate Q&A
        # If using cleaned books dir, text is already clean — skip re-cleaning
        clean_text = str(books_dir) != "data/books/cleaned"

        if args.dry_run:
            # Generate but write to /dev/null equivalent
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as tmp:
                qa_pairs = generate_qa_pairs(
                    book_path=str(book_path),
                    tokenizer_dir=str(tokenizer_dir),
                    output_path=tmp.name,
                    chunk_size=qa_config.get("chunk_size", 500),
                    chunk_overlap=qa_config.get("chunk_overlap", 50),
                    min_pairs=qa_config.get("min_qa_pairs", 500),
                    max_pairs=qa_config.get("max_qa_pairs", 2000),
                    max_cloze_fraction=qa_config.get("max_cloze_fraction", 0.30),
                    min_answer_words=qa_config.get("min_answer_words", 3),
                    max_math_density=qa_config.get("max_math_density", 0.40),
                    clean_text=clean_text,
                )
        else:
            qa_pairs = generate_qa_pairs(
                book_path=str(book_path),
                tokenizer_dir=str(tokenizer_dir),
                output_path=str(qa_path),
                chunk_size=qa_config.get("chunk_size", 500),
                chunk_overlap=qa_config.get("chunk_overlap", 50),
                min_pairs=qa_config.get("min_qa_pairs", 500),
                max_pairs=qa_config.get("max_qa_pairs", 2000),
                max_cloze_fraction=qa_config.get("max_cloze_fraction", 0.30),
                min_answer_words=qa_config.get("min_answer_words", 3),
                max_math_density=qa_config.get("max_math_density", 0.40),
                clean_text=clean_text,
            )

        # Compute stats
        type_counts = {}
        total_answer_words = 0
        for p in qa_pairs:
            t = p.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            total_answer_words += len(p["answer"].split())

        stats = {
            "book_id": book_id,
            "total": len(qa_pairs),
            "type_counts": type_counts,
            "avg_answer_len": total_answer_words / max(len(qa_pairs), 1),
        }
        all_stats.append(stats)

        # Print examples in sample mode
        if args.sample:
            print_qa_examples(qa_pairs, book_id, args.examples_per_type)

    # Print stats table
    print_stats_table(all_stats)

    if args.dry_run:
        print("\n  ⚠️  DRY RUN — no files were written")
    else:
        print(f"\n  ✅ Q&A files written to: {qa_dir}/")


if __name__ == "__main__":
    main()
