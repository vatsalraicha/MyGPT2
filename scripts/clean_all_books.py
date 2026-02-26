#!/usr/bin/env python3
"""Clean all raw book texts and write to data/books/cleaned/.

Applies text_cleaner.clean_book_text() to each book in the manifest,
writing cleaned versions to a parallel directory. Raw files are never modified.

Usage:
    python scripts/clean_all_books.py                          # clean all books
    python scripts/clean_all_books.py --skip-arxiv             # skip arxiv papers
    python scripts/clean_all_books.py --book-id calculus_made_easy  # single book
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.data.text_cleaner import clean_for_pretrain


def main():
    parser = argparse.ArgumentParser(description="Clean all raw book texts")
    parser.add_argument("--manifest", type=str, default="data/books/manifest.json",
                        help="Path to book manifest")
    parser.add_argument("--output-dir", type=str, default="data/books/cleaned",
                        help="Directory for cleaned texts")
    parser.add_argument("--skip-arxiv", action="store_true",
                        help="Skip arxiv papers (book_id starts with 'arxiv_')")
    parser.add_argument("--book-id", type=str, default=None,
                        help="Clean a single book only")
    parser.add_argument("--force", action="store_true",
                        help="Re-clean even if cleaned file already exists")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            print(f"Error: book_id '{args.book_id}' not found in manifest")
            sys.exit(1)

    if args.skip_arxiv:
        before = len(manifest)
        manifest = [e for e in manifest if not e["book_id"].startswith("arxiv_")]
        print(f"Skipping arxiv: {before - len(manifest)} removed, {len(manifest)} remaining")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_orig = 0
    total_clean = 0
    cleaned_count = 0
    skipped_count = 0

    print(f"\n{'#':<3} {'Book ID':<45} {'Original':>10} {'Cleaned':>10} {'Reduction':>10}")
    print("=" * 82)

    for i, entry in enumerate(manifest):
        book_id = entry["book_id"]
        raw_path = Path(entry["file_path"])
        clean_path = output_dir / f"{book_id}.txt"

        if not raw_path.exists():
            print(f"{i+1:<3} {book_id:<45} {'MISSING':>10}")
            continue

        if clean_path.exists() and not args.force:
            skipped_count += 1
            continue

        raw_text = raw_path.read_text(encoding="utf-8")
        cleaned_text = clean_for_pretrain(raw_text)

        clean_path.write_text(cleaned_text, encoding="utf-8")

        orig_len = len(raw_text)
        clean_len = len(cleaned_text)
        reduction = (orig_len - clean_len) / orig_len * 100 if orig_len > 0 else 0
        total_orig += orig_len
        total_clean += clean_len
        cleaned_count += 1

        print(f"{i+1:<3} {book_id:<45} {orig_len:>10,} {clean_len:>10,} {reduction:>9.1f}%")

    print("=" * 82)
    if total_orig > 0:
        total_reduction = (total_orig - total_clean) / total_orig * 100
        print(f"    {'TOTAL':<45} {total_orig:>10,} {total_clean:>10,} {total_reduction:>9.1f}%")
    print(f"\n  Cleaned: {cleaned_count}  |  Skipped (already exists): {skipped_count}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
