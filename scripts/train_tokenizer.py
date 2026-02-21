#!/usr/bin/env python3
"""Script to train BPE tokenizers for each book in the manifest."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.tokenizer.train_bpe import train_bpe_tokenizer, load_tokenizer, tokenize_text, decode_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/tokenizer.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizers for BookGPT")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenizers",
        help="Base directory for tokenizer output",
    )
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--book-id", type=str, default=None, help="Train for specific book only")
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            logger.error(f"Book ID '{args.book_id}' not found in manifest")
            return

    for entry in manifest:
        book_id = entry["book_id"]
        book_path = entry["file_path"]
        output_dir = Path(args.output_dir) / book_id

        logger.info(f"Training tokenizer for: {entry['title']} ({book_id})")

        tokenizer = train_bpe_tokenizer(
            text_path=book_path,
            output_dir=output_dir,
            vocab_size=args.vocab_size,
        )

        # Validation: encode and decode a sample
        text = Path(book_path).read_text(encoding="utf-8")[:500]
        ids = tokenize_text(tokenizer, text)
        decoded = decode_tokens(tokenizer, ids)

        logger.info(f"  Vocab size: {tokenizer.get_vocab_size()}")
        logger.info(f"  Sample: {len(text)} chars -> {len(ids)} tokens")
        logger.info(f"  Round-trip match: {text[:100] == decoded[:100]}")

    print(f"\nTokenizers trained for {len(manifest)} books -> {args.output_dir}/")


if __name__ == "__main__":
    main()
