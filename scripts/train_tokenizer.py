#!/usr/bin/env python3
"""Script to train a shared BPE tokenizer on all books in the manifest."""

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
    parser = argparse.ArgumentParser(description="Train shared BPE tokenizer for BookGPT")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenizers/shared",
        help="Directory for tokenizer output",
    )
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    if not manifest:
        logger.error("Empty manifest — no books to train on")
        return

    # Collect all book file paths
    text_paths = []
    total_chars = 0
    for entry in manifest:
        path = Path(entry["file_path"])
        if path.exists():
            text_paths.append(path)
            total_chars += entry["token_count"]
        else:
            logger.warning(f"File not found: {path}")

    logger.info(
        f"Training shared tokenizer on {len(text_paths)} files "
        f"({total_chars / 1024 / 1024:.1f} MB total text)"
    )

    # Train shared tokenizer on all books
    tokenizer = train_bpe_tokenizer(
        text_paths=text_paths,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )

    # Validation: test on a sample from each book
    print(f"\n{'='*60}")
    print(f"Shared tokenizer trained: vocab_size={tokenizer.get_vocab_size()}")
    print(f"Trained on {len(text_paths)} files ({total_chars / 1024 / 1024:.1f} MB)")
    print(f"{'='*60}\n")

    for entry in manifest[:5]:  # Show first 5
        path = Path(entry["file_path"])
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")[:500]
        ids = tokenize_text(tokenizer, text)
        decoded = decode_tokens(tokenizer, ids)
        compression = len(text) / max(len(ids), 1)

        print(f"  {entry['title'][:50]:50s} | {len(ids):5d} tokens | {compression:.1f} chars/tok")

    # Test special tokens
    print(f"\nSpecial tokens:")
    for tok in ["<|endoftext|>", "<|pad|>", "<|question|>", "<|answer|>", "<|context|>"]:
        tid = tokenizer.token_to_id(tok)
        print(f"  {tok} -> {tid}")

    # Test math content
    math_samples = [
        "The derivative of f(x) = x^2 is f'(x) = 2x",
        "Let $\\int_0^1 x^n dx = \\frac{1}{n+1}$",
        "Theorem: Every bounded sequence has a convergent subsequence.",
    ]
    print(f"\nMath tokenization samples:")
    for sample in math_samples:
        ids = tokenize_text(tokenizer, sample)
        print(f"  \"{sample[:60]}\" -> {len(ids)} tokens")

    print(f"\nTokenizer saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
