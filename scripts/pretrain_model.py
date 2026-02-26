#!/usr/bin/env python3
"""Script to pretrain GPT-2 models on individual books (shared tokenizer)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.model.pretrain import pretrain_book
from bookgpt.model.generate import sample_from_model
from bookgpt.utils.device import get_device
from bookgpt.utils.paths import versioned_paths, add_version_arg, ensure_dirs

# Logging setup deferred to main() so version-aware log path can be used
logger = logging.getLogger(__name__)


def _extract_prompts(book_path: str, n: int = 3) -> list[str]:
    """Extract real sentence openings from a book to use as generation prompts."""
    import re

    text = Path(book_path).read_text(encoding="utf-8")

    # Find clean sentences (no TeX noise, reasonable length)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = []
    for s in sentences:
        s = s.strip()
        # Skip short, TeX-heavy, or noisy sentences
        if len(s) < 30 or len(s) > 200:
            continue
        if s.count("$") > 3 or s.count("\\") > 2:
            continue
        if not s[0].isupper():
            continue
        # Take first 5-8 words as prompt
        words = s.split()
        if len(words) >= 5:
            prompt = " ".join(words[:6])
            candidates.append(prompt)

    if not candidates:
        # Fallback: just grab some text chunks
        for i in range(0, min(len(text), 5000), 1500):
            chunk = text[i:i+60].strip().split("\n")[0]
            if len(chunk) > 10:
                candidates.append(chunk[:40])

    # Pick evenly spaced prompts from the book
    if len(candidates) <= n:
        return candidates or ["The", "In this", "We have"]
    step = len(candidates) // n
    return [candidates[i * step] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 on book(s)")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="data/tokenizers/shared",
        help="Path to shared tokenizer directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for model output (default: version-aware path)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    add_version_arg(parser)
    parser.add_argument("--book-id", type=str, default=None, help="Train specific book only")
    parser.add_argument("--min-chars", type=int, default=5000, help="Minimum chars to train (skip smaller)")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arxiv papers (book_id starts with 'arxiv_')")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    # Load config and resolve versioned paths
    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = versioned_paths(config, args.version)
    ensure_dirs(paths)

    # Override output-dir if not explicitly set
    if args.output_dir is None:
        args.output_dir = paths["pretrained_dir"]

    # Setup logging with versioned log path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{paths['logs_dir']}/pretrain.log"),
        ],
    )
    logger.info(f"Run version: {paths['version']}")

    device = get_device(force_cpu=args.force_cpu)
    logger.info(f"Using device: {device}")

    # Verify shared tokenizer exists
    tokenizer_dir = Path(args.tokenizer_dir)
    if not (tokenizer_dir / "tokenizer.json").exists():
        logger.error(f"Shared tokenizer not found at {tokenizer_dir}")
        logger.error("Run: python scripts/train_tokenizer.py first")
        return

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            logger.error(f"Book ID '{args.book_id}' not found in manifest")
            return

    if args.skip_arxiv:
        before = len(manifest)
        manifest = [e for e in manifest if not e["book_id"].startswith("arxiv_")]
        logger.info(f"Skipping arxiv papers: {before - len(manifest)} removed, {len(manifest)} books remaining")

    # Use raw text for pretraining (cleaned text hurts PPL — see Phase 8 experiments)
    # Cleaned text is used for Q&A generation only
    skipped = []
    trained = []

    for i, entry in enumerate(manifest):
        book_id = entry["book_id"]
        book_path = entry["file_path"]
        char_count = entry.get("token_count", 0)  # Note: manifest calls it token_count but it's chars
        save_dir = Path(args.output_dir) / book_id

        # Skip books that are too small to train meaningfully
        if char_count < args.min_chars:
            logger.info(f"Skipping {book_id}: too small ({char_count:,} chars < {args.min_chars:,})")
            skipped.append(book_id)
            continue

        # Skip if already trained (checkpoint exists)
        if (save_dir / "best" / "model.pt").exists():
            logger.info(f"Skipping {book_id}: already trained (checkpoint exists)")
            trained.append(book_id)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(manifest)}] Pretraining: {entry['title']} ({book_id})")
        logger.info(f"{'='*60}")

        stats = pretrain_book(
            book_path=book_path,
            tokenizer_dir=str(tokenizer_dir),
            save_dir=str(save_dir),
            model_config=config.get("model", {}),
            train_config=config.get("pretrain", {}),
            device=device,
            seed=config.get("seed", 42),
        )

        logger.info(f"Training stats for {book_id}: {stats}")

        # Generate samples using actual sentences from the book as prompts
        print(f"\n--- Sample generations from {book_id} ---")
        prompts = _extract_prompts(book_path, n=3)
        try:
            samples = sample_from_model(
                model_dir=str(save_dir / "best"),
                tokenizer_dir=str(tokenizer_dir),
                prompts=prompts,
                device=device,
                max_new_tokens=100,
                temperature=0.8,
            )
            for prompt, sample in zip(prompts, samples):
                print(f"\nPrompt: {prompt}")
                print(f"Output: {sample[:200]}")
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")

        trained.append(book_id)

    print(f"\n{'='*60}")
    print(f"Pretraining complete!")
    print(f"  Trained: {len(trained)} models")
    print(f"  Skipped (too small): {len(skipped)} ({', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
