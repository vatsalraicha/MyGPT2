#!/usr/bin/env python3
"""Script to pretrain GPT-2 models on individual books."""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pretrain.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 on book(s)")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/books/manifest.json",
        help="Path to book manifest",
    )
    parser.add_argument(
        "--tokenizers-dir",
        type=str,
        default="data/tokenizers",
        help="Base directory for tokenizers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/pretrained",
        help="Base directory for model output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--book-id", type=str, default=None, help="Train specific book only")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(force_cpu=args.force_cpu)
    logger.info(f"Using device: {device}")

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            logger.error(f"Book ID '{args.book_id}' not found in manifest")
            return

    for entry in manifest:
        book_id = entry["book_id"]
        book_path = entry["file_path"]
        tokenizer_dir = Path(args.tokenizers_dir) / book_id
        save_dir = Path(args.output_dir) / book_id

        if not tokenizer_dir.exists():
            logger.warning(f"Tokenizer not found for {book_id}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Pretraining: {entry['title']} ({book_id})")
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

        # Generate samples
        print(f"\n--- Sample generations from {book_id} ---")
        prompts = ["The derivative", "A function is", "Let x be"]
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

    print(f"\nPretraining complete for {len(manifest)} books")


if __name__ == "__main__":
    main()
