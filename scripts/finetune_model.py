#!/usr/bin/env python3
"""Script to generate Q&A data and fine-tune models for each book (shared tokenizer)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.data.qa_generate import generate_qa_pairs
from bookgpt.model.finetune import finetune_book
from bookgpt.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/finetune.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A data and fine-tune book models")
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
        "--pretrained-dir",
        type=str,
        default="data/models/pretrained",
        help="Base directory for pretrained models",
    )
    parser.add_argument(
        "--finetuned-dir",
        type=str,
        default="data/models/finetuned",
        help="Base directory for fine-tuned model output",
    )
    parser.add_argument(
        "--qa-dir",
        type=str,
        default="data/books/qa",
        help="Directory for Q&A data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--book-id", type=str, default=None, help="Process specific book only")
    parser.add_argument("--skip-qa-gen", action="store_true", help="Skip Q&A generation (use existing)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    Path("logs").mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    qa_config = config.get("qa_generate", {})
    ft_config = config.get("finetune", {})

    for entry in manifest:
        book_id = entry["book_id"]
        book_path = entry["file_path"]
        pretrained_dir = Path(args.pretrained_dir) / book_id
        finetuned_dir = Path(args.finetuned_dir) / book_id
        qa_path = Path(args.qa_dir) / f"{book_id}.jsonl"

        if not pretrained_dir.exists():
            logger.warning(f"Pretrained model not found for {book_id}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {entry['title']} ({book_id})")
        logger.info(f"{'='*60}")

        # Step 1: Generate Q&A pairs (if not skipping)
        if not args.skip_qa_gen or not qa_path.exists():
            logger.info("Generating Q&A pairs...")
            qa_pairs = generate_qa_pairs(
                book_path=book_path,
                tokenizer_dir=str(tokenizer_dir),
                output_path=str(qa_path),
                chunk_size=qa_config.get("chunk_size", 500),
                chunk_overlap=qa_config.get("chunk_overlap", 50),
                min_pairs=qa_config.get("min_qa_pairs", 500),
                max_pairs=qa_config.get("max_qa_pairs", 2000),
            )
            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        else:
            logger.info(f"Using existing Q&A data: {qa_path}")

        # Step 2: Fine-tune
        logger.info("Fine-tuning model...")
        stats = finetune_book(
            pretrained_dir=str(pretrained_dir),
            tokenizer_dir=str(tokenizer_dir),
            qa_path=str(qa_path),
            save_dir=str(finetuned_dir),
            finetune_config=ft_config,
            device=device,
            seed=config.get("seed", 42),
        )

        logger.info(f"Fine-tuning stats for {book_id}: {stats}")

    print(f"\nFine-tuning complete for {len(manifest)} books")


if __name__ == "__main__":
    main()
