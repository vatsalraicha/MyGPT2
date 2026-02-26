#!/usr/bin/env python3
"""Test pretraining with an external tokenizer (e.g., GPT-2 tiktoken).

Compares our custom BPE tokenizer against a pretrained tokenizer to measure
the impact of tokenization quality on pretraining perplexity.

Usage:
    python scripts/test_external_tokenizer.py --book the_elements_of_non-euclidean_geometry
    python scripts/test_external_tokenizer.py --book calculus_made_easy --lr 3e-4
    python scripts/test_external_tokenizer.py --book calculus_made_easy --tokenizer cl100k_base
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ext_tokenizer_test")


def main():
    parser = argparse.ArgumentParser(description="Test external tokenizer for pretraining")
    parser.add_argument("--book", type=str, required=True, help="Book ID (e.g., calculus_made_easy)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="tiktoken encoding name (gpt2, cl100k_base, etc.)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--books-dir", type=str, default="data/books/raw", help="Raw books directory")
    parser.add_argument("--output-dir", type=str, default="data/models/v2/pretrained",
                        help="Output directory for model")
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed. Run: pip install tiktoken")
        sys.exit(1)

    # Load config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # Find book
    book_path = Path(args.books_dir) / f"{args.book}.txt"
    if not book_path.exists():
        print(f"Book not found: {book_path}")
        sys.exit(1)

    # Tokenize with external tokenizer
    enc = tiktoken.get_encoding(args.tokenizer)
    text = book_path.read_text(encoding="utf-8")
    token_ids = enc.encode(text)
    logger.info(f"Tokenized: {len(text):,} chars -> {len(token_ids):,} tokens "
                f"(tokenizer={args.tokenizer}, vocab={enc.n_vocab})")

    # Create datasets
    from bookgpt.data.prepare import BookDataset

    context_length = args.context_length
    train_split = config["pretrain"].get("train_split", 0.9)
    split_idx = int(len(token_ids) * train_split)
    train_ids = np.array(token_ids[:split_idx], dtype=np.int64)
    val_ids = np.array(token_ids[split_idx:], dtype=np.int64)

    train_dataset = BookDataset(train_ids, context_length)
    val_dataset = BookDataset(val_ids, context_length)
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Build model
    from bookgpt.model.gpt2 import GPT2, GPT2Config

    mc = config["model"]
    model_config = GPT2Config(
        vocab_size=enc.n_vocab,
        n_layer=mc.get("n_layer", 6),
        n_head=mc.get("n_head", 12),
        n_embd=mc.get("n_embd", 768),
        context_length=context_length,
        dropout=mc.get("dropout", 0.1),
        bias=mc.get("bias", False),
    )
    logger.info(f"Model: {model_config.num_params():,} params")

    model = GPT2(model_config)

    # Train
    from bookgpt.model.pretrain import Trainer
    from bookgpt.utils.device import set_seed

    set_seed(42)

    save_dir = Path(args.output_dir) / args.book
    save_dir.mkdir(parents=True, exist_ok=True)

    tc = config["pretrain"]
    if args.lr:
        tc["learning_rate"] = args.lr

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=tc,
        save_dir=save_dir,
        stage="pretrain",
    )

    stats = trainer.train()

    print(f"\n{'='*60}")
    print(f"EXTERNAL TOKENIZER TEST RESULTS")
    print(f"{'='*60}")
    print(f"Book:          {args.book}")
    print(f"Tokenizer:     {args.tokenizer} (vocab={enc.n_vocab})")
    print(f"Context:       {context_length}")
    print(f"Model params:  {model_config.num_params():,}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"LR:            {tc['learning_rate']}")
    print(f"Best val loss: {stats['best_val_loss']:.4f}")
    print(f"Best val PPL:  {math.exp(min(stats['best_val_loss'], 20)):.1f}")
    print(f"Total epochs:  {stats['total_epochs']}")
    print(f"Total time:    {stats['total_time_seconds']/60:.1f}m")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
