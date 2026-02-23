"""Fine-tuning loop for Q&A on per-book GPT-2 models."""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from bookgpt.model.gpt2 import GPT2
from bookgpt.model.pretrain import Trainer
from bookgpt.data.prepare import QADataset
from bookgpt.tokenizer.train_bpe import load_tokenizer
from bookgpt.utils.device import get_device, set_seed

logger = logging.getLogger(__name__)


def finetune_book(
    pretrained_dir: str | Path,
    tokenizer_dir: str | Path,
    qa_path: str | Path,
    save_dir: str | Path,
    finetune_config: dict | None = None,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Fine-tune a pretrained book model on Q&A data.

    Args:
        pretrained_dir: Path to pretrained model directory (containing best/ or final/).
        tokenizer_dir: Path to tokenizer directory.
        qa_path: Path to Q&A JSONL file.
        save_dir: Where to save fine-tuned model.
        finetune_config: Fine-tuning hyperparameters.
        device: Device to train on.
        seed: Random seed.

    Returns:
        Training statistics dict.
    """
    set_seed(seed)

    if device is None:
        device = get_device()

    # Load pretrained model (prefer 'best' checkpoint)
    pretrained_dir = Path(pretrained_dir)
    model_dir = pretrained_dir / "best"
    if not model_dir.exists():
        model_dir = pretrained_dir / "final"
    if not model_dir.exists():
        model_dir = pretrained_dir

    logger.info(f"Loading pretrained model from {model_dir}")
    model = GPT2.from_pretrained(model_dir, device=device)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)

    # Load Q&A data
    qa_path = Path(qa_path)
    qa_examples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_examples.append(json.loads(line))

    logger.info(f"Loaded {len(qa_examples)} Q&A pairs from {qa_path}")

    if not qa_examples:
        logger.error("No Q&A examples found!")
        return {"error": "no_data"}

    # Create dataset
    full_dataset = QADataset(
        examples=qa_examples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
    )

    # Train/val split (90/10)
    n_train = int(len(full_dataset) * 0.9)
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"QA Train: {n_train} samples, Val: {n_val} samples")

    # Fine-tuning config (lower LR, fewer epochs)
    fc = finetune_config or {}
    config = {
        "batch_size": fc.get("batch_size", 16),
        "learning_rate": fc.get("learning_rate", 5e-5),
        "weight_decay": fc.get("weight_decay", 0.01),
        "betas": fc.get("betas", [0.9, 0.95]),
        "max_epochs": fc.get("max_epochs", 10),
        "warmup_fraction": fc.get("warmup_fraction", 0.1),
        "grad_clip": fc.get("grad_clip", 1.0),
        "patience": fc.get("patience", 3),
        "eval_interval": fc.get("eval_interval", 100),
        "save_interval": fc.get("save_interval", 500),
        "gradient_accumulation_steps": fc.get("gradient_accumulation_steps", 1),
    }

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=save_dir,
        device=device,
        stage="finetune",
    )

    stats = trainer.train()
    logger.info(f"Fine-tuning complete: {stats}")
    return stats
