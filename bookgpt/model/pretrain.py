"""Pretraining loop for GPT-2 on a single book."""

import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bookgpt.model.gpt2 import GPT2, GPT2Config
from bookgpt.utils.device import get_device, set_seed, mps_synchronize, mps_empty_cache

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for GPT-2 pretraining and fine-tuning."""

    def __init__(
        self,
        model: GPT2,
        train_dataset,
        val_dataset,
        config: dict,
        save_dir: str | Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()

        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay only on non-bias, non-layernorm params."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "ln_" in name or "bias" in name or "wpe" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        lr = self.config.get("learning_rate", 3e-4)
        betas = tuple(self.config.get("betas", [0.9, 0.95]))

        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine decay learning rate with warmup."""
        warmup_fraction = self.config.get("warmup_fraction", 0.05)
        warmup_steps = int(total_steps * warmup_fraction)
        lr = self.config.get("learning_rate", 3e-4)
        min_lr = lr * 0.1

        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        elif step > total_steps:
            return min_lr
        else:
            decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (lr - min_lr)

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set, return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)

            _, loss = self.model(input_ids, targets=labels, loss_mask=loss_mask)
            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with training statistics.
        """
        batch_size = self.config.get("batch_size", 32)
        max_epochs = self.config.get("max_epochs", 50)
        grad_clip = self.config.get("grad_clip", 1.0)
        patience = self.config.get("patience", 5)
        eval_interval = self.config.get("eval_interval", 500)
        save_interval = self.config.get("save_interval", 1000)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # MPS doesn't benefit from multiprocess loading
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * max_epochs
        effective_batch_size = batch_size * grad_accum_steps

        logger.info(
            f"Training config: {max_epochs} epochs, {steps_per_epoch} steps/epoch, "
            f"batch_size={batch_size}, grad_accum={grad_accum_steps}, "
            f"effective_batch={effective_batch_size}"
        )

        self.model.train()
        start_time = time.time()

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for step, batch in enumerate(train_loader):
                # Update learning rate
                lr = self._get_lr(self.global_step, total_steps)
                self._set_lr(lr)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask.to(self.device)

                # Forward
                _, loss = self.model(input_ids, targets=labels, loss_mask=loss_mask)
                loss = loss / grad_accum_steps

                # Backward
                loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * grad_accum_steps
                n_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % 100 == 0:
                    avg_loss = epoch_loss / n_batches
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {self.global_step} | Epoch {epoch+1}/{max_epochs} | "
                        f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                # Periodic eval
                if self.global_step % eval_interval == 0 and len(val_loader) > 0:
                    val_loss = self.evaluate(val_loader)
                    val_ppl = math.exp(min(val_loss, 20))
                    logger.info(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    self.val_losses.append({"step": self.global_step, "loss": val_loss})

                # Periodic save
                if self.global_step % save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

            # End of epoch eval
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append({"epoch": epoch + 1, "loss": avg_epoch_loss})

            val_loss = self.evaluate(val_loader) if len(val_loader) > 0 else avg_epoch_loss
            val_ppl = math.exp(min(val_loss, 20))
            train_ppl = math.exp(min(avg_epoch_loss, 20))

            logger.info(
                f"Epoch {epoch+1}/{max_epochs} complete | "
                f"Train Loss: {avg_epoch_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})"
            )

            # Best model tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
                logger.info(f"  New best model (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs (patience={patience})")
                    break

            # Clean up MPS memory
            mps_empty_cache()

        # Save final model
        self._save_checkpoint("final")
        self._save_training_log()

        total_time = time.time() - start_time
        stats = {
            "total_steps": self.global_step,
            "total_epochs": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": math.exp(min(self.best_val_loss, 20)),
            "total_time_seconds": total_time,
        }
        logger.info(f"Training complete: {stats}")
        return stats

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        ckpt_dir = self.save_dir / name
        self.model.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _save_training_log(self):
        """Save training loss curves."""
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        log_path = self.save_dir / "training_log.json"
        log_path.write_text(json.dumps(log, indent=2))


def pretrain_book(
    book_path: str | Path,
    tokenizer_dir: str | Path,
    save_dir: str | Path,
    model_config: dict | None = None,
    train_config: dict | None = None,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Full pretraining pipeline for a single book.

    Args:
        book_path: Path to cleaned book text.
        tokenizer_dir: Path to saved tokenizer.
        save_dir: Where to save model checkpoints.
        model_config: GPT2Config parameters override.
        train_config: Training hyperparameters override.
        device: Device to train on.
        seed: Random seed.

    Returns:
        Training statistics dict.
    """
    from bookgpt.data.prepare import prepare_book_dataset
    from bookgpt.tokenizer.train_bpe import load_tokenizer

    set_seed(seed)

    # Load tokenizer to get vocab size
    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()

    # Model config
    mc = model_config or {}
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=mc.get("n_layer", 6),
        n_head=mc.get("n_head", 8),
        n_embd=mc.get("n_embd", 256),
        context_length=mc.get("context_length", 512),
        dropout=mc.get("dropout", 0.1),
        bias=mc.get("bias", False),
    )

    logger.info(f"Model config: {config}")
    logger.info(f"Estimated params: {config.num_params():,}")

    # Prepare datasets
    tc = train_config or {}
    train_dataset, val_dataset = prepare_book_dataset(
        book_path=book_path,
        tokenizer_dir=tokenizer_dir,
        context_length=config.context_length,
        train_split=tc.get("train_split", 0.9),
    )

    # Create model
    model = GPT2(config)

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=tc,
        save_dir=save_dir,
        device=device,
    )

    stats = trainer.train()
    return stats
