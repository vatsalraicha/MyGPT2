"""Direct Preference Optimization (DPO) training loop.

Implements DPO loss from "Direct Preference Optimization: Your Language Model
is Secretly a Reward Model" (Rafailov et al., 2023).

L_DPO = -log sigma(beta * (log pi(chosen|x)/pi_ref(chosen|x)
                            - log pi(rejected|x)/pi_ref(rejected|x)))

No separate reward model needed — the reference model is the frozen
fine-tuned checkpoint.
"""

import copy
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from bookgpt.model.gpt2 import GPT2
from bookgpt.tokenizer.train_bpe import load_tokenizer, tokenize_text
from bookgpt.utils.device import get_device, set_seed, mps_empty_cache

logger = logging.getLogger("bookgpt.trainer.dpo")


class PreferencePairDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) for DPO training.

    Each sample contains token IDs for the prompt + chosen response
    and prompt + rejected response.
    """

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        context_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.samples = []

        ctx_id = tokenizer.token_to_id("<|context|>")
        q_id = tokenizer.token_to_id("<|question|>")
        a_id = tokenizer.token_to_id("<|answer|>")
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        pad_id = tokenizer.token_to_id("<|pad|>")

        for pair in pairs:
            context = pair.get("context", "")
            question = pair.get("question", "")
            chosen_answer = pair["chosen"]["answer"]
            rejected_answer = pair["rejected"]["answer"]

            ctx_tokens = tokenize_text(tokenizer, context)
            q_tokens = tokenize_text(tokenizer, question)
            chosen_tokens = tokenize_text(tokenizer, chosen_answer)
            rejected_tokens = tokenize_text(tokenizer, rejected_answer)

            # Build prompt (shared prefix)
            prompt_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id]

            # Build chosen and rejected sequences
            chosen_ids = prompt_ids + chosen_tokens + [eos_id]
            rejected_ids = prompt_ids + rejected_tokens + [eos_id]

            # Truncate if needed (truncate context first)
            chosen_ids = self._truncate(chosen_ids, ctx_tokens, q_tokens,
                                         chosen_tokens, ctx_id, q_id, a_id, eos_id)
            rejected_ids = self._truncate(rejected_ids, ctx_tokens, q_tokens,
                                           rejected_tokens, ctx_id, q_id, a_id, eos_id)

            prompt_len = len(prompt_ids)

            # Pad both to context_length
            chosen_padded = self._pad(chosen_ids, pad_id)
            rejected_padded = self._pad(rejected_ids, pad_id)

            self.samples.append({
                "chosen_ids": torch.tensor(chosen_padded, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_padded, dtype=torch.long),
                "chosen_len": len(chosen_ids),
                "rejected_len": len(rejected_ids),
                "prompt_len": min(prompt_len, self.context_length),
            })

        logger.info(f"PreferencePairDataset: {len(self.samples)} pairs prepared")

    def _truncate(self, full_ids, ctx_tokens, q_tokens, answer_tokens,
                  ctx_id, q_id, a_id, eos_id):
        """Truncate context to fit within context_length."""
        if len(full_ids) <= self.context_length:
            return full_ids
        overflow = len(full_ids) - self.context_length
        ctx_tokens = ctx_tokens[: max(1, len(ctx_tokens) - overflow)]
        full_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + answer_tokens + [eos_id]
        return full_ids[: self.context_length]

    def _pad(self, ids, pad_id):
        """Pad to context_length."""
        pad_len = self.context_length - len(ids)
        return ids + [pad_id] * max(0, pad_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DPOTrainer:
    """DPO training loop.

    Takes a fine-tuned model as the policy, creates a frozen copy as
    the reference model, and trains on preference pairs.
    """

    def __init__(
        self,
        model: GPT2,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: dict,
        save_dir: str | Path,
        device: torch.device | None = None,
    ):
        self.model = model  # Policy model (will be trained)
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()

        self.model.to(self.device)

        # Create frozen reference model (deep copy of initial policy)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        lr = self.config.get("learning_rate", 1e-5)
        weight_decay = self.config.get("weight_decay", 0.01)
        betas = tuple(self.config.get("betas", [0.9, 0.95]))

        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def train(self) -> dict:
        """Run the DPO training loop."""
        max_epochs = self.config.get("max_epochs", 5)
        batch_size = self.config.get("batch_size", 8)
        beta = self.config.get("beta", 0.1)
        patience = self.config.get("patience", 3)
        grad_clip = self.config.get("grad_clip", 1.0)

        n_train = len(self.train_dataset)
        if n_train == 0:
            logger.warning("Empty training dataset — skipping DPO")
            return {"total_steps": 0, "total_epochs": 0, "best_val_loss": float("inf")}

        effective_batch = min(batch_size, n_train)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch,
            shuffle=True,
            drop_last=n_train > effective_batch,
            num_workers=0,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=min(batch_size, max(1, len(self.val_dataset))),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        logger.info(
            f"DPO training: {max_epochs} epochs, {len(train_loader)} batches/epoch, "
            f"beta={beta}, batch_size={effective_batch}"
        )

        self.model.train()
        start_time = time.time()

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_chosen_reward = 0.0
            epoch_rejected_reward = 0.0
            n_batches = 0

            for batch in train_loader:
                chosen_ids = batch["chosen_ids"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)
                chosen_len = batch["chosen_len"]
                rejected_len = batch["rejected_len"]
                prompt_len = batch["prompt_len"]

                # Compute DPO loss
                loss, metrics = self._dpo_loss(
                    chosen_ids, rejected_ids,
                    chosen_len, rejected_len,
                    prompt_len, beta,
                )

                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_chosen_reward += metrics["chosen_reward"]
                epoch_rejected_reward += metrics["rejected_reward"]
                n_batches += 1
                self.global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_chosen_r = epoch_chosen_reward / max(n_batches, 1)
            avg_rejected_r = epoch_rejected_reward / max(n_batches, 1)
            reward_margin = avg_chosen_r - avg_rejected_r

            self.train_losses.append({"epoch": epoch + 1, "loss": avg_loss})

            # Validation
            val_loss, val_metrics = self._evaluate(val_loader, beta)
            self.val_losses.append({"step": self.global_step, "loss": val_loss})

            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"DPO Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Reward margin: {reward_margin:.4f} "
                f"(chosen={avg_chosen_r:.4f}, rejected={avg_rejected_r:.4f})"
            )

            # Best model tracking
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
                logger.info(f"  New best model (val_loss={val_loss:.4f}, improved by {improvement:.4f})")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement (patience={self.patience_counter}/{patience})"
                )
                if patience > 0 and self.patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break

            mps_empty_cache()

        # Save final
        self._save_checkpoint("final")
        self._save_training_log()

        total_time = time.time() - start_time
        stats = {
            "total_steps": self.global_step,
            "total_epochs": epoch + 1,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time,
        }
        logger.info(f"DPO training complete: {stats}")
        return stats

    def _dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_len: torch.Tensor,
        rejected_len: torch.Tensor,
        prompt_len: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the DPO loss.

        L = -log sigma(beta * (log_ratio_chosen - log_ratio_rejected))

        where log_ratio = log pi(y|x) - log pi_ref(y|x)
        """
        # Get log probs from policy and reference for chosen
        policy_chosen_logp = self._get_sequence_logprob(
            self.model, chosen_ids, chosen_len, prompt_len
        )
        ref_chosen_logp = self._get_sequence_logprob(
            self.ref_model, chosen_ids, chosen_len, prompt_len
        )

        # Get log probs from policy and reference for rejected
        policy_rejected_logp = self._get_sequence_logprob(
            self.model, rejected_ids, rejected_len, prompt_len
        )
        ref_rejected_logp = self._get_sequence_logprob(
            self.ref_model, rejected_ids, rejected_len, prompt_len
        )

        # DPO loss
        chosen_reward = policy_chosen_logp - ref_chosen_logp
        rejected_reward = policy_rejected_logp - ref_rejected_logp

        logits = beta * (chosen_reward - rejected_reward)
        loss = -F.logsigmoid(logits).mean()

        metrics = {
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
        }

        return loss, metrics

    def _get_sequence_logprob(
        self,
        model: GPT2,
        input_ids: torch.Tensor,
        seq_len: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average log probability of response tokens (after prompt).

        Args:
            model: The model to compute log probs with.
            input_ids: Token IDs of shape (B, T).
            seq_len: Actual sequence lengths (before padding) for each batch item.
            prompt_len: Length of the prompt portion for each batch item.

        Returns:
            Tensor of shape (B,) with average log probs per sequence.
        """
        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        B, T, V = log_probs.shape

        # Gather log probs for actual next tokens
        # shift: logits[:, i] predicts token at position i+1
        shift_logprobs = log_probs[:, :-1, :]  # (B, T-1, V)
        shift_targets = input_ids[:, 1:]  # (B, T-1)

        # Get log prob of each actual next token
        token_logprobs = shift_logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Mask: only count response tokens (after prompt, before padding)
        mask = torch.zeros_like(token_logprobs)
        for b in range(B):
            p_len = prompt_len[b].item()
            s_len = seq_len[b].item()
            # Response tokens are at positions prompt_len to seq_len-1
            # In shifted space, that's positions (prompt_len-1) to (seq_len-2)
            start = max(0, p_len - 1)
            end = min(T - 1, s_len - 1)
            if start < end:
                mask[b, start:end] = 1.0

        # Average log prob per sequence
        masked_logprobs = (token_logprobs * mask).sum(dim=1)
        n_tokens = mask.sum(dim=1).clamp(min=1)
        avg_logprob = masked_logprobs / n_tokens

        return avg_logprob

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader, beta: float) -> tuple[float, dict]:
        """Evaluate DPO loss on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_chosen_r = 0.0
        total_rejected_r = 0.0
        n_batches = 0

        for batch in val_loader:
            chosen_ids = batch["chosen_ids"].to(self.device)
            rejected_ids = batch["rejected_ids"].to(self.device)
            chosen_len = batch["chosen_len"]
            rejected_len = batch["rejected_len"]
            prompt_len = batch["prompt_len"]

            loss, metrics = self._dpo_loss(
                chosen_ids, rejected_ids,
                chosen_len, rejected_len,
                prompt_len, beta,
            )

            total_loss += loss.item()
            total_chosen_r += metrics["chosen_reward"]
            total_rejected_r += metrics["rejected_reward"]
            n_batches += 1

        self.model.train()

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, {
            "chosen_reward": total_chosen_r / max(n_batches, 1),
            "rejected_reward": total_rejected_r / max(n_batches, 1),
        }

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


def dpo_train_book(
    finetuned_dir: str | Path,
    tokenizer_dir: str | Path,
    preferences_path: str | Path,
    save_dir: str | Path,
    dpo_config: dict | None = None,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Run DPO training for a single book.

    Args:
        finetuned_dir: Path to fine-tuned model (policy + reference).
        tokenizer_dir: Path to tokenizer.
        preferences_path: Path to preference pairs JSONL.
        save_dir: Where to save DPO-aligned model.
        dpo_config: DPO hyperparameters.
        device: Device to train on.
        seed: Random seed.

    Returns:
        Training statistics dict.
    """
    set_seed(seed)

    if device is None:
        device = get_device()

    # Load fine-tuned model as starting policy
    finetuned_dir = Path(finetuned_dir)
    model_dir = finetuned_dir / "best"
    if not model_dir.exists():
        model_dir = finetuned_dir / "final"
    if not model_dir.exists():
        model_dir = finetuned_dir

    logger.info(f"Loading fine-tuned model from {model_dir}")
    model = GPT2.from_pretrained(model_dir, device=device)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)

    # Load preference pairs
    preferences_path = Path(preferences_path)
    pairs = []
    with open(preferences_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    logger.info(f"Loaded {len(pairs)} preference pairs from {preferences_path}")

    if not pairs:
        logger.error("No preference pairs found!")
        return {"error": "no_data"}

    # Create dataset
    full_dataset = PreferencePairDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
    )

    # Train/val split
    n_train = int(len(full_dataset) * 0.9)
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"DPO Train: {n_train} pairs, Val: {n_val} pairs")

    # Config
    dc = dpo_config or {}
    config = {
        "batch_size": dc.get("batch_size", 8),
        "learning_rate": dc.get("learning_rate", 1e-5),
        "weight_decay": dc.get("weight_decay", 0.01),
        "betas": dc.get("betas", [0.9, 0.95]),
        "max_epochs": dc.get("max_epochs", 5),
        "beta": dc.get("beta", 0.1),
        "patience": dc.get("patience", 3),
        "grad_clip": dc.get("grad_clip", 1.0),
    }

    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=save_dir,
        device=device,
    )

    return trainer.train()
