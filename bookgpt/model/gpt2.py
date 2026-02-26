"""GPT-2 model architecture with learned absolute positional embeddings.

Per-book GPT-2 for training on Apple Silicon (MPS).
v2: 6 layers, 12 heads, 768 embedding dim, 49M params.
"""

import json
import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""

    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    context_length: int = 512
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    def num_params(self, include_embeddings: bool = True) -> int:
        """Estimate total parameter count."""
        # Token embeddings
        n = self.vocab_size * self.n_embd if include_embeddings else 0
        # Position embeddings
        n += self.context_length * self.n_embd
        # Per transformer block
        for _ in range(self.n_layer):
            # Layer norm 1
            n += 2 * self.n_embd
            # Attention QKV + output projection
            qkv = 3 * self.n_embd * self.n_embd
            out = self.n_embd * self.n_embd
            if self.bias:
                qkv += 3 * self.n_embd
                out += self.n_embd
            n += qkv + out
            # Layer norm 2
            n += 2 * self.n_embd
            # FFN: 4x expansion
            ff_up = self.n_embd * (4 * self.n_embd)
            ff_down = (4 * self.n_embd) * self.n_embd
            if self.bias:
                ff_up += 4 * self.n_embd
                ff_down += self.n_embd
            n += ff_up + ff_down
        # Final layer norm
        n += 2 * self.n_embd
        # Output head (tied with embeddings, so don't double count)
        if not self.tie_weights:
            n += self.vocab_size * self.n_embd
        return n


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # QKV projection in one matrix for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation (GPT-2 style)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (GPT-2 style)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 language model with learned absolute positional embeddings."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                "wpe": nn.Embedding(config.context_length, config.n_embd),  # position embeddings
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_weights:
            self.lm_head.weight = self.transformer["wte"].weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GPT-2 model initialized: {n_params:,} parameters")
        logger.info(f"  Learned absolute positional embeddings (ctx={config.context_length})")

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (B, T).
            targets: Target token IDs of shape (B, T) for computing loss.
            loss_mask: Boolean mask of shape (B, T) indicating which positions
                       to include in loss computation. If None, all positions are used.

        Returns:
            Tuple of (logits, loss). Loss is None if targets is None.
        """
        B, T = input_ids.size()
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context length {self.config.context_length}"
        )

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)  # (T,)
        tok_emb = self.transformer["wte"](input_ids)  # (B, T, C)
        pos_emb = self.transformer["wpe"](pos)  # (T, C)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            if loss_mask is not None:
                shift_mask = loss_mask[:, 1:].contiguous()
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction="none",
                )
                loss_per_token = loss_per_token.view(B, T - 1)
                loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (B, T).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Keep only top-k tokens for sampling.
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation when this token is produced.

        Returns:
            Token IDs including the generated tokens, shape (B, T + new_tokens).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = input_ids[:, -self.config.context_length :]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def save_pretrained(self, save_dir: str | Path):
        """Save model weights and config."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "model.pt")

        config_dict = asdict(self.config)
        (save_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str | Path, device: str | torch.device = "cpu") -> "GPT2":
        """Load model from saved weights and config."""
        load_dir = Path(load_dir)

        config_dict = json.loads((load_dir / "config.json").read_text())
        # Handle loading old configs that may have rope_theta/window_size keys
        config_dict.pop("rope_theta", None)
        config_dict.pop("window_size", None)
        config = GPT2Config(**config_dict)

        model = cls(config)
        state_dict = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        logger.info(f"Model loaded from {load_dir}")
        return model
