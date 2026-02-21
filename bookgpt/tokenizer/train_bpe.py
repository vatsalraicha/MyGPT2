"""BPE tokenizer training per book using HuggingFace tokenizers."""

import json
import logging
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|question|>",
    "<|answer|>",
    "<|context|>",
]


def train_bpe_tokenizer(
    text_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on a single book.

    Args:
        text_path: Path to the cleaned book text file.
        output_dir: Directory to save the tokenizer.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for a token to be included.
        special_tokens: List of special tokens to add.

    Returns:
        The trained Tokenizer.
    """
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    logger.info(f"Training BPE tokenizer on {text_path.name} (vocab_size={vocab_size})")

    # Build tokenizer with byte-level BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor to add special handling
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train([str(text_path)], trainer=trainer)

    # Enable padding
    pad_id = tokenizer.token_to_id("<|pad|>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>")

    # Save
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Save metadata
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "source_file": str(text_path),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        f"Tokenizer saved to {output_dir} "
        f"(vocab_size={tokenizer.get_vocab_size()}, "
        f"special_tokens={meta['special_tokens']})"
    )

    return tokenizer


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer.

    Args:
        tokenizer_dir: Directory containing tokenizer.json.

    Returns:
        The loaded Tokenizer.
    """
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    logger.info(f"Loaded tokenizer from {tokenizer_dir} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def tokenize_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Tokenize a text string into token IDs.

    Args:
        tokenizer: The tokenizer to use.
        text: Text to tokenize.

    Returns:
        List of token IDs.
    """
    encoding = tokenizer.encode(text)
    return encoding.ids


def decode_tokens(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    """Decode token IDs back to text.

    Args:
        tokenizer: The tokenizer to use.
        token_ids: List of token IDs.

    Returns:
        Decoded text string.
    """
    return tokenizer.decode(token_ids)
