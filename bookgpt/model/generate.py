"""Text generation utilities for GPT-2 models."""

import logging
from pathlib import Path

import torch

from bookgpt.model.gpt2 import GPT2
from bookgpt.tokenizer.train_bpe import load_tokenizer, tokenize_text, decode_tokens

logger = logging.getLogger(__name__)


def generate_text(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> str:
    """Generate text from a prompt.

    Args:
        model: The GPT-2 model.
        tokenizer: The tokenizer.
        prompt: Text prompt to start generation.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        device: Device to run on.

    Returns:
        Generated text (including the prompt).
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize prompt
    token_ids = tokenize_text(tokenizer, prompt)

    if not token_ids:
        logger.warning("Empty prompt after tokenization")
        return ""

    # Truncate to fit context
    max_prompt = model.config.context_length - max_new_tokens
    if len(token_ids) > max_prompt:
        token_ids = token_ids[-max_prompt:]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id("<|endoftext|>")

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id,
    )

    # Decode
    generated_ids = output_ids[0].tolist()
    text = decode_tokens(tokenizer, generated_ids)

    return text


def generate_answer(
    model: GPT2,
    tokenizer,
    context: str,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    device: torch.device | None = None,
) -> tuple[str, float]:
    """Generate an answer to a question given context (for Q&A fine-tuned models).

    Args:
        model: The fine-tuned GPT-2 model.
        tokenizer: The tokenizer.
        context: The context passage.
        question: The question to answer.
        max_new_tokens: Maximum tokens in the answer.
        temperature: Sampling temperature.
        device: Device to run on.

    Returns:
        Tuple of (answer_text, avg_log_probability).
    """
    if device is None:
        device = next(model.parameters()).device

    ctx_id = tokenizer.token_to_id("<|context|>")
    q_id = tokenizer.token_to_id("<|question|>")
    a_id = tokenizer.token_to_id("<|answer|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    # Build prompt
    ctx_tokens = tokenize_text(tokenizer, context)
    q_tokens = tokenize_text(tokenizer, question)

    prompt_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id]

    # Truncate context if needed
    max_prompt = model.config.context_length - max_new_tokens
    if len(prompt_ids) > max_prompt:
        overflow = len(prompt_ids) - max_prompt
        ctx_tokens = ctx_tokens[: max(1, len(ctx_tokens) - overflow)]
        prompt_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            eos_token_id=eos_id,
        )

    # Extract answer portion (after the <|answer|> token)
    full_ids = output_ids[0].tolist()
    answer_ids = full_ids[len(prompt_ids) :]

    # Remove EOS if present
    if eos_id in answer_ids:
        answer_ids = answer_ids[: answer_ids.index(eos_id)]

    answer_text = decode_tokens(tokenizer, answer_ids).strip()

    # Compute average log probability of generated tokens
    avg_logprob = _compute_avg_logprob(model, output_ids, len(prompt_ids), device)

    return answer_text, avg_logprob


@torch.no_grad()
def _compute_avg_logprob(
    model: GPT2,
    token_ids: torch.Tensor,
    prompt_len: int,
    device: torch.device,
) -> float:
    """Compute average log probability of generated tokens."""
    import torch.nn.functional as F

    model.eval()

    # Truncate to context length
    ids = token_ids[:, : model.config.context_length]
    logits, _ = model(ids)

    # Get log probs for generated tokens only
    log_probs = F.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    n_tokens = 0

    for i in range(prompt_len, ids.size(1) - 1):
        next_token = ids[0, i + 1]
        total_logprob += log_probs[0, i, next_token].item()
        n_tokens += 1

    return total_logprob / max(n_tokens, 1)


def sample_from_model(
    model_dir: str | Path,
    tokenizer_dir: str | Path,
    prompts: list[str],
    device: torch.device | None = None,
    **gen_kwargs,
) -> list[str]:
    """Load a model and generate samples from multiple prompts.

    Args:
        model_dir: Path to saved model directory.
        tokenizer_dir: Path to saved tokenizer directory.
        prompts: List of text prompts.
        device: Device to use.
        **gen_kwargs: Additional generation kwargs (temperature, top_k, etc.).

    Returns:
        List of generated texts.
    """
    from bookgpt.utils.device import get_device

    if device is None:
        device = get_device()

    model = GPT2.from_pretrained(model_dir, device=device)
    tokenizer = load_tokenizer(tokenizer_dir)

    results = []
    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, device=device, **gen_kwargs)
        results.append(text)
        logger.info(f"Prompt: {prompt[:50]}... -> Generated {len(text)} chars")

    return results
