"""Generate preference pairs for DPO training.

For each Q&A prompt, generates multiple candidate answers at varying
temperatures, scores them, and pairs the best vs worst as (chosen, rejected).
"""

import json
import logging
import random
from pathlib import Path

import torch

from bookgpt.model.gpt2 import GPT2
from bookgpt.model.generate import generate_answer
from bookgpt.model.scoring import score_response
from bookgpt.tokenizer.train_bpe import load_tokenizer

logger = logging.getLogger(__name__)


def generate_preferences(
    model: GPT2,
    tokenizer,
    qa_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    n_candidates: int = 4,
    temperatures: list[float] | None = None,
    max_answer_tokens: int = 128,
    max_pairs: int = 1000,
    min_score_gap: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Generate preference pairs from a Q&A dataset.

    For each Q&A example:
    1. Generate n_candidates answers at different temperatures
    2. Score each candidate
    3. Pair the highest-scored with the lowest-scored (if gap > min_score_gap)

    Args:
        model: Fine-tuned GPT-2 model.
        tokenizer: The tokenizer.
        qa_path: Path to Q&A JSONL file.
        output_path: Where to save preference pairs.
        device: Device for inference.
        n_candidates: Number of candidate answers per prompt.
        temperatures: List of temperatures to sample at.
        max_answer_tokens: Max tokens per generated answer.
        max_pairs: Maximum preference pairs to generate.
        min_score_gap: Minimum composite score difference to form a pair.
        seed: Random seed.

    Returns:
        List of preference pair dicts.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 1.0]

    # Ensure we have enough temperatures for n_candidates
    while len(temperatures) < n_candidates:
        temperatures.append(random.uniform(0.3, 1.2))

    # Load Q&A examples
    qa_path = Path(qa_path)
    qa_examples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_examples.append(json.loads(line))

    logger.info(f"Loaded {len(qa_examples)} Q&A examples from {qa_path}")

    # Shuffle and limit
    random.shuffle(qa_examples)
    # Process more than max_pairs since not all will produce valid pairs
    examples_to_process = min(len(qa_examples), max_pairs * 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    pairs = []

    for i, example in enumerate(qa_examples[:examples_to_process]):
        if len(pairs) >= max_pairs:
            break

        context = example.get("context", "")
        question = example.get("question", "")
        reference_answer = example.get("answer", "")

        if not context or not question:
            continue

        # Generate candidates at different temperatures
        candidates = []
        for j in range(n_candidates):
            temp = temperatures[j % len(temperatures)]
            try:
                answer_text, avg_logprob = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    question=question,
                    max_new_tokens=max_answer_tokens,
                    temperature=temp,
                    device=device,
                )

                if not answer_text.strip():
                    continue

                # Score the response
                scores = score_response(
                    model=model,
                    tokenizer=tokenizer,
                    context=context,
                    question=question,
                    answer=answer_text,
                    device=device,
                )

                candidates.append({
                    "answer": answer_text,
                    "temperature": temp,
                    "avg_logprob": avg_logprob,
                    "scores": scores,
                })
            except Exception as e:
                logger.debug(f"Error generating candidate {j} for example {i}: {e}")
                continue

        if len(candidates) < 2:
            continue

        # Sort by composite score
        candidates.sort(key=lambda c: c["scores"]["composite"], reverse=True)

        best = candidates[0]
        worst = candidates[-1]

        score_gap = best["scores"]["composite"] - worst["scores"]["composite"]

        if score_gap < min_score_gap:
            continue

        pair = {
            "context": context,
            "question": question,
            "reference_answer": reference_answer,
            "chosen": {
                "answer": best["answer"],
                "temperature": best["temperature"],
                "scores": best["scores"],
            },
            "rejected": {
                "answer": worst["answer"],
                "temperature": worst["temperature"],
                "scores": worst["scores"],
            },
            "score_gap": score_gap,
        }

        pairs.append(pair)

        if (i + 1) % 50 == 0:
            logger.info(
                f"  Processed {i+1}/{examples_to_process} examples, "
                f"{len(pairs)} pairs generated"
            )

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated {len(pairs)} preference pairs -> {output_path} "
        f"(from {examples_to_process} examples)"
    )

    return pairs


def load_preferences(path: str | Path) -> list[dict]:
    """Load preference pairs from a JSONL file."""
    path = Path(path)
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs
