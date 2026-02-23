"""Compare pre-DPO vs post-DPO model outputs side by side.

Generates comparison reports showing how DPO alignment changed
the model's responses.
"""

import json
import logging
from pathlib import Path

import torch

from bookgpt.model.gpt2 import GPT2
from bookgpt.model.generate import generate_answer
from bookgpt.model.scoring import score_response
from bookgpt.tokenizer.train_bpe import load_tokenizer

logger = logging.getLogger(__name__)


def compare_models(
    pre_dpo_dir: str | Path,
    post_dpo_dir: str | Path,
    tokenizer_dir: str | Path,
    qa_path: str | Path,
    output_path: str | Path | None = None,
    n_samples: int = 20,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Compare fine-tuned (pre-DPO) vs DPO-aligned model outputs.

    Args:
        pre_dpo_dir: Path to fine-tuned model (before DPO).
        post_dpo_dir: Path to DPO-aligned model.
        tokenizer_dir: Path to tokenizer.
        qa_path: Path to Q&A JSONL file for test prompts.
        output_path: Where to save comparison report.
        n_samples: Number of Q&A pairs to compare.
        device: Device for inference.
        seed: Random seed.

    Returns:
        Summary dict with aggregate metrics.
    """
    from bookgpt.utils.device import get_device

    torch.manual_seed(seed)

    if device is None:
        device = get_device()

    tokenizer = load_tokenizer(tokenizer_dir)

    # Load both models
    pre_model_dir = Path(pre_dpo_dir) / "best"
    if not pre_model_dir.exists():
        pre_model_dir = Path(pre_dpo_dir)
    post_model_dir = Path(post_dpo_dir) / "best"
    if not post_model_dir.exists():
        post_model_dir = Path(post_dpo_dir)

    logger.info(f"Loading pre-DPO model from {pre_model_dir}")
    pre_model = GPT2.from_pretrained(pre_model_dir, device=device)

    logger.info(f"Loading post-DPO model from {post_model_dir}")
    post_model = GPT2.from_pretrained(post_model_dir, device=device)

    # Load Q&A examples
    qa_examples = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_examples.append(json.loads(line))

    # Sample evenly from the dataset
    step = max(1, len(qa_examples) // n_samples)
    samples = qa_examples[::step][:n_samples]

    logger.info(f"Comparing {len(samples)} samples")

    comparisons = []
    pre_scores_total = {"fluency": 0, "relevance": 0, "factuality": 0, "composite": 0}
    post_scores_total = {"fluency": 0, "relevance": 0, "factuality": 0, "composite": 0}
    wins = {"pre": 0, "post": 0, "tie": 0}

    for i, example in enumerate(samples):
        context = example.get("context", "")
        question = example.get("question", "")
        reference = example.get("answer", "")

        if not context or not question:
            continue

        # Generate from pre-DPO model
        pre_answer, pre_logprob = generate_answer(
            pre_model, tokenizer, context, question,
            max_new_tokens=128, temperature=0.5, device=device,
        )
        pre_scores = score_response(
            pre_model, tokenizer, context, question, pre_answer, device=device,
        )

        # Generate from post-DPO model
        post_answer, post_logprob = generate_answer(
            post_model, tokenizer, context, question,
            max_new_tokens=128, temperature=0.5, device=device,
        )
        post_scores = score_response(
            post_model, tokenizer, context, question, post_answer, device=device,
        )

        # Track scores
        for key in pre_scores_total:
            pre_scores_total[key] += pre_scores.get(key, 0)
            post_scores_total[key] += post_scores.get(key, 0)

        # Who wins?
        if post_scores["composite"] > pre_scores["composite"] + 0.02:
            wins["post"] += 1
            winner = "post_dpo"
        elif pre_scores["composite"] > post_scores["composite"] + 0.02:
            wins["pre"] += 1
            winner = "pre_dpo"
        else:
            wins["tie"] += 1
            winner = "tie"

        comparisons.append({
            "question": question,
            "reference": reference[:200],
            "pre_dpo": {
                "answer": pre_answer,
                "scores": pre_scores,
            },
            "post_dpo": {
                "answer": post_answer,
                "scores": post_scores,
            },
            "winner": winner,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Compared {i+1}/{len(samples)} samples")

    n = len(comparisons) or 1
    summary = {
        "n_samples": len(comparisons),
        "wins": wins,
        "avg_scores": {
            "pre_dpo": {k: v / n for k, v in pre_scores_total.items()},
            "post_dpo": {k: v / n for k, v in post_scores_total.items()},
        },
        "improvement": {
            k: (post_scores_total[k] - pre_scores_total[k]) / n
            for k in pre_scores_total
        },
    }

    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"summary": summary, "comparisons": comparisons}
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"Comparison report saved to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  DPO COMPARISON REPORT ({len(comparisons)} samples)")
    print(f"{'='*70}")
    print(f"  Wins:  Post-DPO={wins['post']}  |  Pre-DPO={wins['pre']}  |  Tie={wins['tie']}")
    print(f"")
    print(f"  {'Metric':<15} {'Pre-DPO':>10} {'Post-DPO':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    for key in ["fluency", "relevance", "factuality", "composite"]:
        pre = summary["avg_scores"]["pre_dpo"][key]
        post = summary["avg_scores"]["post_dpo"][key]
        delta = post - pre
        print(f"  {key:<15} {pre:>10.4f} {post:>10.4f} {delta:>+10.4f}")
    print(f"{'='*70}")

    return summary
