"""Response quality scoring for DPO preference pair generation.

Scores model outputs on fluency, relevance, and factuality
to create preference rankings without an external reward model.
"""

import logging
import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from bookgpt.model.gpt2 import GPT2
from bookgpt.tokenizer.train_bpe import tokenize_text, decode_tokens

logger = logging.getLogger(__name__)


def score_response(
    model: GPT2,
    tokenizer,
    context: str,
    question: str,
    answer: str,
    device: torch.device | None = None,
    weights: dict | None = None,
) -> dict:
    """Score a generated response on multiple quality dimensions.

    Args:
        model: The GPT-2 model (used for perplexity scoring).
        tokenizer: The tokenizer.
        context: The source passage from the book.
        question: The question asked.
        answer: The generated answer to score.
        device: Device for model inference.
        weights: Optional dict with keys 'fluency', 'relevance', 'factuality', 'length'.

    Returns:
        Dict with individual scores and composite score.
    """
    if device is None:
        device = next(model.parameters()).device

    w = weights or {"fluency": 0.3, "relevance": 0.3, "factuality": 0.25, "length": 0.15}

    fluency = score_fluency(model, tokenizer, context, question, answer, device)
    relevance = score_relevance(answer, question)
    factuality = score_factuality(answer, context)
    length = score_length(answer)

    composite = (
        w["fluency"] * fluency
        + w["relevance"] * relevance
        + w["factuality"] * factuality
        + w["length"] * length
    )

    return {
        "fluency": fluency,
        "relevance": relevance,
        "factuality": factuality,
        "length": length,
        "composite": composite,
    }


@torch.no_grad()
def score_fluency(
    model: GPT2,
    tokenizer,
    context: str,
    question: str,
    answer: str,
    device: torch.device,
) -> float:
    """Score fluency via negative perplexity of the answer tokens.

    Lower perplexity = higher fluency = higher score.
    Returns a score in [0, 1] where 1 = very fluent.
    """
    model.eval()

    ctx_id = tokenizer.token_to_id("<|context|>")
    q_id = tokenizer.token_to_id("<|question|>")
    a_id = tokenizer.token_to_id("<|answer|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    ctx_tokens = tokenize_text(tokenizer, context)
    q_tokens = tokenize_text(tokenizer, question)
    a_tokens = tokenize_text(tokenizer, answer)

    if not a_tokens:
        return 0.0

    # Build the full sequence
    full_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]

    # Truncate context if needed to fit in context_length
    if len(full_ids) > model.config.context_length:
        overflow = len(full_ids) - model.config.context_length
        ctx_tokens = ctx_tokens[: max(1, len(ctx_tokens) - overflow)]
        full_ids = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]
        if len(full_ids) > model.config.context_length:
            full_ids = full_ids[: model.config.context_length]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids)

    # Compute perplexity only on answer tokens
    answer_start = len(full_ids) - len(a_tokens) - 1  # position of first answer token
    log_probs = F.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    n_tokens = 0
    for i in range(answer_start, len(full_ids) - 1):
        next_token = full_ids[i + 1]
        total_logprob += log_probs[0, i, next_token].item()
        n_tokens += 1

    if n_tokens == 0:
        return 0.0

    avg_logprob = total_logprob / n_tokens
    ppl = math.exp(-avg_logprob)

    # Convert PPL to [0, 1] score: PPL=1 -> 1.0, PPL=1000 -> ~0.0
    # Using sigmoid-like mapping: score = 1 / (1 + log(ppl)/log(100))
    score = max(0.0, min(1.0, 1.0 - math.log(max(ppl, 1.0)) / math.log(1000)))

    return score


def score_relevance(answer: str, question: str) -> float:
    """Score relevance of answer to the question via token overlap.

    Returns a score in [0, 1].
    """
    if not answer or not question:
        return 0.0

    # Tokenize into words (simple whitespace split + lowercase)
    answer_words = set(_get_content_words(answer))
    question_words = set(_get_content_words(question))

    if not question_words or not answer_words:
        return 0.0

    # Check if key question words appear in the answer
    overlap = answer_words & question_words
    recall = len(overlap) / len(question_words)

    # Bonus for answering the question type correctly
    q_lower = question.lower()
    bonus = 0.0
    if q_lower.startswith("what is") or q_lower.startswith("define"):
        # Should contain "is" or definitional language
        if " is " in answer.lower() or " are " in answer.lower():
            bonus = 0.15
    elif q_lower.startswith("state"):
        # Should be a substantial statement
        if len(answer.split()) > 5:
            bonus = 0.1

    return min(1.0, recall * 0.7 + bonus + 0.15)  # Base 0.15 for any non-empty answer


def score_factuality(answer: str, context: str) -> float:
    """Score factuality by measuring n-gram overlap with source context.

    Higher overlap with the book content = more likely factual.
    Returns a score in [0, 1].
    """
    if not answer or not context:
        return 0.0

    answer_lower = answer.lower()
    context_lower = context.lower()

    # Unigram overlap
    answer_words = _get_content_words(answer_lower)
    context_words = _get_content_words(context_lower)

    if not answer_words or not context_words:
        return 0.0

    answer_counts = Counter(answer_words)
    context_counts = Counter(context_words)

    # Precision: what fraction of answer words appear in context
    matched = sum(min(answer_counts[w], context_counts[w]) for w in answer_counts)
    precision = matched / sum(answer_counts.values())

    # Bigram overlap for phrase-level factuality
    answer_bigrams = set(zip(answer_words[:-1], answer_words[1:]))
    context_bigrams = set(zip(context_words[:-1], context_words[1:]))

    if answer_bigrams:
        bigram_overlap = len(answer_bigrams & context_bigrams) / len(answer_bigrams)
    else:
        bigram_overlap = 0.0

    # Weighted combination
    score = 0.6 * precision + 0.4 * bigram_overlap

    return min(1.0, score)


def score_length(answer: str) -> float:
    """Score answer length — penalize too short or too long answers.

    Sweet spot: 10-100 words. Returns a score in [0, 1].
    """
    if not answer:
        return 0.0

    n_words = len(answer.split())

    if n_words < 3:
        return 0.1
    elif n_words < 10:
        return 0.3 + (n_words - 3) * 0.1  # 0.3 to 1.0
    elif n_words <= 100:
        return 1.0
    elif n_words <= 200:
        return 1.0 - (n_words - 100) * 0.005  # gradual decay
    else:
        return 0.5  # very long answers get a floor


def _get_content_words(text: str) -> list[str]:
    """Extract content words (non-stopwords, length > 2)."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "not", "only", "own", "same", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "this", "these",
    "those", "which", "who", "whom", "what", "its", "they", "them",
    "their", "our", "you", "your", "him", "his", "she", "her",
}
