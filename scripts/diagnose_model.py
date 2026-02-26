#!/usr/bin/env python3
"""
Model Diagnostics: Why do low-parameter models fail at Q&A?

Investigates five dimensions:
1. Attention Pattern Analysis — Are heads learning meaningful relationships?
2. Embedding Space Analysis — Are semantically similar tokens clustered?
3. Token Prediction Analysis — Where exactly does generation break down?
4. Capacity Saturation — Are the weights "full" or is there unused capacity?
5. Book Complexity vs Performance — What book properties predict failure?

Usage:
    python scripts/diagnose_model.py --book-id calculus_made_easy
    python scripts/diagnose_model.py --book-id plane_geometry --no-plots
    python scripts/diagnose_model.py --all --no-plots
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bookgpt.model.gpt2 import GPT2
from bookgpt.tokenizer.train_bpe import load_tokenizer, tokenize_text, decode_tokens
from bookgpt.model.generate import generate_answer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────
# 1. ATTENTION PATTERN ANALYSIS
# ─────────────────────────────────────────────────────

def extract_attention_weights(model, input_ids):
    """Run forward pass with hooks to capture attention weights from all layers."""
    attention_maps = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Re-compute attention weights inside the hook
            x = input[0]
            B, T, C = x.size()
            qkv = module.c_attn(x)
            q, k, v = qkv.split(module.n_embd, dim=2)
            q = q.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
            k = k.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(module.head_dim))
            att = att.masked_fill(module.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            attention_maps.append(att.detach().cpu())
        return hook_fn

    model.eval()
    for i, block in enumerate(model.transformer["h"]):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return attention_maps  # List of (B, n_head, T, T) tensors


def analyze_attention_patterns(attention_maps, token_labels, special_positions):
    """Analyze what attention heads are doing.

    Returns dict with:
    - entropy_per_head: How spread out is attention? (high=diffuse, low=focused)
    - special_token_attention: How much attention goes to context/question/answer tokens
    - locality_score: How much attention is on nearby tokens vs distant ones
    """
    results = {"layers": []}

    for layer_idx, attn in enumerate(attention_maps):
        # attn shape: (1, n_heads, T, T)
        attn = attn[0]  # Remove batch dim: (n_heads, T, T)
        n_heads, T, _ = attn.shape

        layer_info = {"layer": layer_idx, "heads": []}

        for head_idx in range(n_heads):
            head_attn = attn[head_idx]  # (T, T)

            # Entropy: how spread out is attention per position?
            # High entropy = looking everywhere equally (unfocused)
            # Low entropy = focused on specific positions
            eps = 1e-10
            entropy = -(head_attn * (head_attn + eps).log()).sum(dim=-1)  # (T,)
            avg_entropy = entropy.mean().item()
            max_possible_entropy = math.log(T)
            normalized_entropy = avg_entropy / max_possible_entropy if max_possible_entropy > 0 else 0

            # Locality: what fraction of attention is on nearby tokens (within 10 positions)?
            locality_window = 10
            locality_mask = torch.zeros(T, T)
            for i in range(T):
                start = max(0, i - locality_window)
                end = min(T, i + 1)  # causal: can only attend to past
                locality_mask[i, start:end] = 1.0
            local_attention = (head_attn * locality_mask).sum(dim=-1).mean().item()

            # Attention to special token regions
            ctx_attn = 0.0
            q_attn = 0.0
            a_attn = 0.0
            if "context_range" in special_positions and "answer_start" in special_positions:
                ctx_start, ctx_end = special_positions["context_range"]
                q_start, q_end = special_positions["question_range"]
                a_start = special_positions["answer_start"]

                # From answer positions, how much attention goes to context vs question?
                if a_start < T:
                    answer_rows = head_attn[a_start:]  # attention from answer tokens
                    if answer_rows.shape[0] > 0:
                        ctx_attn = answer_rows[:, ctx_start:ctx_end].sum(dim=-1).mean().item()
                        q_attn = answer_rows[:, q_start:q_end].sum(dim=-1).mean().item()
                        a_attn = answer_rows[:, a_start:].sum(dim=-1).mean().item()

            head_info = {
                "head": head_idx,
                "avg_entropy": avg_entropy,
                "normalized_entropy": normalized_entropy,
                "locality_score": local_attention,
                "ctx_attention": ctx_attn,
                "q_attention": q_attn,
                "self_attention": a_attn,
            }
            layer_info["heads"].append(head_info)

        results["layers"].append(layer_info)

    return results


# ─────────────────────────────────────────────────────
# 2. EMBEDDING SPACE ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_embeddings(model, tokenizer):
    """Analyze the token embedding space.

    Checks:
    - Are math-related tokens clustered?
    - What's the effective dimensionality (how many dimensions are actually used)?
    - How spread out are embeddings (are they using the full space)?
    """
    embeddings = model.transformer["wte"].weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    vocab_size, n_embd = embeddings.shape

    # 1. Singular value analysis: effective dimensionality
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    total_variance = (S ** 2).sum()
    cumulative_variance = np.cumsum(S ** 2) / total_variance

    # How many dimensions capture 90% / 95% / 99% of variance?
    dims_90 = int(np.searchsorted(cumulative_variance, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative_variance, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumulative_variance, 0.99)) + 1

    # 2. Embedding norms: are all tokens using similar magnitudes?
    norms = np.linalg.norm(embeddings, axis=1)

    # 3. Cosine similarity between math concept groups
    math_groups = {
        "calculus": ["derivative", "integral", "limit", "differentiate", "function"],
        "algebra": ["equation", "variable", "polynomial", "factor", "solve"],
        "geometry": ["triangle", "circle", "angle", "line", "point"],
        "numbers": ["number", "zero", "one", "two", "three"],
        "operations": ["add", "subtract", "multiply", "divide", "equal"],
    }

    group_similarities = {}
    for group_name, words in math_groups.items():
        token_ids = []
        found_words = []
        for word in words:
            tokens = tokenize_text(tokenizer, word)
            if len(tokens) == 1:  # Only use single-token words
                token_ids.append(tokens[0])
                found_words.append(word)
            else:
                # Use the first token as approximation
                token_ids.append(tokens[0])
                found_words.append(f"{word}[0]")

        if len(token_ids) >= 2:
            group_embs = embeddings[token_ids]
            # Pairwise cosine similarities
            norms_g = np.linalg.norm(group_embs, axis=1, keepdims=True)
            normalized = group_embs / (norms_g + 1e-8)
            sim_matrix = normalized @ normalized.T
            # Average off-diagonal similarity
            n = len(token_ids)
            mask = ~np.eye(n, dtype=bool)
            avg_sim = sim_matrix[mask].mean()
            group_similarities[group_name] = {
                "words": found_words,
                "avg_cosine_similarity": float(avg_sim),
                "token_ids": token_ids,
            }

    # 4. Cross-group similarity (should be lower than within-group)
    all_group_embs = {}
    for group_name, info in group_similarities.items():
        all_group_embs[group_name] = embeddings[info["token_ids"]].mean(axis=0)

    cross_sims = {}
    group_names = list(all_group_embs.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            e1 = all_group_embs[g1]
            e2 = all_group_embs[g2]
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
            cross_sims[f"{g1}-{g2}"] = sim

    # 5. Random baseline: average similarity between random token pairs
    rng = np.random.RandomState(42)
    random_pairs = rng.choice(vocab_size, size=(200, 2), replace=True)
    random_sims = []
    for i, j in random_pairs:
        e1, e2 = embeddings[i], embeddings[j]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        random_sims.append(sim)

    return {
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "singular_values": S.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "dims_for_90pct": dims_90,
        "dims_for_95pct": dims_95,
        "dims_for_99pct": dims_99,
        "embedding_norms": {"mean": float(norms.mean()), "std": float(norms.std()), "min": float(norms.min()), "max": float(norms.max())},
        "group_similarities": group_similarities,
        "cross_group_similarities": cross_sims,
        "random_baseline_similarity": float(np.mean(random_sims)),
    }


# ─────────────────────────────────────────────────────
# 3. TOKEN PREDICTION ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_token_predictions(model, tokenizer, qa_examples, device, n_samples=10):
    """Analyze where exactly the model's predictions go wrong.

    For each Q&A example:
    - What are the model's top-k predictions at each answer position?
    - Where does it first diverge from the correct answer?
    - What's the confidence (probability) at each position?
    - Does it get worse as the answer gets longer (compounding errors)?
    """
    ctx_id = tokenizer.token_to_id("<|context|>")
    q_id = tokenizer.token_to_id("<|question|>")
    a_id = tokenizer.token_to_id("<|answer|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    results = []

    for example in qa_examples[:n_samples]:
        ctx_tokens = tokenize_text(tokenizer, example["context"])
        q_tokens = tokenize_text(tokenizer, example["question"])
        a_tokens = tokenize_text(tokenizer, example["answer"])

        # Build full sequence
        full_tokens = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]

        # Truncate if needed
        max_len = model.config.context_length
        if len(full_tokens) > max_len:
            overflow = len(full_tokens) - max_len
            ctx_tokens = ctx_tokens[:max(1, len(ctx_tokens) - overflow)]
            full_tokens = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]

        if len(full_tokens) > max_len:
            continue

        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        answer_start = len([ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id]) - 1  # Position before first answer token

        model.eval()
        with torch.no_grad():
            logits, _ = model(input_ids)

        # Analyze predictions at each answer token position
        position_analysis = []
        correct_count = 0
        total_answer_tokens = len(a_tokens)

        for i, target_token in enumerate(a_tokens):
            pos = answer_start + i  # Position in sequence that predicts this token
            if pos >= logits.shape[1]:
                break

            token_logits = logits[0, pos]  # (vocab_size,)
            probs = F.softmax(token_logits, dim=-1)

            # Top-5 predictions
            top5_probs, top5_ids = probs.topk(5)
            top5_tokens = [decode_tokens(tokenizer, [tid.item()]) for tid in top5_ids]

            target_prob = probs[target_token].item()
            predicted_token = top5_ids[0].item()
            is_correct = predicted_token == target_token

            if is_correct:
                correct_count += 1

            # Rank of correct token
            sorted_indices = probs.argsort(descending=True)
            rank = (sorted_indices == target_token).nonzero(as_tuple=True)[0].item()

            position_analysis.append({
                "position_in_answer": i,
                "target_token": decode_tokens(tokenizer, [target_token]),
                "target_token_id": target_token,
                "target_prob": target_prob,
                "target_rank": rank,
                "predicted_token": top5_tokens[0],
                "predicted_prob": top5_probs[0].item(),
                "is_correct": is_correct,
                "top5": list(zip(top5_tokens, top5_probs.tolist())),
                "entropy": -(probs * (probs + 1e-10).log()).sum().item(),
            })

        accuracy = correct_count / total_answer_tokens if total_answer_tokens > 0 else 0.0

        # Compute accuracy in first half vs second half of answer
        half = total_answer_tokens // 2
        first_half_correct = sum(1 for p in position_analysis[:half] if p["is_correct"])
        second_half_correct = sum(1 for p in position_analysis[half:] if p["is_correct"])
        first_half_acc = first_half_correct / half if half > 0 else 0
        second_half_acc = second_half_correct / (total_answer_tokens - half) if (total_answer_tokens - half) > 0 else 0

        results.append({
            "question": example["question"][:80],
            "answer": example["answer"][:80],
            "total_answer_tokens": total_answer_tokens,
            "accuracy": accuracy,
            "first_half_accuracy": first_half_acc,
            "second_half_accuracy": second_half_acc,
            "avg_target_prob": np.mean([p["target_prob"] for p in position_analysis]) if position_analysis else 0,
            "avg_target_rank": np.mean([p["target_rank"] for p in position_analysis]) if position_analysis else 0,
            "positions": position_analysis,
        })

    return results


# ─────────────────────────────────────────────────────
# 4. CAPACITY SATURATION ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_capacity(model):
    """Analyze whether the model's parameters are saturated.

    Checks:
    - Weight distribution: are weights near their initialization or have they learned?
    - Dead neurons: what fraction of neurons are effectively zero?
    - Gradient of weights: how spread out are the weight values?
    - Layer-by-layer utilization: are some layers more saturated than others?
    """
    results = {"layers": [], "global": {}}

    all_weights = []
    total_params = 0
    near_zero_params = 0

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().flatten()
        all_weights.extend(w.tolist())
        total_params += len(w)
        near_zero_params += int((np.abs(w) < 0.001).sum())

    all_weights = np.array(all_weights)

    results["global"] = {
        "total_params": total_params,
        "mean": float(all_weights.mean()),
        "std": float(all_weights.std()),
        "min": float(all_weights.min()),
        "max": float(all_weights.max()),
        "near_zero_fraction": near_zero_params / total_params,
        "kurtosis": float((((all_weights - all_weights.mean()) / all_weights.std()) ** 4).mean() - 3) if all_weights.std() > 0 else 0,
    }

    # Kurtosis fix
    centered = all_weights - all_weights.mean()
    if all_weights.std() > 0:
        normalized = centered / all_weights.std()
        results["global"]["kurtosis"] = float((normalized ** 4).mean() - 3)

    # Per-layer analysis
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        w = param.detach().cpu().numpy()
        flat = w.flatten()

        layer_info = {
            "name": name,
            "shape": list(w.shape),
            "n_params": len(flat),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "abs_mean": float(np.abs(flat).mean()),
            "near_zero_fraction": float((np.abs(flat) < 0.001).sum() / len(flat)),
            "max_abs": float(np.abs(flat).max()),
        }

        # For 2D weight matrices, compute rank utilization
        if len(w.shape) == 2:
            try:
                S = np.linalg.svd(w, compute_uv=False)
                total = (S ** 2).sum()
                cumvar = np.cumsum(S ** 2) / total
                effective_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
                max_rank = min(w.shape)
                layer_info["effective_rank_90pct"] = effective_rank_90
                layer_info["max_rank"] = max_rank
                layer_info["rank_utilization"] = effective_rank_90 / max_rank
            except:
                pass

        results["layers"].append(layer_info)

    return results


# ─────────────────────────────────────────────────────
# 5. BOOK COMPLEXITY ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_book_complexity(book_id, tokenizer, manifest_path="data/manifest.json",
                           qa_dir="data/books/qa", raw_dir="data/books/raw"):
    """Analyze book text complexity and Q&A characteristics."""

    # Load book text
    raw_path = Path(raw_dir) / f"{book_id}.txt"
    if not raw_path.exists():
        return None

    text = raw_path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenize_text(tokenizer, text)

    # Token-level stats
    token_counts = Counter(tokens)
    unique_tokens = len(token_counts)
    total_tokens = len(tokens)
    type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0

    # Vocabulary richness: what fraction of the total vocab does this book use?
    vocab_coverage = unique_tokens / tokenizer.get_vocab_size()

    # Math symbol density: count LaTeX-like patterns
    math_indicators = ["$", "\\", "frac", "sqrt", "int", "sum", "lim", "infty", "alpha", "beta", "theta"]
    math_density = sum(text.count(m) for m in math_indicators) / max(1, len(text.split()))

    # Sentence length (rough: split on periods)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # Load Q&A data
    qa_path = Path(qa_dir) / f"{book_id}.jsonl"
    qa_examples = []
    if qa_path.exists():
        with open(qa_path) as f:
            for line in f:
                qa_examples.append(json.loads(line))

    # Q&A stats
    avg_answer_len = np.mean([len(e["answer"].split()) for e in qa_examples]) if qa_examples else 0
    avg_context_len = np.mean([len(e["context"].split()) for e in qa_examples]) if qa_examples else 0
    avg_question_len = np.mean([len(e["question"].split()) for e in qa_examples]) if qa_examples else 0

    return {
        "book_id": book_id,
        "total_chars": len(text),
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": type_token_ratio,
        "vocab_coverage": vocab_coverage,
        "math_density": math_density,
        "avg_sentence_length": float(avg_sentence_len),
        "n_qa_pairs": len(qa_examples),
        "avg_answer_words": float(avg_answer_len),
        "avg_context_words": float(avg_context_len),
        "avg_question_words": float(avg_question_len),
    }


# ─────────────────────────────────────────────────────
# 6. GENERATION BREAKDOWN ANALYSIS
# ─────────────────────────────────────────────────────

def analyze_generation_breakdown(model, tokenizer, qa_examples, device, n_samples=5):
    """Generate answers and analyze token-by-token what goes wrong.

    Compares teacher-forced predictions vs free-running generation.
    """
    results = []

    for example in qa_examples[:n_samples]:
        # Teacher-forced: model sees correct previous tokens
        answer_text, avg_logprob = generate_answer(
            model, tokenizer,
            context=example["context"],
            question=example["question"],
            max_new_tokens=64,
            temperature=0.3,  # Low temp for most likely output
            device=device,
        )

        # Also try with temperature=0 (greedy)
        greedy_text, greedy_logprob = generate_answer(
            model, tokenizer,
            context=example["context"],
            question=example["question"],
            max_new_tokens=64,
            temperature=0.01,  # Near-greedy
            device=device,
        )

        results.append({
            "question": example["question"][:100],
            "expected_answer": example["answer"][:100],
            "generated_t03": answer_text[:100],
            "generated_greedy": greedy_text[:100],
            "logprob_t03": avg_logprob,
            "logprob_greedy": greedy_logprob,
        })

    return results


# ─────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────

def plot_diagnostics(book_id, attention_results, embedding_results, prediction_results,
                     capacity_results, generation_results, output_dir="plots/diagnostics"):
    """Generate diagnostic visualizations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Attention Entropy & Locality Heatmap ──
    n_layers = len(attention_results["layers"])
    n_heads = len(attention_results["layers"][0]["heads"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Entropy heatmap
    entropy_grid = np.zeros((n_layers, n_heads))
    for layer_info in attention_results["layers"]:
        for head_info in layer_info["heads"]:
            entropy_grid[layer_info["layer"], head_info["head"]] = head_info["normalized_entropy"]

    im = axes[0].imshow(entropy_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Attention Entropy\n(1.0 = uniform/unfocused, 0.0 = sharp/focused)")
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    plt.colorbar(im, ax=axes[0])

    # Locality heatmap
    locality_grid = np.zeros((n_layers, n_heads))
    for layer_info in attention_results["layers"]:
        for head_info in layer_info["heads"]:
            locality_grid[layer_info["layer"], head_info["head"]] = head_info["locality_score"]

    im = axes[1].imshow(locality_grid, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].set_title("Locality Score\n(1.0 = only nearby tokens, 0.0 = long-range)")
    axes[1].set_xticks(range(n_heads))
    axes[1].set_yticks(range(n_layers))
    plt.colorbar(im, ax=axes[1])

    # Context vs Question attention from answer tokens
    ctx_grid = np.zeros((n_layers, n_heads))
    q_grid = np.zeros((n_layers, n_heads))
    for layer_info in attention_results["layers"]:
        for head_info in layer_info["heads"]:
            ctx_grid[layer_info["layer"], head_info["head"]] = head_info["ctx_attention"]
            q_grid[layer_info["layer"], head_info["head"]] = head_info["q_attention"]

    # Stacked bar showing ctx vs question attention
    x = np.arange(n_heads)
    width = 0.35
    for layer_idx in range(n_layers):
        axes[2].bar(x + layer_idx * 0.15, ctx_grid[layer_idx], width=0.12, label=f"L{layer_idx} ctx" if layer_idx == 0 else "", alpha=0.6, color=f"C{layer_idx}")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Attention Weight")
    axes[2].set_title("Answer→Context Attention by Layer\n(Higher = model reads context when answering)")
    axes[2].set_xticks(range(n_heads))

    fig.suptitle(f"Attention Analysis: {book_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{book_id}_attention.png", dpi=120)
    plt.close(fig)

    # ── Plot 2: Embedding Space ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Singular value spectrum
    svs = embedding_results["singular_values"][:50]
    axes[0].bar(range(len(svs)), svs, color="steelblue", alpha=0.8)
    dim90_idx = min(embedding_results["dims_for_90pct"] - 1, len(svs) - 1)
    axes[0].axhline(y=svs[dim90_idx], color="red", linestyle="--",
                    label=f"90% variance (dim {embedding_results['dims_for_90pct']})")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title(f"Embedding Singular Values\n(90%: {embedding_results['dims_for_90pct']}, "
                      f"95%: {embedding_results['dims_for_95pct']}, "
                      f"99%: {embedding_results['dims_for_99pct']} of {embedding_results['n_embd']} dims)")
    axes[0].legend()

    # Group similarities
    groups = embedding_results["group_similarities"]
    group_names = list(groups.keys())
    within_sims = [groups[g]["avg_cosine_similarity"] for g in group_names]
    random_baseline = embedding_results["random_baseline_similarity"]

    axes[1].bar(range(len(group_names)), within_sims, color="coral", alpha=0.8)
    axes[1].axhline(y=random_baseline, color="gray", linestyle="--", label=f"Random baseline: {random_baseline:.3f}")
    axes[1].set_xticks(range(len(group_names)))
    axes[1].set_xticklabels(group_names, rotation=30)
    axes[1].set_ylabel("Avg Cosine Similarity")
    axes[1].set_title("Within-Group Token Similarity\n(Higher = better clustering)")
    axes[1].legend()

    # Cross-group similarities
    cross = embedding_results["cross_group_similarities"]
    cross_names = list(cross.keys())
    cross_vals = list(cross.values())

    axes[2].barh(range(len(cross_names)), cross_vals, color="mediumpurple", alpha=0.8)
    axes[2].set_yticks(range(len(cross_names)))
    axes[2].set_yticklabels(cross_names, fontsize=8)
    axes[2].axvline(x=random_baseline, color="gray", linestyle="--", label="Random baseline")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_title("Cross-Group Similarity\n(Lower = better separation)")
    axes[2].legend()

    fig.suptitle(f"Embedding Analysis: {book_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{book_id}_embeddings.png", dpi=120)
    plt.close(fig)

    # ── Plot 3: Token Prediction Accuracy ──
    if prediction_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy per position in answer
        max_pos = max(r["total_answer_tokens"] for r in prediction_results)
        position_correct = [[] for _ in range(max_pos)]
        position_prob = [[] for _ in range(max_pos)]
        position_rank = [[] for _ in range(max_pos)]

        for r in prediction_results:
            for p in r["positions"]:
                pos = p["position_in_answer"]
                if pos < max_pos:
                    position_correct[pos].append(1 if p["is_correct"] else 0)
                    position_prob[pos].append(p["target_prob"])
                    position_rank[pos].append(p["target_rank"])

        valid_positions = [(i, np.mean(pc)) for i, pc in enumerate(position_correct) if pc]
        if valid_positions:
            pos_x, pos_acc = zip(*valid_positions)
            axes[0].plot(pos_x, pos_acc, "b-o", markersize=3, alpha=0.7)
            axes[0].set_xlabel("Position in Answer")
            axes[0].set_ylabel("Accuracy (Top-1 Match)")
            axes[0].set_title("Prediction Accuracy by Position\n(Does it get worse deeper into the answer?)")
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].grid(True, alpha=0.3)

        # Target token probability by position
        valid_probs = [(i, np.mean(pp)) for i, pp in enumerate(position_prob) if pp]
        if valid_probs:
            pos_x, pos_p = zip(*valid_probs)
            axes[1].plot(pos_x, pos_p, "r-o", markersize=3, alpha=0.7)
            axes[1].set_xlabel("Position in Answer")
            axes[1].set_ylabel("P(correct token)")
            axes[1].set_title("Confidence in Correct Token\n(How sure is it about the right answer?)")
            axes[1].grid(True, alpha=0.3)

        # First half vs second half accuracy
        first_halves = [r["first_half_accuracy"] for r in prediction_results]
        second_halves = [r["second_half_accuracy"] for r in prediction_results]
        x = range(len(prediction_results))
        width = 0.35
        axes[2].bar([i - width/2 for i in x], first_halves, width, label="First half", color="steelblue", alpha=0.8)
        axes[2].bar([i + width/2 for i in x], second_halves, width, label="Second half", color="coral", alpha=0.8)
        axes[2].set_xlabel("Sample")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("First Half vs Second Half Accuracy\n(Measures error compounding)")
        axes[2].legend()
        axes[2].set_ylim(0, 1.05)

        fig.suptitle(f"Token Prediction Analysis: {book_id}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"{book_id}_predictions.png", dpi=120)
        plt.close(fig)

    # ── Plot 4: Capacity Saturation ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Weight distribution histogram
    layer_names = [l["name"][:30] for l in capacity_results["layers"]]
    stds = [l["std"] for l in capacity_results["layers"]]
    near_zeros = [l["near_zero_fraction"] for l in capacity_results["layers"]]

    axes[0].barh(range(len(layer_names)), stds, color="steelblue", alpha=0.8)
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=6)
    axes[0].set_xlabel("Weight Std Dev")
    axes[0].set_title("Weight Spread by Layer\n(Low std = not learning much)")
    axes[0].axvline(x=0.02, color="red", linestyle="--", label="Init std (0.02)")
    axes[0].legend()

    axes[1].barh(range(len(layer_names)), near_zeros, color="coral", alpha=0.8)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names, fontsize=6)
    axes[1].set_xlabel("Fraction Near Zero (<0.001)")
    axes[1].set_title("Dead Weight Fraction\n(High = wasted capacity)")

    # Rank utilization for weight matrices
    rank_layers = [l for l in capacity_results["layers"] if "rank_utilization" in l]
    if rank_layers:
        r_names = [l["name"][:30] for l in rank_layers]
        r_utils = [l["rank_utilization"] for l in rank_layers]
        axes[2].barh(range(len(r_names)), r_utils, color="mediumpurple", alpha=0.8)
        axes[2].set_yticks(range(len(r_names)))
        axes[2].set_yticklabels(r_names, fontsize=6)
        axes[2].set_xlabel("Rank Utilization (90% variance)")
        axes[2].set_title("Matrix Rank Utilization\n(1.0 = fully used, low = spare capacity)")
        axes[2].set_xlim(0, 1.05)

    fig.suptitle(f"Capacity Analysis: {book_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{book_id}_capacity.png", dpi=120)
    plt.close(fig)

    print(f"\nDiagnostic plots saved to {output_dir}/")


# ─────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────

def print_report(book_id, attention_results, embedding_results, prediction_results,
                 capacity_results, complexity_results, generation_results):
    """Print a comprehensive text report."""

    print(f"\n{'='*80}")
    print(f"  MODEL DIAGNOSTIC REPORT: {book_id}")
    print(f"{'='*80}")

    # ── Attention ──
    print(f"\n{'─'*80}")
    print("  1. ATTENTION PATTERN ANALYSIS")
    print(f"{'─'*80}")
    print(f"  {'Layer':<8} {'Head':<6} {'Entropy':<10} {'Locality':<10} {'Ctx Attn':<10} {'Q Attn':<10} {'Self Attn':<10}")
    print(f"  {'─'*68}")

    all_entropies = []
    all_ctx_attn = []
    for layer_info in attention_results["layers"]:
        for head_info in layer_info["heads"]:
            e = head_info["normalized_entropy"]
            all_entropies.append(e)
            all_ctx_attn.append(head_info["ctx_attention"])
            print(f"  L{layer_info['layer']:<7} H{head_info['head']:<5} "
                  f"{e:<10.3f} {head_info['locality_score']:<10.3f} "
                  f"{head_info['ctx_attention']:<10.3f} {head_info['q_attention']:<10.3f} "
                  f"{head_info['self_attention']:<10.3f}")

    avg_entropy = np.mean(all_entropies)
    avg_ctx = np.mean(all_ctx_attn)
    print(f"\n  Summary:")
    print(f"    Avg Normalized Entropy: {avg_entropy:.3f} (1.0=uniform/useless, 0.0=sharp/focused)")
    if avg_entropy > 0.85:
        print(f"    ⚠ HIGH ENTROPY: Most heads are attending almost uniformly — not learning specific patterns")
    elif avg_entropy > 0.7:
        print(f"    ⚡ MODERATE ENTROPY: Some focus developing, but still quite diffuse")
    else:
        print(f"    ✓ GOOD ENTROPY: Heads are learning focused attention patterns")

    print(f"    Avg Context Attention from Answer: {avg_ctx:.3f}")
    if avg_ctx < 0.1:
        print(f"    ⚠ LOW CONTEXT ATTENTION: Answer tokens barely look at context — model ignoring input")
    elif avg_ctx < 0.3:
        print(f"    ⚡ MODERATE CONTEXT ATTENTION: Some grounding in context, but weak")
    else:
        print(f"    ✓ GOOD CONTEXT ATTENTION: Model reads context when generating answers")

    # ── Embeddings ──
    print(f"\n{'─'*80}")
    print("  2. EMBEDDING SPACE ANALYSIS")
    print(f"{'─'*80}")
    emb = embedding_results
    print(f"    Embedding dim: {emb['n_embd']}, Vocab size: {emb['vocab_size']}")
    print(f"    Effective dimensions (90% variance): {emb['dims_for_90pct']} / {emb['n_embd']}")
    print(f"    Effective dimensions (95% variance): {emb['dims_for_95pct']} / {emb['n_embd']}")
    print(f"    Effective dimensions (99% variance): {emb['dims_for_99pct']} / {emb['n_embd']}")

    utilization = emb['dims_for_90pct'] / emb['n_embd']
    if utilization < 0.3:
        print(f"    ⚠ LOW UTILIZATION: Only {utilization:.0%} of dimensions carry 90% of info — embedding space underused")
    elif utilization > 0.7:
        print(f"    ⚠ SATURATED: {utilization:.0%} of dimensions needed for 90% variance — no room to grow")
    else:
        print(f"    ✓ MODERATE UTILIZATION: {utilization:.0%} — reasonable use of space")

    print(f"\n    Math concept clustering (within-group cosine similarity):")
    random_bl = emb["random_baseline_similarity"]
    print(f"    {'Group':<15} {'Similarity':<12} {'vs Random':<12} {'Status'}")
    print(f"    {'─'*55}")
    for group, info in emb["group_similarities"].items():
        sim = info["avg_cosine_similarity"]
        delta = sim - random_bl
        status = "✓ Clustered" if delta > 0.05 else "⚠ Not clustered" if delta < 0.01 else "~ Weak"
        print(f"    {group:<15} {sim:<12.4f} {delta:+<12.4f} {status}")
    print(f"    Random baseline: {random_bl:.4f}")

    # ── Token Predictions ──
    print(f"\n{'─'*80}")
    print("  3. TOKEN PREDICTION ANALYSIS")
    print(f"{'─'*80}")
    if prediction_results:
        overall_acc = np.mean([r["accuracy"] for r in prediction_results])
        overall_1st = np.mean([r["first_half_accuracy"] for r in prediction_results])
        overall_2nd = np.mean([r["second_half_accuracy"] for r in prediction_results])
        avg_rank = np.mean([r["avg_target_rank"] for r in prediction_results])
        avg_prob = np.mean([r["avg_target_prob"] for r in prediction_results])

        print(f"    Overall top-1 accuracy (teacher-forced): {overall_acc:.1%}")
        print(f"    First half of answer accuracy:  {overall_1st:.1%}")
        print(f"    Second half of answer accuracy: {overall_2nd:.1%}")
        print(f"    Avg rank of correct token: {avg_rank:.1f}")
        print(f"    Avg probability of correct token: {avg_prob:.3f}")

        if overall_2nd < overall_1st * 0.7:
            print(f"    ⚠ SEVERE ERROR COMPOUNDING: Accuracy drops {((overall_1st - overall_2nd)/overall_1st)*100:.0f}% from first to second half")
        elif overall_2nd < overall_1st * 0.9:
            print(f"    ⚡ MODERATE ERROR COMPOUNDING: Some degradation in longer answers")
        else:
            print(f"    ✓ STABLE: Accuracy holds through answer length")

        if overall_acc < 0.1:
            print(f"    ⚠ VERY LOW ACCURACY: Model barely predicts correct next tokens even with teacher forcing")
        elif overall_acc < 0.3:
            print(f"    ⚡ LOW ACCURACY: Model struggles with token prediction")

        print(f"\n    Per-sample breakdown:")
        print(f"    {'Question':<50} {'Acc':<8} {'1st½':<8} {'2nd½':<8} {'AvgRank'}")
        print(f"    {'─'*82}")
        for r in prediction_results:
            print(f"    {r['question'][:48]:<50} {r['accuracy']:<8.1%} "
                  f"{r['first_half_accuracy']:<8.1%} {r['second_half_accuracy']:<8.1%} "
                  f"{r['avg_target_rank']:<.0f}")

    # ── Capacity ──
    print(f"\n{'─'*80}")
    print("  4. CAPACITY SATURATION ANALYSIS")
    print(f"{'─'*80}")
    cap = capacity_results
    print(f"    Total parameters: {cap['global']['total_params']:,}")
    print(f"    Weight std: {cap['global']['std']:.4f} (init was 0.02)")
    print(f"    Near-zero fraction: {cap['global']['near_zero_fraction']:.1%}")

    if cap['global']['near_zero_fraction'] > 0.3:
        print(f"    ⚠ HIGH DEAD WEIGHTS: {cap['global']['near_zero_fraction']:.0%} of parameters are near zero — wasted capacity")

    print(f"\n    Layer rank utilization (2D weight matrices):")
    print(f"    {'Layer':<35} {'Shape':<18} {'Rank@90%':<10} {'MaxRank':<10} {'Utilization'}")
    print(f"    {'─'*88}")
    rank_layers = [l for l in cap["layers"] if "rank_utilization" in l]
    saturated_count = 0
    for l in rank_layers:
        util = l["rank_utilization"]
        marker = " ⚠ SATURATED" if util > 0.85 else ""
        if util > 0.85:
            saturated_count += 1
        print(f"    {l['name'][:33]:<35} {str(l['shape']):<18} {l['effective_rank_90pct']:<10} "
              f"{l['max_rank']:<10} {util:<.1%}{marker}")

    if saturated_count > len(rank_layers) * 0.5:
        print(f"\n    ⚠ {saturated_count}/{len(rank_layers)} layers are >85% rank saturated")
        print(f"      This means the model needs MORE dimensions (n_embd) to represent what it's learning")

    # ── Book Complexity ──
    if complexity_results:
        print(f"\n{'─'*80}")
        print("  5. BOOK COMPLEXITY")
        print(f"{'─'*80}")
        c = complexity_results
        print(f"    Total tokens: {c['total_tokens']:,}")
        print(f"    Unique tokens: {c['unique_tokens']:,}")
        print(f"    Type-token ratio: {c['type_token_ratio']:.3f} (higher = more diverse vocabulary)")
        print(f"    Vocab coverage: {c['vocab_coverage']:.1%} of 8192 tokens used")
        print(f"    Math symbol density: {c['math_density']:.3f} per word")
        print(f"    Avg sentence length: {c['avg_sentence_length']:.1f} words")
        print(f"    Q&A pairs: {c['n_qa_pairs']}")
        print(f"    Avg answer length: {c['avg_answer_words']:.1f} words")

    # ── Generation Samples ──
    if generation_results:
        print(f"\n{'─'*80}")
        print("  6. GENERATION SAMPLES (what the model actually outputs)")
        print(f"{'─'*80}")
        for i, g in enumerate(generation_results):
            print(f"\n    Sample {i+1}:")
            print(f"    Q: {g['question']}")
            print(f"    Expected: {g['expected_answer']}")
            print(f"    Generated (t=0.3): {g['generated_t03']}")
            print(f"    Generated (greedy): {g['generated_greedy']}")
            print(f"    LogProb: t=0.3: {g['logprob_t03']:.3f}, greedy: {g['logprob_greedy']:.3f}")

    print(f"\n{'='*80}")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model Diagnostics")
    parser.add_argument("--book-id", type=str, help="Book to diagnose")
    parser.add_argument("--all", action="store_true", help="Run on all models (summary only)")
    parser.add_argument("--stage", type=str, default="finetune",
                        choices=["pretrain", "finetune"],
                        help="Which model stage to diagnose (default: finetune)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of Q&A samples to analyze")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--pretrained-dir", type=str, default=None)
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizers/shared")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--version", type=str, default=None,
                        help="Run version (e.g., v1, v2). Controls output directories.")
    args = parser.parse_args()

    if not args.book_id and not args.all:
        print("Error: provide --book-id or --all")
        sys.exit(1)

    # Resolve versioned paths
    import yaml
    from bookgpt.utils.paths import versioned_paths
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    paths = versioned_paths(config, args.version)

    if args.models_dir is None:
        if args.stage == "pretrain":
            args.models_dir = paths["pretrained_dir"]
        else:
            args.models_dir = paths["finetuned_dir"]
    if args.pretrained_dir is None:
        args.pretrained_dir = paths["pretrained_dir"]
    if args.qa_dir is None:
        args.qa_dir = paths["qa_dir"]
    if args.output_dir is None:
        args.output_dir = str(Path(paths["diagnostics_dir"]) / args.stage)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.all:
        # Summary mode: run key metrics on all models
        run_all_summary(args, tokenizer, device)
    else:
        run_single_book(args, tokenizer, device)


def run_single_book(args, tokenizer, device):
    """Full diagnostic on a single book."""
    book_id = args.book_id
    model_path = Path(args.models_dir) / book_id / "best"

    if not model_path.exists():
        print(f"Error: No {args.stage} model at {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = GPT2.from_pretrained(model_path, device=device)
    model.eval()

    # Load Q&A data
    qa_path = Path(args.qa_dir) / f"{book_id}.jsonl"
    qa_examples = []
    if qa_path.exists():
        with open(qa_path) as f:
            for line in f:
                qa_examples.append(json.loads(line))
    print(f"Loaded {len(qa_examples)} Q&A examples")

    # Build a sample input for attention analysis
    if qa_examples:
        ex = qa_examples[0]
        ctx_id = tokenizer.token_to_id("<|context|>")
        q_id = tokenizer.token_to_id("<|question|>")
        a_id = tokenizer.token_to_id("<|answer|>")
        eos_id = tokenizer.token_to_id("<|endoftext|>")

        ctx_tokens = tokenize_text(tokenizer, ex["context"])
        q_tokens = tokenize_text(tokenizer, ex["question"])
        a_tokens = tokenize_text(tokenizer, ex["answer"])

        # Truncate context to fit
        max_len = model.config.context_length
        full = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]
        if len(full) > max_len:
            overflow = len(full) - max_len
            ctx_tokens = ctx_tokens[:max(1, len(ctx_tokens) - overflow)]
            full = [ctx_id] + ctx_tokens + [q_id] + q_tokens + [a_id] + a_tokens + [eos_id]

        input_ids = torch.tensor([full[:max_len]], dtype=torch.long, device=device)
        token_labels = [decode_tokens(tokenizer, [t]) for t in full[:max_len]]

        ctx_end = 1 + len(ctx_tokens)
        q_start = ctx_end  # <|question|> token
        q_end = q_start + 1 + len(q_tokens)
        a_start = q_end + 1  # first answer token position

        special_positions = {
            "context_range": (0, ctx_end),
            "question_range": (q_start, q_end),
            "answer_start": a_start,
        }
    else:
        # Fallback: just use some text
        text = "What is the derivative of x squared?"
        tokens = tokenize_text(tokenizer, text)[:model.config.context_length]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        token_labels = [decode_tokens(tokenizer, [t]) for t in tokens]
        special_positions = {}

    # Run all analyses
    print("\n[1/6] Analyzing attention patterns...")
    attention_maps = extract_attention_weights(model, input_ids)
    attention_results = analyze_attention_patterns(attention_maps, token_labels, special_positions)

    print("[2/6] Analyzing embedding space...")
    embedding_results = analyze_embeddings(model, tokenizer)

    print("[3/6] Analyzing token predictions...")
    prediction_results = analyze_token_predictions(model, tokenizer, qa_examples, device, n_samples=args.n_samples)

    print("[4/6] Analyzing capacity saturation...")
    capacity_results = analyze_capacity(model)

    print("[5/6] Analyzing book complexity...")
    complexity_results = analyze_book_complexity(book_id, tokenizer)

    print("[6/6] Generating sample outputs...")
    generation_results = analyze_generation_breakdown(model, tokenizer, qa_examples, device, n_samples=5)

    # Print report
    print_report(book_id, attention_results, embedding_results, prediction_results,
                 capacity_results, complexity_results, generation_results)

    # Generate plots
    if not args.no_plots:
        plot_diagnostics(book_id, attention_results, embedding_results, prediction_results,
                        capacity_results, generation_results, args.output_dir)

    # Save raw data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "book_id": book_id,
        "attention_summary": {
            "avg_entropy": float(np.mean([h["normalized_entropy"]
                                          for l in attention_results["layers"]
                                          for h in l["heads"]])),
            "avg_ctx_attention": float(np.mean([h["ctx_attention"]
                                                for l in attention_results["layers"]
                                                for h in l["heads"]])),
        },
        "embedding_summary": {
            "dims_for_90pct": embedding_results["dims_for_90pct"],
            "dims_for_95pct": embedding_results["dims_for_95pct"],
            "n_embd": embedding_results["n_embd"],
        },
        "prediction_summary": {
            "accuracy": float(np.mean([r["accuracy"] for r in prediction_results])) if prediction_results else 0,
            "first_half_acc": float(np.mean([r["first_half_accuracy"] for r in prediction_results])) if prediction_results else 0,
            "second_half_acc": float(np.mean([r["second_half_accuracy"] for r in prediction_results])) if prediction_results else 0,
        },
        "capacity_summary": {
            "total_params": capacity_results["global"]["total_params"],
            "near_zero_fraction": capacity_results["global"]["near_zero_fraction"],
            "saturated_layers": sum(1 for l in capacity_results["layers"]
                                   if l.get("rank_utilization", 0) > 0.85),
            "total_weight_layers": sum(1 for l in capacity_results["layers"]
                                       if "rank_utilization" in l),
        },
        "complexity": complexity_results,
    }
    report_path = output_dir / f"{book_id}_report.json"
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nReport data saved to {report_path}")


def run_all_summary(args, tokenizer, device):
    """Run quick summary diagnostics on all models."""
    models_dir = Path(args.models_dir)
    book_dirs = sorted([d for d in models_dir.iterdir() if (d / "best").exists()])

    if not book_dirs:
        print(f"No {args.stage} models found")
        return

    print(f"Found {len(book_dirs)} {args.stage} models\n")
    print(f"{'Book ID':<45} {'Params':<10} {'NearZero%':<10} {'Sat.Layers':<12} {'Emb90%':<8} {'Acc':<8} {'ErrComp'}")
    print(f"{'─'*105}")

    all_results = []

    for book_dir in book_dirs:
        book_id = book_dir.name
        model_path = book_dir / "best"

        try:
            model = GPT2.from_pretrained(model_path, device=device)
            model.eval()

            # Quick capacity check
            cap = analyze_capacity(model)
            saturated = sum(1 for l in cap["layers"] if l.get("rank_utilization", 0) > 0.85)
            total_rank_layers = sum(1 for l in cap["layers"] if "rank_utilization" in l)

            # Quick embedding check
            emb = analyze_embeddings(model, tokenizer)

            # Quick prediction check (3 samples)
            qa_path = Path(args.qa_dir) / f"{book_id}.jsonl"
            qa_examples = []
            if qa_path.exists():
                with open(qa_path) as f:
                    for line in f:
                        qa_examples.append(json.loads(line))

            pred = analyze_token_predictions(model, tokenizer, qa_examples, device, n_samples=3)
            acc = np.mean([r["accuracy"] for r in pred]) if pred else 0
            first_half = np.mean([r["first_half_accuracy"] for r in pred]) if pred else 0
            second_half = np.mean([r["second_half_accuracy"] for r in pred]) if pred else 0
            err_compound = (first_half - second_half) / first_half if first_half > 0 else 0

            print(f"{book_id:<45} {cap['global']['total_params']:<10,} "
                  f"{cap['global']['near_zero_fraction']:<10.1%} "
                  f"{saturated}/{total_rank_layers:<10} "
                  f"{emb['dims_for_90pct']:<8} "
                  f"{acc:<8.1%} "
                  f"{err_compound:<+.0%}")

            all_results.append({
                "book_id": book_id,
                "near_zero": cap["global"]["near_zero_fraction"],
                "saturated_layers": saturated,
                "total_rank_layers": total_rank_layers,
                "emb_dims_90": emb["dims_for_90pct"],
                "accuracy": acc,
                "error_compounding": err_compound,
            })

            # Clean up memory
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        except Exception as e:
            print(f"{book_id:<45} ERROR: {e}")

    # Summary
    if all_results:
        print(f"\n{'─'*105}")
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_sat = np.mean([r["saturated_layers"] / max(1, r["total_rank_layers"]) for r in all_results])
        avg_nz = np.mean([r["near_zero"] for r in all_results])
        print(f"  Average accuracy: {avg_acc:.1%}")
        print(f"  Average saturation: {avg_sat:.1%}")
        print(f"  Average near-zero weights: {avg_nz:.1%}")


if __name__ == "__main__":
    main()
