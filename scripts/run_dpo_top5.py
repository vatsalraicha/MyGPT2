#!/usr/bin/env python3
"""Run DPO on top 5 books, test each, and run post-DPO diagnostics.

Diagnostics are saved to plots/diagnostics_post_dpo/ to avoid overwriting pre-DPO results.

Usage:
    python scripts/run_dpo_top5.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path


TOP5_BOOKS = [
    "arxiv_2110_14321v1",
    "arxiv_2507_15908v2",
    "introduction_to_infinitesimal_analysis",
    "arxiv_1010_4298v7",
    "arxiv_1805_06560v1",
]

TEST_QUESTIONS = {
    "arxiv_2110_14321v1": [
        "What is a manifold?",
        "Define a topological space.",
        "What is a differential form?",
    ],
    "arxiv_2507_15908v2": [
        "What is an algorithm?",
        "Define complexity.",
        "What is optimization?",
    ],
    "introduction_to_infinitesimal_analysis": [
        "What is a limit?",
        "Define continuity.",
        "What is a derivative?",
    ],
    "arxiv_1010_4298v7": [
        "What is a group?",
        "Define a ring.",
        "What is a homomorphism?",
    ],
    "arxiv_1805_06560v1": [
        "What is a matrix?",
        "Define eigenvalue.",
        "What is a linear transformation?",
    ],
}


def run_dpo(book_id: str) -> bool:
    """Run DPO for a single book."""
    print(f"\n{'='*60}")
    print(f"  DPO: {book_id}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "scripts/run_dpo.py",
        "--book-id", book_id,
    ]

    result = subprocess.run(cmd, capture_output=False, timeout=7200)
    return result.returncode == 0


def test_book(book_id: str, model_type: str, models_dir: str) -> list[dict]:
    """Test a book model with sample questions and return results."""
    results = []
    questions = TEST_QUESTIONS.get(book_id, ["What is the main topic?", "Define the key concept.", "Explain the theorem."])

    for question in questions:
        cmd = [
            sys.executable, "-c",
            f"""
import torch
from bookgpt.model.gpt2 import GPT2
from bookgpt.model.generate import generate_answer
from bookgpt.tokenizer.train_bpe import load_tokenizer
from bookgpt.utils.device import get_device

device = get_device()
tokenizer = load_tokenizer('data/tokenizers/shared')
model = GPT2.from_pretrained('{models_dir}/{book_id}/best', device=device)
model.eval()

# Use first QA context as prompt context
import json
with open('data/books/qa/{book_id}.jsonl') as f:
    ex = json.loads(f.readline())

answer, logprob = generate_answer(
    model=model, tokenizer=tokenizer,
    context=ex.get('context', ''),
    question='''{question}''',
    max_new_tokens=64, temperature=0.3, device=device
)
print(f"Q: {question}")
print(f"A: {{answer}}")
print(f"LogProb: {{logprob:.3f}}")
"""
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = result.stdout.strip()
            print(f"  [{model_type}] {output}")
            results.append({"question": question, "output": output, "model_type": model_type})
        except Exception as e:
            print(f"  [{model_type}] Error: {e}")
            results.append({"question": question, "output": f"Error: {e}", "model_type": model_type})

    return results


def run_diagnostics(book_id: str, models_dir: str, output_dir: str) -> bool:
    """Run diagnostics for a book, saving to custom output dir."""
    cmd = [
        sys.executable, "scripts/diagnose_model.py",
        "--book-id", book_id,
        "--models-dir", models_dir,
        "--output-dir", output_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0


def main():
    start_all = time.time()

    post_dpo_diag_dir = "plots/diagnostics_post_dpo_v1"
    Path(post_dpo_diag_dir).mkdir(parents=True, exist_ok=True)

    all_test_results = {}

    for i, book_id in enumerate(TOP5_BOOKS):
        book_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/5] BOOK: {book_id}")
        print(f"{'#'*60}")

        # ── Step 1: DPO ──
        print(f"\n--- Step 1: Running DPO ---")
        dpo_success = run_dpo(book_id)
        if not dpo_success:
            print(f"  ✗ DPO failed for {book_id}, skipping tests and diagnostics")
            continue
        print(f"  ✓ DPO complete")

        # ── Step 2: Test pre-DPO (finetuned) vs post-DPO ──
        print(f"\n--- Step 2: Testing pre-DPO vs post-DPO ---")

        print(f"\n  Pre-DPO (finetuned):")
        pre_results = test_book(book_id, "pre-dpo", "data/models/finetuned")

        dpo_model_dir = "data/dpo/models"
        if (Path(dpo_model_dir) / book_id / "best").exists():
            print(f"\n  Post-DPO:")
            post_results = test_book(book_id, "post-dpo", dpo_model_dir)
        else:
            print(f"  ⚠ No DPO model found, skipping post-DPO test")
            post_results = []

        all_test_results[book_id] = {
            "pre_dpo": pre_results,
            "post_dpo": post_results,
        }

        # ── Step 3: Post-DPO diagnostics ──
        print(f"\n--- Step 3: Post-DPO diagnostics ---")
        if (Path(dpo_model_dir) / book_id / "best").exists():
            diag_success = run_diagnostics(book_id, dpo_model_dir, post_dpo_diag_dir)
            if diag_success:
                print(f"  ✓ Diagnostics saved to {post_dpo_diag_dir}/")
            else:
                print(f"  ✗ Diagnostics failed")
        else:
            print(f"  ⚠ Skipped (no DPO model)")

        elapsed = time.time() - book_start
        print(f"\n  Book total time: {elapsed/60:.1f} min")

    # ── Save all test results ──
    results_path = Path(post_dpo_diag_dir) / "dpo_test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_test_results, f, indent=2, ensure_ascii=False)
    print(f"\nTest results saved to {results_path}")

    total_time = time.time() - start_all
    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  Pre-DPO diagnostics: plots/diagnostics/")
    print(f"  Post-DPO diagnostics: {post_dpo_diag_dir}/")
    print(f"  Test comparisons: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
