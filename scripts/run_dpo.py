#!/usr/bin/env python3
"""Run DPO alignment pipeline for all fine-tuned book models.

For each book:
1. Generate preference pairs (multiple candidates scored and paired)
2. Train with DPO loss
3. Optionally compare pre/post DPO

Usage:
    python scripts/run_dpo.py
    python scripts/run_dpo.py --book-id calculus_made_easy
    python scripts/run_dpo.py --skip-pref-gen   # use existing preference data
    python scripts/run_dpo.py --compare-only     # skip training, just compare
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.model.gpt2 import GPT2
from bookgpt.model.preference_gen import generate_preferences
from bookgpt.model.dpo import dpo_train_book
from bookgpt.model.compare import compare_models
from bookgpt.tokenizer.train_bpe import load_tokenizer
from bookgpt.utils.device import get_device
from bookgpt.utils.paths import versioned_paths, add_version_arg, ensure_dirs

# Logging setup deferred to main() so version-aware log path can be used
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run DPO alignment for book models")
    parser.add_argument("--manifest", type=str, default="data/books/manifest.json")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizers/shared")
    parser.add_argument("--finetuned-dir", type=str, default=None,
                        help="Base directory for fine-tuned models (default: version-aware)")
    parser.add_argument("--dpo-dir", type=str, default=None,
                        help="Base directory for DPO-aligned models (default: version-aware)")
    parser.add_argument("--preferences-dir", type=str, default=None,
                        help="Directory for preference pair data (default: version-aware)")
    parser.add_argument("--qa-dir", type=str, default=None,
                        help="Directory for Q&A data (default: version-aware)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Directory for comparison reports (default: version-aware)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    add_version_arg(parser)
    parser.add_argument("--book-id", type=str, default=None, help="Process specific book only")
    parser.add_argument("--min-chars", type=int, default=5000)
    parser.add_argument("--skip-pref-gen", action="store_true",
                        help="Skip preference generation (use existing)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training, just run comparisons")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison step")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = versioned_paths(config, args.version)
    ensure_dirs(paths)

    # Override dirs if not explicitly set
    if args.finetuned_dir is None:
        args.finetuned_dir = paths["finetuned_dir"]
    if args.dpo_dir is None:
        args.dpo_dir = f"{paths['dpo_dir']}/models"
    if args.preferences_dir is None:
        args.preferences_dir = f"{paths['dpo_dir']}/preferences"
    if args.qa_dir is None:
        args.qa_dir = paths["qa_dir"]
    if args.reports_dir is None:
        args.reports_dir = f"{paths['dpo_dir']}/reports"

    # Setup logging with versioned log path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{paths['logs_dir']}/dpo.log"),
        ],
    )
    logger.info(f"Run version: {paths['version']}")

    device = get_device(force_cpu=args.force_cpu)
    logger.info(f"Using device: {device}")

    # Verify tokenizer
    tokenizer_dir = Path(args.tokenizer_dir)
    if not (tokenizer_dir / "tokenizer.json").exists():
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.book_id:
        manifest = [e for e in manifest if e["book_id"] == args.book_id]
        if not manifest:
            logger.error(f"Book ID '{args.book_id}' not found")
            return

    dpo_config = config.get("dpo", {})
    pref_config = dpo_config.get("preference_gen", {})

    completed = []
    skipped = []

    for i, entry in enumerate(manifest):
        book_id = entry["book_id"]
        char_count = entry.get("token_count", 0)
        finetuned_dir = Path(args.finetuned_dir) / book_id
        dpo_dir = Path(args.dpo_dir) / book_id
        pref_path = Path(args.preferences_dir) / f"{book_id}.jsonl"
        qa_path = Path(args.qa_dir) / f"{book_id}.jsonl"
        report_path = Path(args.reports_dir) / f"{book_id}.json"

        # Skip conditions
        if char_count < args.min_chars:
            skipped.append(book_id)
            continue

        if not (finetuned_dir / "best" / "model.pt").exists():
            logger.info(f"Skipping {book_id}: no fine-tuned model")
            skipped.append(book_id)
            continue

        if not qa_path.exists():
            logger.info(f"Skipping {book_id}: no Q&A data")
            skipped.append(book_id)
            continue

        # Skip if already DPO-trained
        if not args.compare_only and (dpo_dir / "best" / "model.pt").exists():
            logger.info(f"Skipping {book_id}: already DPO-aligned")
            completed.append(book_id)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(manifest)}] DPO: {entry['title']} ({book_id})")
        logger.info(f"{'='*60}")

        # Step 1: Generate preference pairs
        if not args.compare_only:
            if not args.skip_pref_gen or not pref_path.exists():
                logger.info("Generating preference pairs...")
                model = GPT2.from_pretrained(
                    finetuned_dir / "best", device=device
                )
                tokenizer = load_tokenizer(str(tokenizer_dir))

                generate_preferences(
                    model=model,
                    tokenizer=tokenizer,
                    qa_path=str(qa_path),
                    output_path=str(pref_path),
                    device=device,
                    n_candidates=pref_config.get("n_candidates", 4),
                    max_answer_tokens=pref_config.get("max_answer_tokens", 128),
                    max_pairs=pref_config.get("max_pairs", 500),
                    min_score_gap=pref_config.get("min_score_gap", 0.1),
                )

                del model  # Free memory before DPO training
                import gc
                gc.collect()
            else:
                logger.info(f"Using existing preferences: {pref_path}")

            # Step 2: DPO training
            if not pref_path.exists():
                logger.warning(f"No preference data for {book_id}, skipping DPO")
                skipped.append(book_id)
                continue

            logger.info("Running DPO training...")
            stats = dpo_train_book(
                finetuned_dir=str(finetuned_dir),
                tokenizer_dir=str(tokenizer_dir),
                preferences_path=str(pref_path),
                save_dir=str(dpo_dir),
                dpo_config=dpo_config,
                device=device,
                seed=config.get("seed", 42),
            )

            logger.info(f"DPO stats for {book_id}: {stats}")

        # Step 3: Compare (if DPO model exists and comparison requested)
        if not args.no_compare and (dpo_dir / "best" / "model.pt").exists():
            logger.info("Running pre/post DPO comparison...")
            compare_models(
                pre_dpo_dir=str(finetuned_dir),
                post_dpo_dir=str(dpo_dir),
                tokenizer_dir=str(tokenizer_dir),
                qa_path=str(qa_path),
                output_path=str(report_path),
                n_samples=dpo_config.get("compare_samples", 20),
                device=device,
            )

        completed.append(book_id)

    print(f"\n{'='*60}")
    print(f"DPO Pipeline Complete!")
    print(f"  Completed: {len(completed)} models")
    print(f"  Skipped: {len(skipped)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
