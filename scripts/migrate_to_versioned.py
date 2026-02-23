#!/usr/bin/env python3
"""One-time migration: move existing unversioned outputs into v1/ structure.

Run this ONCE before using --version flags on other scripts.

What it does:
  data/models/pretrained/     -> data/models/v1/pretrained/
  data/models/finetuned/      -> data/models/v1/finetuned/
  data/books/qa/*.jsonl        -> data/books/qa/v1/*.jsonl
  logs/pretrain.log            -> logs/v1/pretrain.log
  logs/finetune.log            -> logs/v1/finetune.log
  logs/router.log              -> logs/v1/router.log
  logs/dpo.log                 -> logs/v1/dpo.log
  plots/*.png                  -> plots/v1/*.png
  plots/diagnostics/           -> plots/v1/diagnostics/

What it preserves (shared, not moved):
  data/books/raw/
  data/books/manifest.json
  data/books/tokenized/
  data/tokenizers/

Usage:
    python scripts/migrate_to_versioned.py          # dry run (preview)
    python scripts/migrate_to_versioned.py --apply  # actually move files
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Migrate to versioned directory layout")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform the migration (default: dry run)")
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY RUN" if dry_run else "APPLYING"

    print(f"\n{'='*60}")
    print(f"  Migration to Versioned Layout ({mode})")
    print(f"{'='*60}\n")

    moves = []

    # 1. Models
    for subdir in ["pretrained", "finetuned"]:
        src = Path(f"data/models/{subdir}")
        dst = Path(f"data/models/v1/{subdir}")
        if src.exists() and src.is_dir() and not dst.exists():
            # Make sure src is not already inside a version dir
            if "v1" not in str(src):
                moves.append((src, dst, "dir"))

    # 2. Q&A data (move JSONL files, not the directory itself)
    qa_src = Path("data/books/qa")
    qa_dst = Path("data/books/qa/v1")
    if qa_src.exists() and qa_src.is_dir():
        jsonl_files = list(qa_src.glob("*.jsonl"))
        if jsonl_files and not qa_dst.exists():
            moves.append((qa_src, qa_dst, "qa_special"))

    # 3. Log files
    log_files = ["pretrain.log", "finetune.log", "router.log", "dpo.log"]
    for lf in log_files:
        src = Path(f"logs/{lf}")
        dst = Path(f"logs/v1/{lf}")
        if src.exists() and not dst.exists():
            moves.append((src, dst, "file"))

    # 4. Plots
    plots_src = Path("plots")
    plots_dst = Path("plots/v1")
    if plots_src.exists() and plots_src.is_dir():
        # Check if there are PNG files directly in plots/ (not in a version subdir)
        png_files = list(plots_src.glob("*.png"))
        diag_dir = plots_src / "diagnostics"
        if (png_files or diag_dir.exists()) and not plots_dst.exists():
            moves.append((plots_src, plots_dst, "plots_special"))

    if not moves:
        print("  Nothing to migrate — already versioned or no outputs found.")
        return

    for src, dst, move_type in moves:
        if move_type == "dir":
            print(f"  MOVE DIR:  {src}/ -> {dst}/")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        elif move_type == "file":
            print(f"  MOVE FILE: {src} -> {dst}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        elif move_type == "qa_special":
            # Move JSONL files into v1/ subdirectory
            jsonl_files = list(src.glob("*.jsonl"))
            print(f"  MOVE Q&A:  {len(jsonl_files)} JSONL files -> {dst}/")
            if not dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                for f in jsonl_files:
                    shutil.move(str(f), str(dst / f.name))

        elif move_type == "plots_special":
            # Move PNG files and diagnostics/ into v1/
            print(f"  MOVE PLOTS: {src}/*.png + diagnostics/ -> {dst}/")
            if not dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                # Move PNGs
                for f in src.glob("*.png"):
                    shutil.move(str(f), str(dst / f.name))
                # Move diagnostics dir
                diag = src / "diagnostics"
                if diag.exists():
                    shutil.move(str(diag), str(dst / "diagnostics"))

    print(f"\n{'─'*60}")
    if dry_run:
        print("  This was a DRY RUN. No files were moved.")
        print("  Run with --apply to perform the migration:")
        print("    python scripts/migrate_to_versioned.py --apply")
    else:
        print("  Migration complete!")
        print("  Current version in config: v1")
        print("  To start Phase 8, change version to 'v2' in configs/default.yaml")
        print("  Or use: python scripts/pretrain_model.py --version v2")
    print()


if __name__ == "__main__":
    main()
