#!/usr/bin/env python3
"""Run full diagnostics (with plots) on every finetuned book model.

Usage:
    python scripts/diagnose_all_books.py                          # current version
    python scripts/diagnose_all_books.py --version v2             # specific version
    python scripts/diagnose_all_books.py --skip-arxiv             # skip arxiv papers
    python scripts/diagnose_all_books.py --no-plots               # reports only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.utils.paths import versioned_paths, add_version_arg


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics on all finetuned models")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arxiv papers")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    add_version_arg(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    paths = versioned_paths(config, args.version)

    models_dir = Path(paths["finetuned_dir"])
    book_dirs = sorted([d for d in models_dir.iterdir() if (d / "best").exists()])

    if args.skip_arxiv:
        before = len(book_dirs)
        book_dirs = [d for d in book_dirs if not d.name.startswith("arxiv_")]
        print(f"Skipping arxiv: {before - len(book_dirs)} removed")

    total = len(book_dirs)
    print(f"Found {total} finetuned models to diagnose (version: {paths['version']})")
    print(f"Plots: {'DISABLED' if args.no_plots else 'ENABLED'}")
    print(f"=" * 60)

    completed = 0
    failed = []
    start_all = time.time()

    for i, book_dir in enumerate(book_dirs):
        book_id = book_dir.name
        start = time.time()
        print(f"\n[{i+1}/{total}] Diagnosing: {book_id}")

        cmd = [
            sys.executable, "scripts/diagnose_model.py",
            "--book-id", book_id,
            "--version", paths["version"],
        ]
        if args.no_plots:
            cmd.append("--no-plots")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max per book
            )

            elapsed = time.time() - start

            if result.returncode == 0:
                completed += 1
                print(f"  ✓ Done in {elapsed:.0f}s")
            else:
                failed.append(book_id)
                # Print last few lines of stderr for debugging
                err_lines = result.stderr.strip().split("\n")[-3:]
                print(f"  ✗ Failed in {elapsed:.0f}s")
                for line in err_lines:
                    print(f"    {line}")

        except subprocess.TimeoutExpired:
            failed.append(book_id)
            print(f"  ✗ Timed out (>600s)")
        except Exception as e:
            failed.append(book_id)
            print(f"  ✗ Error: {e}")

    total_time = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTICS COMPLETE")
    print(f"  Version: {paths['version']}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Completed: {completed}/{total}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed books: {', '.join(failed)}")
    print(f"  Plots saved to: {paths['diagnostics_dir']}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
