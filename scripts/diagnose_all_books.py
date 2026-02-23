#!/usr/bin/env python3
"""Run full diagnostics (with plots) on every finetuned book model.

Usage:
    python scripts/diagnose_all_books.py
    python scripts/diagnose_all_books.py --no-plots   # reports only, skip plot generation
"""

import subprocess
import sys
import time
from pathlib import Path


def main():
    no_plots = "--no-plots" in sys.argv

    models_dir = Path("data/models/finetuned")
    book_dirs = sorted([d for d in models_dir.iterdir() if (d / "best").exists()])

    total = len(book_dirs)
    print(f"Found {total} finetuned models to diagnose")
    print(f"Plots: {'DISABLED' if no_plots else 'ENABLED'}")
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
        ]
        if no_plots:
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
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Completed: {completed}/{total}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed books: {', '.join(failed)}")
    print(f"  Plots saved to: plots/diagnostics/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
