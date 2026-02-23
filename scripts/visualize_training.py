#!/usr/bin/env python3
"""Visualize training progress for pretrained and fine-tuned models.

Generates loss curve plots for each model and a summary dashboard.

Usage:
    python scripts/visualize_training.py                        # pretrain summary (current version)
    python scripts/visualize_training.py --stage finetune       # finetune summary
    python scripts/visualize_training.py --stage all            # both stages side by side
    python scripts/visualize_training.py --version v2           # visualize v2 run
    python scripts/visualize_training.py --no-plots
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_book_sizes(manifest_path: str = "data/books/manifest.json") -> dict:
    """Load book sizes from manifest and actual file sizes."""
    manifest_path = Path(manifest_path)
    sizes = {}

    if not manifest_path.exists():
        return sizes

    with open(manifest_path) as f:
        manifest = json.load(f)

    for entry in manifest:
        book_id = entry["book_id"]
        chars = entry.get("token_count", 0)
        file_path = Path(entry.get("file_path", ""))
        file_kb = file_path.stat().st_size / 1024 if file_path.exists() else 0
        sizes[book_id] = {"chars": chars, "file_kb": file_kb}

    return sizes


def load_qa_stats(qa_dir: str = "data/books/qa") -> dict:
    """Load Q&A pair counts from generated JSONL files."""
    qa_dir = Path(qa_dir)
    stats = {}

    if not qa_dir.exists():
        return stats

    for qa_file in qa_dir.glob("*.jsonl"):
        book_id = qa_file.stem
        count = sum(1 for line in qa_file.open() if line.strip())
        stats[book_id] = count

    return stats


def parse_training_log(log_path: str, marker: str = "Pretraining") -> dict:
    """Parse a training log for per-book timing data from the latest run.

    Args:
        log_path: Path to the log file.
        marker: 'Pretraining' or 'Processing' to match the right log lines.
    """
    log_path = Path(log_path)
    timing = {}

    if not log_path.exists():
        return timing

    with open(log_path) as f:
        lines = f.readlines()

    # Find the LAST run start — look for the latest "[1/N]"
    last_run_start = 0
    for i, line in enumerate(lines):
        if re.search(r"\[1/\d+\] " + marker + ":", line):
            last_run_start = i

    # Parse from the last run
    current_book = None
    for line in lines[last_run_start:]:
        m = re.search(r"\[\d+/\d+\] " + marker + r": .+? \((\S+)\)", line)
        if m:
            current_book = m.group(1)

        m = re.search(r"Training complete: \{(.+)\}", line)
        if m and current_book:
            s = m.group(1)
            total_time = float(re.search(r"'total_time_seconds': ([\d.]+)", s).group(1))
            steps_m = re.search(r"'total_steps': (\d+)", s)
            steps = int(steps_m.group(1)) if steps_m else 0
            timing[current_book] = {"time_sec": total_time, "steps": steps}
            current_book = None

    return timing


def parse_finetune_log(log_path: str = "logs/finetune.log") -> dict:
    """Parse finetune.log for per-book timing data from the latest run."""
    log_path = Path(log_path)
    timing = {}

    if not log_path.exists():
        return timing

    with open(log_path) as f:
        lines = f.readlines()

    # Find the LAST run start
    last_run_start = 0
    for i, line in enumerate(lines):
        if re.search(r"\[1/\d+\] Processing:", line):
            last_run_start = i

    current_book = None
    for line in lines[last_run_start:]:
        m = re.search(r"\[\d+/\d+\] Processing: .+? \((\S+)\)", line)
        if m:
            current_book = m.group(1)

        m = re.search(r"Fine-tuning stats for (\S+): \{(.+)\}", line)
        if m:
            book_id = m.group(1)
            s = m.group(2)
            time_m = re.search(r"'total_time_seconds': ([\d.]+)", s)
            steps_m = re.search(r"'total_steps': (\d+)", s)
            if time_m:
                timing[book_id] = {
                    "time_sec": float(time_m.group(1)),
                    "steps": int(steps_m.group(1)) if steps_m else 0,
                }
            current_book = None

    return timing


def load_training_logs(models_dir: str) -> dict:
    """Load all training logs from model directories."""
    models_dir = Path(models_dir)
    logs = {}

    if not models_dir.exists():
        return logs

    for model_dir in sorted(models_dir.iterdir()):
        log_path = model_dir / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                logs[model_dir.name] = json.load(f)

    return logs


def print_summary(logs: dict, book_sizes: dict, timing: dict, stage: str = "pretrain",
                  qa_stats: dict | None = None, pretrain_logs: dict | None = None):
    """Print a text-based summary of all training runs with size and timing."""
    if not logs:
        print(f"No {stage} training logs found.")
        return

    is_finetune = stage == "finetune"

    # Build header
    W = 130 if is_finetune else 120
    title = "FINE-TUNING REPORT" if is_finetune else "PRETRAINING REPORT"
    print(f"\n  {title}")
    print(f"{'='*W}")

    if is_finetune:
        print(
            f"{'#':<3} {'Book ID':<42} {'QA Pairs':>8} "
            f"{'Epochs':>6} {'Time':>7} {'s/ep':>5} "
            f"{'FT Val':>9} {'FT PPL':>9} {'PT Val':>9} {'Δ Loss':>7}"
        )
    else:
        print(
            f"{'#':<3} {'Book ID':<45} {'Size In KB':>7} {'Chars':>10} "
            f"{'Epochs':>6} {'Time':>7} {'s/ep':>5} "
            f"{'Val Loss':>9} {'Val PPL':>9} {'Train Loss':>11}"
        )
    print(f"{'='*W}")

    total_epochs = 0
    total_time = 0.0
    best_models = []
    rows = []
    improved_count = 0

    for book_id, log in sorted(logs.items()):
        n_epochs = len(log.get("train_losses", []))
        best_val = log.get("best_val_loss", float("inf"))
        best_ppl = math.exp(min(best_val, 20)) if best_val < float("inf") else float("inf")

        train_losses = log.get("train_losses", [])
        final_train = train_losses[-1]["loss"] if train_losses else float("inf")

        size_info = book_sizes.get(book_id, {})
        file_kb = size_info.get("file_kb", 0)
        chars = size_info.get("chars", 0)

        time_info = timing.get(book_id, {})
        time_sec = time_info.get("time_sec", 0)
        sec_per_epoch = time_sec / max(n_epochs, 1) if time_sec > 0 else 0

        qa_count = qa_stats.get(book_id, 0) if qa_stats else 0

        # Get pretrain val loss for comparison
        pt_val = None
        if pretrain_logs and book_id in pretrain_logs:
            pt_val = pretrain_logs[book_id].get("best_val_loss")

        delta = (best_val - pt_val) if pt_val is not None else None
        if delta is not None and delta < 0:
            improved_count += 1

        total_epochs += n_epochs
        total_time += time_sec
        best_models.append((book_id, best_val, best_ppl))
        rows.append({
            "book_id": book_id, "file_kb": file_kb, "chars": chars,
            "n_epochs": n_epochs, "time_sec": time_sec, "sec_per_epoch": sec_per_epoch,
            "best_val": best_val, "best_ppl": best_ppl, "final_train": final_train,
            "qa_count": qa_count, "pt_val": pt_val, "delta": delta,
        })

    for i, r in enumerate(rows, 1):
        time_str = f"{r['time_sec']/60:.1f}m" if r["time_sec"] > 0 else "-"
        spe_str = f"{r['sec_per_epoch']:.1f}" if r["sec_per_epoch"] > 0 else "-"

        if is_finetune:
            qa_str = f"{r['qa_count']:,}" if r["qa_count"] > 0 else "-"
            pt_str = f"{r['pt_val']:.4f}" if r["pt_val"] is not None else "-"
            delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "-"
            print(
                f"{i:<3} {r['book_id']:<42} {qa_str:>8} "
                f"{r['n_epochs']:>6} {time_str:>7} {spe_str:>5} "
                f"{r['best_val']:>9.4f} {r['best_ppl']:>9.1f} {pt_str:>9} {delta_str:>7}"
            )
        else:
            size_str = f"{r['file_kb']:.0f}" if r["file_kb"] > 0 else "-"
            chars_str = f"{r['chars']:,}" if r["chars"] > 0 else "-"
            print(
                f"{i:<3} {r['book_id']:<45} {size_str:>7} {chars_str:>10} "
                f"{r['n_epochs']:>6} {time_str:>7} {spe_str:>5} "
                f"{r['best_val']:>9.4f} {r['best_ppl']:>9.1f} {r['final_train']:>11.4f}"
            )

    print(f"{'='*W}")
    total_time_str = f"{total_time/60:.1f}m" if total_time > 0 else "-"
    avg_time_str = f"{total_time/len(rows)/60:.1f}m" if total_time > 0 and rows else "-"
    summary = f"    Models: {len(logs)}  |  Total epochs: {total_epochs}  |  Total time: {total_time_str}  |  Avg time/book: {avg_time_str}"
    if is_finetune and qa_stats:
        total_qa = sum(qa_stats.get(r["book_id"], 0) for r in rows)
        summary += f"  |  Total Q&A pairs: {total_qa:,}"
    if is_finetune:
        summary += f"  |  Improved vs pretrain: {improved_count}/{len(rows)}"
    print(summary)

    # Top 5 best models
    best_models.sort(key=lambda x: x[1])
    print(f"\n  Top 5 (best val loss):")
    for book_id, val_loss, val_ppl in best_models[:5]:
        print(f"    {book_id}: val_loss={val_loss:.4f}, PPL={val_ppl:.1f}")

    # Bottom 5 (worst)
    print(f"\n  Bottom 5 (worst val loss):")
    for book_id, val_loss, val_ppl in best_models[-5:]:
        print(f"    {book_id}: val_loss={val_loss:.4f}, PPL={val_ppl:.1f}")


def plot_loss_curves(logs: dict, output_dir: str = "plots", stage: str = "pretrain"):
    """Generate loss curve plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Falling back to text summary only.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{stage}_" if stage != "pretrain" else ""

    # Individual model plots
    for book_id, log in logs.items():
        train_losses = log.get("train_losses", [])
        val_losses = log.get("val_losses", [])

        if not train_losses:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Train losses (by epoch)
        epochs = [e["epoch"] for e in train_losses]
        t_losses = [e["loss"] for e in train_losses]
        ax.plot(epochs, t_losses, "b-", label="Train Loss", alpha=0.8)

        # Val losses — use step-based x-axis scaled to epochs
        if val_losses:
            v_steps = [v["step"] for v in val_losses]
            v_losses = [v["loss"] for v in val_losses]
            max_step = v_steps[-1] if v_steps else 1
            max_epoch = epochs[-1] if epochs else 1
            # Scale steps to epoch range for alignment
            v_epochs = [s / max_step * max_epoch for s in v_steps]
            ax.plot(v_epochs, v_losses, "r--", label="Val Loss", alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{stage.title()}: {book_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{book_id}_loss.png", dpi=100)
        plt.close(fig)

    # Summary dashboard
    if len(logs) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(34, 20))

        # Plot 1: Best val loss distribution
        ax = axes[0, 0]
        val_losses = [(k, v.get("best_val_loss", float("inf"))) for k, v in logs.items()]
        val_losses.sort(key=lambda x: x[1])
        names = [v[0][:60] for v in val_losses[:60]]  # Top 60
        losses = [v[1] for v in val_losses[:60]]
        ax.barh(range(len(names)), losses, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Best Val Loss")
        ax.set_title(f"{stage.title()} — Top {len(names)} Models by Val Loss")
        ax.invert_yaxis()

        # Plot 2: PPL distribution
        ax = axes[0, 1]
        ppls = [math.exp(min(v[1], 15)) for v in val_losses]
        ax.hist(ppls, bins=20, color="coral", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Best Val Perplexity")
        ax.set_ylabel("Count")
        ax.set_title(f"{stage.title()} — PPL Distribution")

        # Plot 3: Epochs trained
        ax = axes[1, 0]
        epoch_counts = [len(v.get("train_losses", [])) for v in logs.values()]
        max_ep = max(epoch_counts) if epoch_counts else 200
        bin_size = max(5, (max_ep // 15) + 1)  # ~15 bins, minimum width 5
        bins = range(0, max_ep + bin_size + 1, bin_size)
        ax.hist(epoch_counts, bins=bins, color="mediumpurple", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Epochs Trained")
        ax.set_ylabel("Count")
        ax.set_title(f"{stage.title()} — Training Duration Distribution")

        # Plot 4: All loss curves overlaid
        ax = axes[1, 1]
        for book_id, log in list(logs.items())[:60]:  # Top 60
            train_losses = log.get("train_losses", [])
            if train_losses:
                epochs = [e["epoch"] for e in train_losses]
                t_losses = [e["loss"] for e in train_losses]
                ax.plot(epochs, t_losses, alpha=0.5, linewidth=0.8, label=book_id[:15])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title(f"{stage.title()} — Loss Curves (first 60 models)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}training_dashboard.png", dpi=120)
        plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")
    print(f"  - {len(logs)} individual loss curves ({prefix}*_loss.png)")
    if len(logs) > 1:
        print(f"  - {prefix}training_dashboard.png (summary)")


def main():
    from bookgpt.utils.paths import versioned_paths, add_version_arg

    parser = argparse.ArgumentParser(description="Visualize BookGPT training progress")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "dpo", "all"],
                        help="Which training stage to visualize")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Override model directory (default: auto from stage)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save plots (default: version-aware path)")
    parser.add_argument("--manifest", type=str, default="data/books/manifest.json",
                        help="Path to book manifest for size info")
    parser.add_argument("--log", type=str, default=None,
                        help="Override log file path")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only print text summary, skip plot generation")
    add_version_arg(parser)
    args = parser.parse_args()

    # Resolve versioned paths
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    paths = versioned_paths(config, args.version)

    output_dir = args.output_dir or paths["plots_dir"]

    book_sizes = load_book_sizes(args.manifest)
    qa_stats = load_qa_stats(paths["qa_dir"])

    stages = ["pretrain", "finetune", "dpo"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "pretrain":
            models_dir = args.models_dir or paths["pretrained_dir"]
            log_path = args.log or f"{paths['logs_dir']}/pretrain.log"
            logs = load_training_logs(models_dir)
            timing = parse_training_log(log_path, marker="Pretraining")
            print_summary(logs, book_sizes, timing, stage="pretrain")
            if not args.no_plots and logs:
                plot_loss_curves(logs, output_dir, stage="pretrain")

        elif stage == "finetune":
            models_dir = args.models_dir or paths["finetuned_dir"]
            log_path = args.log or f"{paths['logs_dir']}/finetune.log"
            logs = load_training_logs(models_dir)
            timing = parse_finetune_log(log_path)
            pretrain_logs = load_training_logs(paths["pretrained_dir"])
            print_summary(logs, book_sizes, timing, stage="finetune",
                          qa_stats=qa_stats, pretrain_logs=pretrain_logs)
            if not args.no_plots and logs:
                plot_loss_curves(logs, output_dir, stage="finetune")

        elif stage == "dpo":
            models_dir = args.models_dir or f"{paths['dpo_dir']}/models"
            logs = load_training_logs(models_dir)
            timing = {}  # DPO timing from training_log.json directly
            # Extract timing from the logs themselves
            for book_id, log in logs.items():
                train_losses = log.get("train_losses", [])
                if train_losses:
                    # Approximate time from epoch count (not precise but useful)
                    timing[book_id] = {"time_sec": 0, "steps": 0}
            finetune_logs = load_training_logs(paths["finetuned_dir"])
            print_summary(logs, book_sizes, timing, stage="dpo",
                          pretrain_logs=finetune_logs)
            if not args.no_plots and logs:
                plot_loss_curves(logs, output_dir, stage="dpo")

            # Also print DPO comparison summaries if reports exist
            reports_dir = Path(f"{paths['dpo_dir']}/reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json"))
                if report_files:
                    print(f"\n  DPO COMPARISON SUMMARIES ({len(report_files)} books)")
                    print(f"  {'Book ID':<42} {'Pre Comp':>9} {'Post Comp':>10} {'Delta':>7} {'Wins':>12}")
                    print(f"  {'='*82}")
                    for rf in sorted(report_files):
                        try:
                            report = json.loads(rf.read_text())
                            s = report.get("summary", {})
                            pre_c = s.get("avg_scores", {}).get("pre_dpo", {}).get("composite", 0)
                            post_c = s.get("avg_scores", {}).get("post_dpo", {}).get("composite", 0)
                            wins = s.get("wins", {})
                            win_str = f"P:{wins.get('post', 0)} F:{wins.get('pre', 0)} T:{wins.get('tie', 0)}"
                            print(f"  {rf.stem:<42} {pre_c:>9.4f} {post_c:>10.4f} {post_c-pre_c:>+7.4f} {win_str:>12}")
                        except Exception:
                            pass


if __name__ == "__main__":
    main()
