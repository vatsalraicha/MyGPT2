#!/usr/bin/env python3
"""Interactive CLI chatbot for BookGPT.

Routes questions to specialized book models and returns answers.

Usage:
    python scripts/chat.py
    python scripts/chat.py --no-finetuned  # use pretrained models only
    python scripts/chat.py --top-k 1       # consult only the top-1 book
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.orchestrator.orchestrator import Orchestrator
from bookgpt.utils.device import get_device, set_seed

logging.basicConfig(
    level=logging.WARNING,  # Keep CLI clean; only show warnings+
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   📚  BookGPT  —  Book-Specialized Micro GPT-2 Models   ║
║                                                          ║
║   Ask questions about math, and specialized book models  ║
║   will collaborate to answer.                            ║
║                                                          ║
║   Commands:                                              ║
║     /quit or /exit  — exit the chat                      ║
║     /books          — list available books                ║
║     /strategy <s>   — change merge strategy              ║
║     /topk <n>       — change number of models consulted  ║
║     /verbose        — toggle verbose mode                ║
║     /help           — show this help                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="BookGPT Interactive Chat")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="data/books/manifest.json")
    parser.add_argument("--router-dir", type=str, default="data/router")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizers/shared")
    parser.add_argument("--no-finetuned", action="store_true", help="Use pretrained models only")
    parser.add_argument("--top-k", type=int, default=3, help="Number of models to consult")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--strategy", type=str, default="confidence",
                        choices=["confidence", "voting", "concat"])
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = get_device(force_cpu=args.force_cpu)

    # Verify required files exist
    for path, name in [
        (args.manifest, "Book manifest"),
        (args.router_dir, "Router artifacts"),
    ]:
        if not Path(path).exists():
            print(f"Error: {name} not found at {path}")
            print("Run the pipeline first: crawl_books → train_tokenizer → pretrain → finetune → train_router")
            sys.exit(1)

    # Initialize orchestrator
    print("Loading models... (this may take a moment)")
    orchestrator = Orchestrator(
        manifest_path=args.manifest,
        router_dir=args.router_dir,
        models_dir=args.models_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=device,
        merge_strategy=args.strategy,
        use_finetuned=not args.no_finetuned,
    )

    # State
    top_k = args.top_k
    temperature = args.temperature
    verbose = True

    print(BANNER)
    print(f"Device: {device} | Strategy: {args.strategy} | Top-K: {top_k}")
    print(f"Models: {'fine-tuned' if not args.no_finetuned else 'pretrained'}")
    print(f"Books indexed: {len(orchestrator.router.book_ids)}")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            cmd_name = cmd[0]

            if cmd_name in ("/quit", "/exit"):
                print("Goodbye!")
                break

            elif cmd_name == "/help":
                print(BANNER)

            elif cmd_name == "/books":
                print("\nAvailable books:")
                for book_id in orchestrator.router.book_ids:
                    meta = orchestrator.manifest.get(book_id, {})
                    title = meta.get("title", book_id)
                    source = meta.get("source", "unknown")
                    print(f"  [{source}] {book_id}: {title}")
                print()

            elif cmd_name == "/strategy":
                if len(cmd) > 1 and cmd[1] in ("confidence", "voting", "concat"):
                    orchestrator.merge_strategy = cmd[1]
                    print(f"Merge strategy set to: {cmd[1]}")
                else:
                    print("Usage: /strategy <confidence|voting|concat>")

            elif cmd_name == "/topk":
                if len(cmd) > 1:
                    try:
                        top_k = int(cmd[1])
                        print(f"Top-K set to: {top_k}")
                    except ValueError:
                        print("Usage: /topk <number>")
                else:
                    print(f"Current top-k: {top_k}")

            elif cmd_name == "/verbose":
                verbose = not verbose
                print(f"Verbose mode: {'on' if verbose else 'off'}")

            else:
                print(f"Unknown command: {cmd_name}. Type /help for commands.")

            continue

        # Process query
        result = orchestrator.query(
            question=user_input,
            top_k=top_k,
            max_answer_tokens=config.get("generate", {}).get("max_tokens", 256),
            temperature=temperature,
            verbose=verbose,
        )

        if not verbose:
            # In non-verbose mode, just show the final answer
            print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    main()
