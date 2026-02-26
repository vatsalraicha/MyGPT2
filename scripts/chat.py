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
║   📚 BookGPT  —  Book-Specialized Micro GPT-2 Models     ║
║                                                          ║
║   Ask questions about math, and specialized book models  ║
║   will collaborate to answer.                            ║
║                                                          ║
║   Commands:                                              ║
║     /quit or /exit  — exit the chat                      ║
║     /books          — list available books               ║
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
    parser.add_argument("--router-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizers/shared")
    parser.add_argument("--no-finetuned", action="store_true", help="Use pretrained models only")
    parser.add_argument("--book-id", type=str, default=None,
                        help="Force a specific book model (skip router)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of models to consult")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--strategy", type=str, default="confidence",
                        choices=["confidence", "voting", "concat"])
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--version", type=str, default=None,
                        help="Run version (e.g., v1, v2)")
    args = parser.parse_args()

    # Load config and resolve versioned paths
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from bookgpt.utils.paths import versioned_paths
    paths = versioned_paths(config, args.version)

    if args.router_dir is None:
        args.router_dir = paths["router_dir"]
    if args.models_dir is None:
        # models_dir is the parent containing pretrained/ and finetuned/ subdirs.
        # Use the resolved finetuned_dir's parent (handles both legacy and versioned layouts).
        args.models_dir = str(Path(paths["finetuned_dir"]).parent)

    set_seed(config.get("seed", 42))
    device = get_device(force_cpu=args.force_cpu)

    # Verify required files exist
    if not Path(args.manifest).exists():
        print(f"Error: Book manifest not found at {args.manifest}")
        print("Run the pipeline first: crawl_books → train_tokenizer → pretrain → finetune → train_router")
        sys.exit(1)

    if not Path(args.router_dir).exists() and not args.book_id:
        print(f"Error: Router not found at {args.router_dir}")
        print("Either run 'python scripts/train_router.py' first, or use --book-id to select a book directly.")
        sys.exit(1)

    # Initialize orchestrator
    print(f"Loading models (version: {paths['version']})...")
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
    forced_book = args.book_id

    print(BANNER)
    print(f"Device: {device} | Strategy: {args.strategy} | Top-K: {top_k}")
    print(f"Models: {'fine-tuned' if not args.no_finetuned else 'pretrained'}")
    if forced_book:
        print(f"Forced book: {forced_book}")
    elif orchestrator.router:
        print(f"Books indexed: {len(orchestrator.router.book_ids)}")
    else:
        print(f"Router: not loaded (use --book-id or /book to select)")
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
                book_ids = orchestrator.router.book_ids if orchestrator.router else sorted(orchestrator.manifest.keys())
                for book_id in book_ids:
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

            elif cmd_name == "/book":
                if len(cmd) > 1:
                    forced_book = cmd[1]
                    print(f"Forced book: {forced_book}")
                else:
                    forced_book = None
                    print("Book filter cleared — using router")

            else:
                print(f"Unknown command: {cmd_name}. Type /help for commands.")

            continue

        # Process query
        if forced_book:
            # Bypass router — query a specific book model directly
            result = orchestrator.query_book(
                book_id=forced_book,
                question=user_input,
                max_answer_tokens=config.get("generate", {}).get("max_tokens", 256),
                temperature=temperature,
                verbose=verbose,
            )
        elif orchestrator.router:
            result = orchestrator.query(
                question=user_input,
                top_k=top_k,
                max_answer_tokens=config.get("generate", {}).get("max_tokens", 256),
                temperature=temperature,
                verbose=verbose,
            )
        else:
            print("No router loaded. Use /book <book_id> to select a book first.")
            continue

        if not verbose:
            # In non-verbose mode, just show the final answer
            print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    main()
