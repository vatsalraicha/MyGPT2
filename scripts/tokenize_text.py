#!/usr/bin/env python3
"""Tokenize input text and display tokens with their IDs.

Usage:
    python scripts/tokenize_text.py "The derivative of f(x) = x^2 is 2x"
    python scripts/tokenize_text.py --file path/to/file.txt
    echo "Hello math world" | python scripts/tokenize_text.py --stdin
    python scripts/tokenize_text.py --interactive

Examples:
    $ python scripts/tokenize_text.py "Let x be a prime number."

    Input: Let x be a prime number.
    Tokens (8): [1743, 372, 288, 256, 3099, 2178, 46]

    Token breakdown:
      [0] 1743  'Let'
      [1]  372  ' x'
      [2]  288  ' be'
      [3]  256  ' a'
      [4] 3099  ' prime'
      [5] 2178  ' number'
      [6]   46  '.'
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.tokenizer.train_bpe import load_tokenizer, tokenize_text, decode_tokens


def display_tokens(tokenizer, text: str, show_bytes: bool = False):
    """Tokenize text and display a detailed breakdown."""
    # Get token IDs
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    tokens = encoding.tokens  # The raw token strings (byte-level)

    print(f"\n  Input: {text}")
    print(f"  Tokens ({len(token_ids)}): {token_ids}")
    print()

    # Show breakdown
    print(f"  {'Idx':>5s}  {'ID':>6s}  {'Token':20s}  {'Decoded':30s}", end="")
    if show_bytes:
        print(f"  {'Bytes'}", end="")
    print()
    print(f"  {'─'*5}  {'─'*6}  {'─'*20}  {'─'*30}", end="")
    if show_bytes:
        print(f"  {'─'*20}", end="")
    print()

    for i, (tid, tok) in enumerate(zip(token_ids, tokens)):
        # Decode individual token to see what it represents
        decoded = decode_tokens(tokenizer, [tid])
        # Escape non-printable chars for display
        display_tok = repr(tok) if not tok.isprintable() else tok
        display_decoded = repr(decoded) if any(not c.isprintable() and c not in ' \t' for c in decoded) else decoded

        print(f"  [{i:3d}]  {tid:>6d}  {display_tok:20s}  {display_decoded:30s}", end="")
        if show_bytes:
            byte_repr = " ".join(f"{b:02x}" for b in decoded.encode("utf-8"))
            print(f"  {byte_repr}", end="")
        print()

    # Stats
    print()
    compression = len(text) / max(len(token_ids), 1)
    print(f"  Compression: {compression:.2f} chars/token")
    print(f"  Characters: {len(text)}, Tokens: {len(token_ids)}")

    # Verify round-trip
    roundtrip = decode_tokens(tokenizer, token_ids)
    if roundtrip == text:
        print(f"  Round-trip: ✓ perfect")
    else:
        print(f"  Round-trip: ✗ mismatch!")
        print(f"    Original:   {repr(text[:100])}")
        print(f"    Roundtrip:  {repr(roundtrip[:100])}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text using the BookGPT shared tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to tokenize (positional arguments joined with spaces)",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Read text from a file",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read text from stdin",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: keep prompting for text",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="data/tokenizers/shared",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--show-bytes", "-b",
        action="store_true",
        help="Show UTF-8 byte representation",
    )
    parser.add_argument(
        "--vocab-search", "-v",
        type=str,
        default=None,
        help="Search the vocabulary for tokens matching a pattern",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # Vocab search mode
    if args.vocab_search:
        pattern = args.vocab_search.lower()
        vocab = tokenizer.get_vocab()
        matches = [(tok, tid) for tok, tid in vocab.items() if pattern in tok.lower()]
        matches.sort(key=lambda x: x[1])
        print(f"\nVocab search for '{pattern}': {len(matches)} matches")
        for tok, tid in matches[:50]:
            print(f"  {tid:>6d}  {repr(tok)}")
        if len(matches) > 50:
            print(f"  ... and {len(matches) - 50} more")
        return

    # Get text to tokenize
    if args.interactive:
        print("\nInteractive tokenizer (type 'quit' to exit)")
        print("=" * 60)
        while True:
            try:
                text = input("\nEnter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                display_tokens(tokenizer, text, show_bytes=args.show_bytes)

    elif args.stdin:
        text = sys.stdin.read().strip()
        if text:
            display_tokens(tokenizer, text, show_bytes=args.show_bytes)

    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        text = filepath.read_text(encoding="utf-8")
        # For files, show first 500 chars tokenized + overall stats
        print(f"\nFile: {filepath} ({len(text):,} chars)")
        ids = tokenize_text(tokenizer, text)
        print(f"Total tokens: {len(ids):,}")
        print(f"Compression: {len(text)/max(len(ids),1):.2f} chars/token")
        print(f"\nFirst 500 chars:")
        display_tokens(tokenizer, text[:500], show_bytes=args.show_bytes)

    elif args.text:
        text = " ".join(args.text)
        display_tokens(tokenizer, text, show_bytes=args.show_bytes)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
