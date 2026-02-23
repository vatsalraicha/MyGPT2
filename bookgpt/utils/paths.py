"""Version-aware path resolution for BookGPT.

Manages versioned output directories so that different experiment runs
(v1, v2, etc.) don't overwrite each other.

Shared data (raw books, tokenizer, manifest) is NOT versioned.
Output data (models, logs, plots, Q&A) IS versioned.

Directory structure:
    data/
    ├── books/raw/          (shared, not versioned)
    ├── books/tokenized/    (shared, not versioned)
    ├── books/manifest.json (shared, not versioned)
    ├── tokenizers/shared/  (shared, not versioned)
    ├── models/v1/pretrained/
    ├── models/v1/finetuned/
    ├── models/v2/pretrained/
    ├── books/qa/v1/
    ├── books/qa/v2/
    logs/v1/
    logs/v2/
    plots/v1/
    plots/v2/
"""

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load config and return it."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_version(config: dict | None = None, cli_version: str | None = None) -> str:
    """Get the version string. CLI flag takes priority over config."""
    if cli_version:
        return cli_version
    if config and "version" in config:
        return config["version"]
    return "v1"


def _resolve(versioned: str, legacy: str) -> str:
    """Return versioned path if it exists, otherwise fall back to legacy path.

    For output dirs that don't exist yet (new version), returns the versioned path.
    For existing data from before migration, falls back to the legacy path.
    """
    if Path(versioned).exists():
        return versioned
    if Path(legacy).exists():
        return legacy
    # Neither exists — return versioned (will be created by ensure_dirs)
    return versioned


def versioned_paths(config: dict, version: str | None = None) -> dict:
    """Return a dict of all versioned paths.

    Uses fallback logic: if the versioned directory doesn't exist but the
    legacy (unversioned) directory does, uses the legacy path. This allows
    scripts to work seamlessly before and after migration.

    Args:
        config: Loaded config dict.
        version: Override version (if None, uses config["version"]).

    Returns:
        Dict with keys matching config["data"] but with versioned paths.
    """
    v = get_version(config, version)
    data = config.get("data", {})

    return {
        # Shared (not versioned)
        "books_dir": data.get("books_dir", "data/books/raw"),
        "tokenized_dir": data.get("tokenized_dir", "data/books/tokenized"),
        "manifest_path": data.get("manifest_path", "data/books/manifest.json"),
        "tokenizers_dir": data.get("tokenizers_dir", "data/tokenizers"),

        # Versioned outputs (with legacy fallback for pre-migration)
        "pretrained_dir": _resolve(f"data/models/{v}/pretrained", data.get("pretrained_dir", "data/models/pretrained")),
        "finetuned_dir": _resolve(f"data/models/{v}/finetuned", data.get("finetuned_dir", "data/models/finetuned")),
        "qa_dir": _resolve(f"data/books/qa/{v}", data.get("qa_dir", "data/books/qa")),
        "router_dir": _resolve(f"data/router/{v}", data.get("router_dir", "data/router")),
        "logs_dir": _resolve(f"logs/{v}", data.get("logs_dir", "logs")),
        "plots_dir": _resolve(f"plots/{v}", "plots"),
        "diagnostics_dir": _resolve(f"plots/{v}/diagnostics", "plots/diagnostics"),
        "dpo_dir": _resolve(f"data/dpo/{v}", "data/dpo"),

        # Metadata
        "version": v,
    }


def ensure_dirs(paths: dict):
    """Create all versioned output directories if they don't exist."""
    for key in ["pretrained_dir", "finetuned_dir", "qa_dir", "router_dir",
                "logs_dir", "plots_dir", "diagnostics_dir", "dpo_dir"]:
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)


def add_version_arg(parser):
    """Add --version argument to an argparse parser."""
    parser.add_argument(
        "--version", type=str, default=None,
        help="Run version (e.g., v1, v2). Controls output directories. "
             "Overrides config value if set.",
    )


def migrate_v1_to_versioned():
    """One-time helper: move existing unversioned outputs into v1/ structure.

    Only moves if the versioned directories don't already exist.
    Call this once before switching to versioned layout.
    """
    moves = [
        ("data/models/pretrained", "data/models/v1/pretrained"),
        ("data/models/finetuned", "data/models/v1/finetuned"),
        ("data/books/qa", "data/books/qa_v1_staging"),  # Special: qa has subdirs
    ]

    log_moves = [
        ("logs/pretrain.log", "logs/v1/pretrain.log"),
        ("logs/finetune.log", "logs/v1/finetune.log"),
        ("logs/router.log", "logs/v1/router.log"),
        ("logs/dpo.log", "logs/v1/dpo.log"),
    ]

    plot_moves = [
        ("plots", "plots/v1_staging"),  # Rename existing plots dir
    ]

    print("Migration plan (v1 versioned layout):")
    print("=" * 60)

    import shutil

    for src, dst in moves:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.exists() and not dst_path.exists():
            print(f"  MOVE: {src} -> {dst}")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
        elif dst_path.exists():
            print(f"  SKIP: {dst} already exists")
        else:
            print(f"  SKIP: {src} not found")

    # Handle logs (file-level moves)
    Path("logs/v1").mkdir(parents=True, exist_ok=True)
    for src, dst in log_moves:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.exists() and not dst_path.exists():
            print(f"  MOVE: {src} -> {dst}")
            shutil.move(str(src_path), str(dst_path))
        elif dst_path.exists():
            print(f"  SKIP: {dst} already exists")

    print("\nDone. Verify the migration, then update configs/default.yaml version to 'v1'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_v1_to_versioned()
    else:
        config = load_config()
        paths = versioned_paths(config)
        print("Current versioned paths:")
        for k, v in paths.items():
            print(f"  {k:<20} {v}")
