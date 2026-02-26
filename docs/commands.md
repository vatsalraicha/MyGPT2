# BookGPT Commands Reference

All commands are run from the project root (`MyGPT2/`).

---

## Full Pipeline (in order)

```
1. Crawl       -->  2. Tokenizer  -->  3. Clean
4. Pretrain    -->  5. Q&A Gen    -->  6. Finetune
7. DPO         -->  8. Router     -->  9. Chat
```

---

## 1. Crawl Books

Download math books from Gutenberg and OpenStax.

```bash
# All sources
python scripts/crawl_books.py

# Gutenberg only, max 50
python scripts/crawl_books.py --sources gutenberg --max-books 50

# Append to existing manifest (don't overwrite)
python scripts/crawl_books.py --append
```

## 2. Train Tokenizer

Train the shared BPE tokenizer on all books.

```bash
# Default (vocab 8192)
python scripts/train_tokenizer.py

# Custom vocab size
python scripts/train_tokenizer.py --vocab-size 16384
```

## 3. Clean Books

Apply text cleaning (for Q&A generation, not pretraining).

```bash
# Clean all books
python scripts/clean_all_books.py --skip-arxiv

# Clean a single book
python scripts/clean_all_books.py --book-id calculus_made_easy

# Force re-clean
python scripts/clean_all_books.py --skip-arxiv --force
```

## 4. Pretrain

Pretrain per-book GPT-2 models on raw text.

```bash
# All books (skip arxiv)
python scripts/pretrain_model.py --skip-arxiv

# Single book
python scripts/pretrain_model.py --book-id calculus_made_easy

# Force CPU
python scripts/pretrain_model.py --skip-arxiv --force-cpu
```

**Config**: `configs/default.yaml` controls LR, epochs, patience, model size, etc.

**Logs**: `logs/v2/pretrain.log`

## 5. Generate Q&A

Generate question-answer pairs from cleaned book text.

```bash
# All books
python scripts/generate_qa.py --skip-arxiv

# Single book
python scripts/generate_qa.py --book-id calculus_made_easy

# Dry run (show stats, don't write)
python scripts/generate_qa.py --book-id calculus_made_easy --dry-run

# Show example Q&A pairs per type
python scripts/generate_qa.py --book-id calculus_made_easy --sample --examples-per-type 5
```

## 6. Finetune

Finetune pretrained models on Q&A data.

```bash
# All books (skip arxiv)
python scripts/finetune_model.py --skip-arxiv

# Single book
python scripts/finetune_model.py --book-id calculus_made_easy

# Skip Q&A generation (use existing)
python scripts/finetune_model.py --skip-arxiv --skip-qa-gen
```

## 7. DPO (Direct Preference Optimization)

Run DPO alignment to improve answer quality.

```bash
# All books
python scripts/run_dpo.py --skip-arxiv

# Single book
python scripts/run_dpo.py --book-id calculus_made_easy

# Compare only (no training, just evaluate pre vs post)
python scripts/run_dpo.py --book-id calculus_made_easy --compare-only

# Skip preference generation (use existing)
python scripts/run_dpo.py --skip-arxiv --skip-pref-gen
```

## 8. Train Router

Build the TF-IDF query router that maps questions to books.

```bash
# Default
python scripts/train_router.py

# With test queries
python scripts/train_router.py --test-queries "What is a derivative?" "Prove the Pythagorean theorem"
```

## 9. Chat

Interactive chat with the trained models.

```bash
# Default (uses router to pick books)
python scripts/chat.py

# Force a specific book
python scripts/chat.py --book-id calculus_made_easy

# Use pretrained models (skip finetune)
python scripts/chat.py --no-finetuned

# Adjust generation
python scripts/chat.py --temperature 0.5 --top-k 30
```

---

## Diagnostics & Visualization

### Visualize Training

Generate loss curves, PPL plots, and dashboards.

```bash
# Pretrain plots (default)
python scripts/visualize_training.py --skip-arxiv

# Finetune plots
python scripts/visualize_training.py --stage finetune --skip-arxiv

# DPO plots
python scripts/visualize_training.py --stage dpo --skip-arxiv

# All stages
python scripts/visualize_training.py --stage all --skip-arxiv

# Text report only (no plot images)
python scripts/visualize_training.py --skip-arxiv --no-plots
```

**Output**: `plots/v2/`
- `training_dashboard.png` — 8-panel summary (4x2 layout)
- `val_loss_curves.png` — all models val loss overlaid
- `val_ppl_curves.png` — all models val PPL overlaid
- `<book_id>_loss.png` — individual train + val loss per book

### Diagnose Model

Detailed per-model diagnostics (attention, generation quality, loss breakdown).

```bash
# Single book
python scripts/diagnose_model.py --book-id calculus_made_easy

# Single book, pretrained stage
python scripts/diagnose_model.py --book-id calculus_made_easy --stage pretrain

# All models summary
python scripts/diagnose_model.py --all
```

### Diagnose All Books

Run diagnostics across all finetuned models.

```bash
python scripts/diagnose_all_books.py --skip-arxiv

# No plots
python scripts/diagnose_all_books.py --skip-arxiv --no-plots
```

---

## Utilities

### Tokenize Text

Inspect how text gets tokenized.

```bash
# Inline text
python scripts/tokenize_text.py "The derivative of x squared is 2x"

# From file
python scripts/tokenize_text.py -f data/books/raw/calculus_made_easy.txt

# Interactive mode
python scripts/tokenize_text.py --interactive

# Search vocab
python scripts/tokenize_text.py --vocab-search "deriv"

# Show byte representation
python scripts/tokenize_text.py --show-bytes "hello world"
```

### Test External Tokenizer

Compare our BPE against production tokenizers (tiktoken).

```bash
# GPT-2 tokenizer on a specific book
python scripts/test_external_tokenizer.py --book the_elements_of_non-euclidean_geometry --lr 2e-4

# Different tokenizer
python scripts/test_external_tokenizer.py --book calculus_made_easy --tokenizer cl100k_base
```

### Export to ONNX

Export a trained model for deployment.

```bash
python scripts/export_onnx.py data/models/v2/finetuned/calculus_made_easy/best/model.pt
```

---

## Version Control

Most scripts accept `--version` to target a specific run:

```bash
# Use v1 models
python scripts/visualize_training.py --version v1

# Use v2 models (default)
python scripts/pretrain_model.py --version v2
```

The active version is set in `configs/default.yaml` under `version: "v2"`.

---

## Directory Structure

```
data/
  books/
    raw/              # Original downloaded texts
    cleaned/          # Cleaned texts (for Q&A)
    tokenized/        # Tokenized data
    qa/               # Generated Q&A pairs (.jsonl)
    manifest.json     # Book registry
  tokenizers/
    shared/           # BPE tokenizer files
  models/
    v2/
      pretrained/     # Per-book pretrained models
      finetuned/      # Per-book finetuned models
  router/             # TF-IDF router artifacts
  dpo/                # DPO models, preferences, reports
configs/
  default.yaml        # All hyperparameters and paths
logs/
  v2/                 # Training logs
plots/
  v2/                 # Generated plots
docs/                 # Documentation
```
