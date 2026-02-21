# BookGPT — Book-Specialized Micro GPT-2 Models with Orchestration

Train a small GPT-2 model **from scratch** on each book, fine-tune for Q&A, and orchestrate multiple models to answer questions. Runs entirely locally on Apple Silicon (MPS).

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Crawl books (downloads ~5-10 public domain math books)
python scripts/crawl_books.py

# 3. Train a BPE tokenizer per book
python scripts/train_tokenizer.py

# 4. Pretrain GPT-2 on each book
python scripts/pretrain_model.py

# 5. Generate Q&A data and fine-tune
python scripts/finetune_model.py

# 6. Build the query router
python scripts/train_router.py

# 7. Chat!
python scripts/chat.py
```

## Architecture

```
User Query → Router → Top-K Book Models → Answer Merger → Final Answer
```

Each book gets its own:
- **BPE tokenizer** (domain-specific vocabulary)
- **GPT-2 model** (~1-10M parameters, trained from scratch)
- **Fine-tuned variant** (Q&A format with loss masking)

The **router** uses TF-IDF similarity to select the most relevant book models for a query. The **orchestrator** runs inference on selected models and merges answers by confidence scoring.

## Project Structure

```
bookgpt/
├── crawl/          # Book downloading (Gutenberg, OpenStax)
├── tokenizer/      # BPE tokenizer training
├── model/          # GPT-2 architecture, training, generation
├── data/           # Dataset preparation, Q&A generation
├── router/         # Query routing
├── orchestrator/   # Multi-model orchestration
└── utils/          # MPS device utilities
```

## Per-Book Pipeline

```bash
# Process a single book
python scripts/train_tokenizer.py --book-id calculus_made_easy
python scripts/pretrain_model.py  --book-id calculus_made_easy
python scripts/finetune_model.py  --book-id calculus_made_easy
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.n_layer` | 6 | Transformer layers |
| `model.n_head` | 8 | Attention heads |
| `model.n_embd` | 256 | Embedding dimension |
| `model.context_length` | 512 | Max sequence length |
| `tokenizer.vocab_size` | 8192 | BPE vocabulary size |
| `pretrain.learning_rate` | 3e-4 | Pretraining LR |
| `pretrain.max_epochs` | 50 | Max pretraining epochs |

## Requirements

- Python 3.11+
- PyTorch 2.x with MPS backend
- Apple Silicon Mac (M1/M2/M3/M4)
- No cloud APIs — everything runs locally
