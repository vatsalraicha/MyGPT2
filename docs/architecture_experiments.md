# Architecture Experiments Log — Phase 8

## Context
BookGPT v2: per-book GPT-2 (49M params, 6L/12H/768E), trained from scratch on 42 math textbooks.
Goal: improve pretraining quality, especially for hard books (non-Euclidean geometry, PPL 1267).

## Baseline (v2)
- Learned absolute positional embeddings: `nn.Embedding(512, 768)`
- Full causal attention, context length 512
- LR 2e-4, patience 15
- Best PPL on non-Euclidean geometry: **1267** (epoch ~21)

## Experiments

### 1. RoPE + Sliding Window Attention (SWA W=512) + Context 1024
**Hypothesis**: RoPE encodes relative positions (better for math cross-references), SWA reduces compute, longer context sees more of the proof chain.

| Config | LR | Best Val PPL | Best Epoch |
|--------|-----|-------------|------------|
| RoPE + SWA + ctx 1024 | 2e-4 | 1471 | ~39 |
| RoPE + SWA + ctx 1024 | 3e-4 | 1517 | 7 |

**Result**: WORSE. SWA at W=512 with ctx 1024 gives the same effective attention range as full attention at ctx 512, but with half the training samples (151 vs 339).

### 2. RoPE + Full Attention + Context 512
**Hypothesis**: Isolate RoPE effect — same sample count as baseline, just different positional encoding.

| Config | LR | Best Val PPL | Best Epoch |
|--------|-----|-------------|------------|
| RoPE + full attn + ctx 512 | 3e-4 | 1349 | ~20 |

**Result**: WORSE than baseline (1349 vs 1267). RoPE itself doesn't help for per-book training from scratch.

### 3. RoPE + Full Attention + Context 1024
**Hypothesis**: Longer context with full attention (no SWA restriction).

| Config | LR | Best Val PPL | Best Epoch |
|--------|-----|-------------|------------|
| RoPE + full attn + ctx 1024 | 3e-4 | 1519 | ~7 |

**Result**: WORST. Fewer samples (151 vs 339) hurts more than longer context helps.

## Summary Table

| # | Config | Context | Pos Encoding | Attention | LR | PPL | vs Baseline |
|---|--------|---------|-------------|-----------|-----|-----|-------------|
| 0 | **Baseline** | **512** | **Absolute** | **Full** | **2e-4** | **1267** | — |
| 1 | RoPE+SWA | 1024 | RoPE | SWA W=512 | 2e-4 | 1471 | +16% worse |
| 2 | RoPE+SWA | 1024 | RoPE | SWA W=512 | 3e-4 | 1517 | +20% worse |
| 3 | RoPE only | 512 | RoPE | Full | 3e-4 | 1349 | +6% worse |
| 4 | RoPE+long ctx | 1024 | RoPE | Full | 3e-4 | 1519 | +20% worse |

## Why RoPE Didn't Help
1. **Per-book training from scratch**: With only 339 training samples, the model has enough capacity to memorize absolute position patterns for 512 positions. RoPE's generalization advantage doesn't materialize with so little data.
2. **Context 1024 = fewer samples**: Doubling context halves training samples, which hurts optimization more than longer context helps for a data-starved model.
3. **SWA negated context benefit**: W=512 window with 1024 context = same effective attention as 512 full. Worst of both worlds.

## Suggestions Received (from Mistral AI analysis)

### Tried
- [x] RoPE (Rotary Position Embeddings) — didn't help
- [x] Sliding Window Attention — hurt performance
- [x] Longer context (1024) — fewer samples, worse results
- [x] Higher LR (3e-4) — marginal, not the bottleneck

### Not Tried (potentially useful for future projects)
- [ ] **Custom math tokenizer**: Train tokenizer that treats math symbols as single tokens. Our current BPE handles ASCII math fine, but unicode-heavy texts might benefit.
- [ ] **ALiBi** (Attention with Linear Biases): Simpler than RoPE, adds linear distance bias to attention. Worth trying as lighter-weight alternative.
- [ ] **Reference markers**: Preprocess text to replace "By Theorem 1" with `<ref_theorem_1>`. Turns relative references into absolute tokens. Complex preprocessing, unclear benefit for small per-book models.
- [ ] **Curriculum learning**: Train on short definitions first, then full proofs. Adds training complexity.
- [ ] **Multi-task pretraining**: Masked symbol prediction, proof step ordering. Significant architecture change.
- [ ] **Different theta values for RoPE**: theta=5000 or 20000 instead of 10000. Not tested.
- [ ] **Partial RoPE**: Apply to subset of attention heads only.

### Rejected (not applicable to our setup)
- Leverage pretrained math models (Galactica, Minerva) — violates no-LLM-dependency principle
- Graph attention for symbolic relationships — too complex for per-book setup

## Tokenization Analysis
- Book uses Gutenberg ASCII with `$...$` inline math (4016 expressions)
- No unicode math symbols to worry about
- BPE tokenizer handles it correctly: `$ABC$` → `['$', 'ABC', '$']`
- Variable names, operators all get reasonable tokens

## Root Cause
The high PPL on hard books (non-Euclidean geometry, mathematical recreations, etc.) is primarily a **data/capacity issue**, not an architecture one:
- 49M params trained on ~170K tokens is severely data-limited
- The model overfits training data while val plateaus early
- Complex formal math with heavy cross-referencing requires more data or more capacity

## External Tokenizer Test (GPT-2 tiktoken)

**Hypothesis**: Our custom BPE (8192 vocab, trained on 42 math books) might have poor merge rules. Using a production tokenizer trained on billions of tokens could help.

| Config | Tokenizer | Vocab | Model Params | Pos Encoding | Best Val PPL |
|--------|-----------|-------|-------------|-------------|-------------|
| Baseline | Custom BPE | 8,192 | 49M | Absolute | **1267** |
| tiktoken | GPT-2 | 50,257 | 81M | RoPE+SWA | **649** |

**Result**: PPL nearly halved (1267 → 649). BUT the comparison is not clean:
- 81M params vs 49M (65% more capacity from larger embedding table)
- Uses RoPE instead of absolute positions (though RoPE alone was worse)
- Token counts similar: 174K (BPE) vs 180K (tiktoken)

**Conclusion**: Better tokenization + more parameters helps, but we cannot isolate which factor contributed more. For a future project, a tokenizer trained on a large corpus would be beneficial. For now, our 8,192-vocab BPE is adequate — the primary bottleneck is data scarcity.

## Decision
Revert to absolute positional embeddings (baseline v2 architecture). Keep improvements:
- LR 2e-4 (proven better than 1e-4)
- Patience 30, min_epochs 50
- Epoch-level val loss logging
- min_delta 0.001

## Future Project Ideas
- Build a dedicated math tokenizer trained on large math corpus
- Experiment with ALiBi (simpler than RoPE, might work better for small models)
- Try larger shared pretraining across all books before per-book specialization
- Investigate whether some books should simply be excluded (too hard for the model size)
