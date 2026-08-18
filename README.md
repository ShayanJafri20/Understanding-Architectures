# Understanding Architectures

Project-driven AI/ML learning: one evolving language model trained on the Harry Potter books, rebuilt with a progressively better architecture at each milestone — bigram → MLP → RNN/LSTM → attention → Transformer → Tiny GPT → ...

Same data, same eval, only the mechanism changes. Each milestone is motivated by a concrete failure of the previous one, not by a fixed curriculum.

## Structure

- `data/` — raw text, shared across every milestone
- `common/data.py` — shared data loading, vocab, train/val split (same for every architecture)
- `n_grams/`, `mlp/`, `rnn/`, ... — one folder per milestone, model-specific code only

## NLP Architectures — Progress

- [x] Bigram character-level LM — data loading, count matrix, generation, perplexity
- [x] Trigram extension (2-char context) — sparsity artifacts already visible
- [x] MLP language model (3-char context, learned embeddings)
- [x] RNN (unlimited context via hidden state)
- [x] LSTM (gated memory, fixes vanishing gradients on longer sequences)
- [x] Self-attention, standalone (no positional encoding yet - see finding below)
- [x] Positional encoding (learned, absolute) - fixes order-blindness but not perplexity yet, see finding below
- [ ] RoPE / ALiBi (modern positional encoding)
- [x] Transformer (full block: multi-head attention + feedforward + residuals + layer norm, 2 layers)
- [x] Tiny GPT (weight tying + LR warmup/cosine decay + top-k/temperature sampling) - code complete, training result pending

### Perplexity comparison (lower = better)

| Model         | Context                     | Params  | Val perplexity |
|---------------|-------------------------------|---------|-----------------|
| Bigram        | 1 char                        | 8,464   | 11.25           |
| Trigram       | 2 chars                       | 778,688 | 6.76            |
| MLP           | 3 chars (fixed)                | 19,612  | 5.27            |
| RNN           | unlimited (hidden state)       | 32,028  | 4.91            |
| LSTM          | unlimited (gated memory)       | 88,092  | 4.27            |
| Attention*    | 64 chars (no position info)    | 9,148   | 11.35           |
| + Positional**| 64 chars (position-aware)      | 11,196  | 11.75           |
| Transformer***| 64 chars (2 layers, 4 heads)   | 33,500  | 7.00            |
| Tiny GPT      | 64 chars (2 layers, 4 heads, tied weights) | 30,556 | *pending - training was still running when this was pushed, real number to follow* |

\* Self-attention without positional encoding scored barely above bigram, worse than every other model. Proven (not assumed) why: with a fixed final character, scrambling the entire preceding 19-character context produced a numerically identical predicted distribution (`torch.allclose` true, max diff ~1e-9) — the model is provably blind to character order.

\*\* Adding learned positional embeddings genuinely fixed the blindness (same scramble test now gives `torch.allclose` false, real prediction differences) - but perplexity got slightly *worse*, not better. Order-awareness alone isn't sufficient: this is still a single attention layer with no feedforward network, no residual connections, no layer norm, and only one attention head - components a real Transformer block needs to actually *use* positional information well, not just have access to it.

\*\*\* The full block (multi-head attention + FFN + residuals + layer norm) fixed that gap dramatically: 11.75 -> 7.00. But it still didn't beat LSTM (4.27) or even MLP (5.27) at this training budget (2,000 steps). The loss curve was still steadily decreasing with no plateau at step 2,000, unlike every other model which had flattened out - the honest read is this run is likely undertrained relative to its capacity, not fundamentally weaker. Transformers are well-documented to need more training steps/data than RNNs to reach their potential, partly because RNNs have sequence order built into how they compute, while Transformers have to learn it via position embeddings from scratch; a learning-rate warmup schedule (used in the original paper, not implemented here) would likely help too. Left as an open finding rather than a fixed result.
