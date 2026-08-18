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
- [ ] LSTM (fixes vanishing gradients on longer sequences)
- [ ] Attention (standalone)
- [ ] Transformer
- [ ] Tiny GPT

### Perplexity comparison (lower = better)

| Model    | Context             | Params  | Val perplexity |
|----------|----------------------|---------|-----------------|
| Bigram   | 1 char               | 8,464   | 11.25           |
| Trigram  | 2 chars              | 778,688 | 6.76            |
| MLP      | 3 chars (fixed)      | 19,612  | 5.27            |
| RNN      | unlimited (hidden state) | 32,028  | 4.91        |

Same data, same eval, every row — only the mechanism changes. MLP beats trigram with 40x fewer parameters by generalizing instead of memorizing exact contexts. RNN improves further by replacing the fixed context window with a hidden state that can in principle carry information across the whole sequence.
