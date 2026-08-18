# Understanding Architectures

Project-driven AI/ML learning: one evolving language model trained on the Harry Potter books, rebuilt with a progressively better architecture at each milestone — bigram → MLP → RNN/LSTM → attention → Transformer → Tiny GPT → ...

Same data, same eval, only the mechanism changes. Each milestone is motivated by a concrete failure of the previous one, not by a fixed curriculum.

## Structure

- `data/` — raw text, shared across every milestone
- `common/data.py` — shared data loading, vocab, train/val split (same for every architecture)
- `bigram/`, `mlp/`, `rnn/`, ... — one folder per milestone, model-specific code only

## Progress

- [x] Bigram character-level LM — data loading, count matrix, generation, perplexity (val perplexity: 11.25)
- [ ] Trigram extension (2-char context)
- [ ] MLP language model
