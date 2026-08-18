import os
import sys
import numpy as np

# bigram/ and common/ are sibling folders - add the project root to sys.path
# so "from common.data import load_data" can find it regardless of how this
# script is run
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load_data

# Windows' terminal defaults to cp1252, which can't display every Unicode
# character (e.g. some OCR artifacts in the source text). Reconfiguring
# stdout to UTF-8 is the output-side counterpart to encoding="utf-8" used
# when reading the files in common/data.py
sys.stdout.reconfigure(encoding="utf-8")

data = load_data()
train_text = data["train_text"]
val_text = data["val_text"]
vocab = data["vocab"]
vocab_size = data["vocab_size"]
char_to_idx = data["char_to_idx"]
idx_to_char = data["idx_to_char"]

print("train:", len(train_text), "chars")
print("val:  ", len(val_text), "chars")
print("vocab size:", vocab_size)

# encode every character to its integer id (still a Python loop, but cheap:
# one dict lookup per char, no numpy indexing yet)
indices = np.array([char_to_idx[ch] for ch in train_text], dtype=np.int64)

current = indices[:-1]  # indices[i]   for pairs (train_text[i], train_text[i+1])
next_ = indices[1:]     # indices[i+1] for the same pairs, shifted by one

# flatten each (current, next) pair into a single integer, the same way a
# 2D array is laid out in memory: row-major position = row * num_cols + col
flat_idx = current * vocab_size + next_

# bincount tallies how many times each integer value occurs - one vectorized
# C-level pass instead of millions of Python-level += 1 calls
counts_flat = np.bincount(flat_idx, minlength=vocab_size * vocab_size)
count_matrix = counts_flat.reshape(vocab_size, vocab_size)

print("count matrix shape:", count_matrix.shape)
print("total pairs counted:", count_matrix.sum())

# add-one (Laplace) smoothing: pretend every pair was seen 1 extra time,
# so no probability ever ends up exactly 0 (log(0) would break perplexity later)
smoothed_counts = count_matrix + 1

# sum each row to get the new row totals; keepdims=True keeps the result as
# a column so it can divide the matrix row-by-row without extra reshaping
row_sums = smoothed_counts.sum(axis=1, keepdims=True)

probs = smoothed_counts / row_sums

print("probs shape:", probs.shape)
print("each row sums to 1:", np.allclose(probs.sum(axis=1), 1.0))


def generate(start_char, length):
    result = [start_char]
    current = start_char
    for _ in range(length - 1):
        row_idx = char_to_idx[current]
        # pick one of the vocab_size next-character ids, weighted by probs[row_idx]
        # (NOT uniformly random - characters with higher probability get picked more often)
        next_idx = np.random.choice(vocab_size, p=probs[row_idx])
        current = idx_to_char[next_idx]
        result.append(current)
    return "".join(result)


print(generate("T", 300))


def perplexity(text, probs, char_to_idx):
    log_probs = []
    for i in range(len(text) - 1):
        curr_idx = char_to_idx[text[i]]
        next_idx = char_to_idx[text[i + 1]]
        p = probs[curr_idx, next_idx]
        log_probs.append(np.log(p))  # log of the probability the model gave this real pair

    avg_neg_log_prob = -np.mean(log_probs)  # average "surprise" per character
    return np.exp(avg_neg_log_prob)  # undo the log to get back to a readable scale


val_perplexity = perplexity(val_text, probs, char_to_idx)
print("val perplexity:", val_perplexity)
