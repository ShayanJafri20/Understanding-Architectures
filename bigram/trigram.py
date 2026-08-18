import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load_data

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

# encode every character to its integer id
indices = np.array([char_to_idx[ch] for ch in train_text], dtype=np.int64)

# context = 2 previous chars, so we need triplets: (ctx1, ctx2, next)
ctx1 = indices[:-2]
ctx2 = indices[1:-1]
next_ = indices[2:]

# flatten 3 dimensions into 1 flat index, same idea as bigram but one more axis:
# flat position = ctx1 * vocab_size^2 + ctx2 * vocab_size + next
flat_idx = ctx1 * (vocab_size ** 2) + ctx2 * vocab_size + next_

counts_flat = np.bincount(flat_idx, minlength=vocab_size ** 3)
count_matrix = counts_flat.reshape(vocab_size, vocab_size, vocab_size)

print("count matrix shape:", count_matrix.shape)
print("total triplets counted:", count_matrix.sum())

# add-one smoothing, same idea as bigram
smoothed_counts = count_matrix + 1

# sum over axis=2 (the "next char" axis) - for each (ctx1, ctx2) pair, get
# the total count across all possible next characters
row_sums = smoothed_counts.sum(axis=2, keepdims=True)

probs = smoothed_counts / row_sums

print("probs shape:", probs.shape)
print("each context row sums to 1:", np.allclose(probs.sum(axis=2), 1.0))


def generate(start_ctx, length):
    # start_ctx must be 2 characters - trigram needs 2 characters of context
    # before it can predict anything
    result = list(start_ctx)
    c1, c2 = start_ctx[0], start_ctx[1]
    for _ in range(length - 2):
        idx1 = char_to_idx[c1]
        idx2 = char_to_idx[c2]
        next_idx = np.random.choice(vocab_size, p=probs[idx1, idx2])
        next_char = idx_to_char[next_idx]
        result.append(next_char)
        c1, c2 = c2, next_char  # slide the 2-char window forward
    return "".join(result)


print(generate("Th", 300))


def perplexity(text, probs, char_to_idx):
    log_probs = []
    for i in range(len(text) - 2):
        idx1 = char_to_idx[text[i]]
        idx2 = char_to_idx[text[i + 1]]
        idx3 = char_to_idx[text[i + 2]]
        p = probs[idx1, idx2, idx3]
        log_probs.append(np.log(p))

    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)


val_perplexity = perplexity(val_text, probs, char_to_idx)
print("val perplexity:", val_perplexity)
