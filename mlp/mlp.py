import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

CONTEXT_LENGTH = 3


def build_dataset(text, char_to_idx, context_length):
    indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)
    n = len(indices) - context_length

    # X[:, i] = indices shifted by i positions; stacking context_length of
    # these shifted slices as columns gives, per row, the context_length
    # characters immediately before the target - no Python loop needed
    X = np.stack([indices[i:i + n] for i in range(context_length)], axis=1)
    Y = indices[context_length:context_length + n]

    return torch.tensor(X), torch.tensor(Y)


X_train, Y_train = build_dataset(train_text, char_to_idx, CONTEXT_LENGTH)
X_val, Y_val = build_dataset(val_text, char_to_idx, CONTEXT_LENGTH)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

# sanity check: print the first example in human-readable form
context_chars = [idx_to_char[i.item()] for i in X_train[0]]
target_char = idx_to_char[Y_train[0].item()]
print("first example context:", context_chars, "-> target:", repr(target_char))

EMBED_DIM = 16
HIDDEN_DIM = 128


class MLP(nn.Module):
    def __init__(self, vocab_size, context_length, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # (vocab_size, embed_dim) lookup table
        self.hidden = nn.Linear(context_length * embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch, context_length) - a batch of contexts, each context_length char-ids
        emb = self.embedding(x)  # -> (batch, context_length, embed_dim)
        emb = emb.view(emb.shape[0], -1)  # flatten last 2 dims -> (batch, context_length * embed_dim)
        h = torch.tanh(self.hidden(emb))  # -> (batch, hidden_dim)
        logits = self.output(h)  # -> (batch, vocab_size), raw scores per next-char
        return logits


model = MLP(vocab_size, CONTEXT_LENGTH, EMBED_DIM, HIDDEN_DIM)

# sanity check: run one batch through and confirm the output shape
sample_logits = model(X_train[:5])
print("sample output logits shape:", sample_logits.shape)
print("total trainable parameters:", sum(p.numel() for p in model.parameters()))

LEARNING_RATE = 0.01
BATCH_SIZE = 4096
NUM_STEPS = 5000

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for step in range(NUM_STEPS):
    # random mini-batch: sample BATCH_SIZE random rows from the 6M examples
    batch_idx = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    Xb, Yb = X_train[batch_idx], Y_train[batch_idx]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)  # softmax + negative log likelihood, in one step

    optimizer.zero_grad()  # clear old gradients (they'd otherwise accumulate)
    loss.backward()        # compute how much each parameter contributed to the loss
    optimizer.step()       # nudge every parameter to reduce the loss

    losses.append(loss.item())
    if step % 500 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

plt.plot(losses)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("MLP training loss")
plt.savefig("mlp/loss_curve.png")
print("saved loss curve to mlp/loss_curve.png")


@torch.no_grad()
def perplexity(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)  # mean negative log likelihood, natural log
    return torch.exp(loss).item()


@torch.no_grad()
def generate(model, start_context, length):
    context = [char_to_idx[ch] for ch in start_context]
    result = list(start_context)
    for _ in range(length - len(start_context)):
        x = torch.tensor([context[-CONTEXT_LENGTH:]])
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        result.append(idx_to_char[next_idx])
        context.append(next_idx)
    return "".join(result)


train_perplexity = perplexity(model, X_train, Y_train)
val_perplexity = perplexity(model, X_val, Y_val)
print("train perplexity:", train_perplexity)
print("val perplexity:  ", val_perplexity)

print(generate(model, "The", 300))
