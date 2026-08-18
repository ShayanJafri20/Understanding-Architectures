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

SEQ_LEN = 64


def build_sequences(text, char_to_idx, seq_len):
    indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)

    num_sequences = (len(indices) - 1) // seq_len
    indices = indices[: num_sequences * seq_len + 1]

    X = indices[:-1].reshape(num_sequences, seq_len)
    Y = indices[1:].reshape(num_sequences, seq_len)

    return torch.tensor(X), torch.tensor(Y)


X_train, Y_train = build_sequences(train_text, char_to_idx, SEQ_LEN)
X_val, Y_val = build_sequences(val_text, char_to_idx, SEQ_LEN)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

EMBED_DIM = 32


class SelfAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

        # causal mask, computed once - True = "block this position" (future position)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)  # travels with the model, but not a trained parameter

    def forward(self, x):
        # x shape: (batch, seq_len)
        seq_len = x.shape[1]
        emb = self.embedding(x)  # -> (batch, seq_len, embed_dim)

        Q = self.query(emb)
        K = self.key(emb)
        V = self.value(emb)

        scores = Q @ K.transpose(-2, -1) / (self.embed_dim ** 0.5)  # (batch, seq_len, seq_len)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len) - attention pattern
        attended = weights @ V  # (batch, seq_len, embed_dim) - weighted mix of Values

        logits = self.output(attended)  # -> (batch, seq_len, vocab_size)
        return logits, weights


model = SelfAttention(vocab_size, EMBED_DIM, SEQ_LEN)

sample_logits, sample_weights = model(X_train[:5])
print("sample output logits shape:", sample_logits.shape)
print("sample attention weights shape:", sample_weights.shape)
print("total trainable parameters:", sum(p.numel() for p in model.parameters()))

LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_STEPS = 2000

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for step in range(NUM_STEPS):
    batch_idx = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    Xb, Yb = X_train[batch_idx], Y_train[batch_idx]

    logits, _ = model(Xb)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Yb.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

plt.figure()
plt.plot(losses)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("Self-attention training loss")
plt.savefig("attention/loss_curve.png")
print("saved loss curve to attention/loss_curve.png")


@torch.no_grad()
def perplexity(model, X, Y):
    logits, _ = model(X)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
    return torch.exp(loss).item()


@torch.no_grad()
def generate(model, start_text, length):
    # unlike RNN/LSTM, there's no persistent hidden state to carry forward -
    # every step re-runs full attention over whatever context we have so far,
    # capped at SEQ_LEN since that's what the causal mask was built for
    context = [char_to_idx[ch] for ch in start_text]
    result = list(start_text)
    for _ in range(length - len(start_text)):
        x = torch.tensor([context[-SEQ_LEN:]])
        logits, _ = model(x)
        next_logits = logits[0, -1]
        probs = F.softmax(next_logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        result.append(idx_to_char[next_idx])
        context.append(next_idx)
    return "".join(result)


train_perplexity = perplexity(model, X_train, Y_train)
val_perplexity = perplexity(model, X_val, Y_val)
print("train perplexity:", train_perplexity)
print("val perplexity:  ", val_perplexity)

print(generate(model, "The", 300))

# visualize a real trained attention pattern on an actual sequence
sample_x = X_train[0:1]
_, trained_weights = model(sample_x)
sample_chars = [idx_to_char[i.item()] for i in sample_x[0]]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(trained_weights[0].detach().numpy(), cmap="viridis")
ax.set_xticks(range(SEQ_LEN)); ax.set_xticklabels(sample_chars, fontsize=6, rotation=90)
ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(sample_chars, fontsize=6)
ax.set_title("trained attention weights (real sequence)")
plt.colorbar(im)
plt.savefig("attention/attention_heatmap.png", bbox_inches="tight")
print("saved attention heatmap to attention/attention_heatmap.png")
