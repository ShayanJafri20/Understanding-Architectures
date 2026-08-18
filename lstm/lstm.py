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
    indices = indices[: num_sequences * seq_len + 1]  # trim leftover tail

    X = indices[:-1].reshape(num_sequences, seq_len)  # chunk i:   chars [0..seq_len)
    Y = indices[1:].reshape(num_sequences, seq_len)   # chunk i+1: same chars shifted by 1

    return torch.tensor(X), torch.tensor(Y)


X_train, Y_train = build_sequences(train_text, char_to_idx, SEQ_LEN)
X_val, Y_val = build_sequences(val_text, char_to_idx, SEQ_LEN)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

EMBED_DIM = 16
HIDDEN_DIM = 128


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        emb = self.embedding(x)  # -> (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # hidden is (h, c) now, not a single tensor
        logits = self.output(out)  # -> (batch, seq_len, vocab_size)
        return logits, hidden


model = LSTMModel(vocab_size, EMBED_DIM, HIDDEN_DIM)

sample_logits, _ = model(X_train[:5])
print("sample output logits shape:", sample_logits.shape)
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

plt.plot(losses)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("LSTM training loss")
plt.savefig("lstm/loss_curve.png")
print("saved loss curve to lstm/loss_curve.png")


@torch.no_grad()
def perplexity(model, X, Y):
    logits, _ = model(X)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
    return torch.exp(loss).item()


@torch.no_grad()
def generate(model, start_text, length):
    result = list(start_text)

    x = torch.tensor([[char_to_idx[ch] for ch in start_text]])
    logits, hidden = model(x)
    next_logits = logits[0, -1]

    for _ in range(length - len(start_text)):
        probs = F.softmax(next_logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        result.append(idx_to_char[next_idx])

        x = torch.tensor([[next_idx]])
        logits, hidden = model(x, hidden)
        next_logits = logits[0, -1]

    return "".join(result)


train_perplexity = perplexity(model, X_train, Y_train)
val_perplexity = perplexity(model, X_val, Y_val)
print("train perplexity:", train_perplexity)
print("val perplexity:  ", val_perplexity)

print(generate(model, "The", 300))
