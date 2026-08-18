import os
import sys
import math
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
    n = (len(indices) - 1) // seq_len
    indices = indices[: n * seq_len + 1]
    X = indices[:-1].reshape(n, seq_len)
    Y = indices[1:].reshape(n, seq_len)
    return torch.tensor(X), torch.tensor(Y)


X_train, Y_train = build_sequences(train_text, char_to_idx, SEQ_LEN)
X_val, Y_val = build_sequences(val_text, char_to_idx, SEQ_LEN)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

EMBED_DIM = 32
NUM_HEADS = 4
FF_HIDDEN_DIM = EMBED_DIM * 4
NUM_LAYERS = 2


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ V

        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, ff_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, ff_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, seq_len, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

        # weight tying: reuse the SAME parameter tensor for token embedding and
        # output projection - both are (vocab_size, embed_dim), so this is valid.
        # "how do I represent this character" and "how do I predict this
        # character" share the same underlying knowledge
        self.output.weight = self.embedding.weight

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        h = self.embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        logits = self.output(h)
        return logits


model = TinyGPT(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN, FF_HIDDEN_DIM)

sample_logits = model(X_train[:5])
print("sample output logits shape:", sample_logits.shape)
print("total trainable parameters:", sum(p.numel() for p in model.parameters()))
print("embedding and output share weights:", model.embedding.weight is model.output.weight)

MAX_LR = 0.001
WARMUP_STEPS = 200
BATCH_SIZE = 64
NUM_STEPS = 2000


def get_lr(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps  # linear ramp up from ~0 to max_lr
    # cosine decay from max_lr down toward 0 over the remaining steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)

losses = []
lrs = []
for step in range(NUM_STEPS):
    lr = get_lr(step, WARMUP_STEPS, NUM_STEPS, MAX_LR)
    for group in optimizer.param_groups:
        group["lr"] = lr

    batch_idx = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    Xb, Yb = X_train[batch_idx], Y_train[batch_idx]

    logits = model(Xb)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Yb.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    lrs.append(lr)
    if step % 200 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(losses)
ax1.set_xlabel("training step"); ax1.set_ylabel("loss"); ax1.set_title("Tiny GPT training loss")
ax2.plot(lrs)
ax2.set_xlabel("training step"); ax2.set_ylabel("learning rate"); ax2.set_title("LR schedule (warmup + cosine decay)")
plt.tight_layout()
plt.savefig("gpt/loss_curve.png")
print("saved loss curve to gpt/loss_curve.png")


@torch.no_grad()
def perplexity(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
    return torch.exp(loss).item()


@torch.no_grad()
def generate(model, start_text, length, temperature=0.8, top_k=20):
    context = [char_to_idx[ch] for ch in start_text]
    result = list(start_text)
    for _ in range(length - len(start_text)):
        x = torch.tensor([context[-SEQ_LEN:]])
        logits = model(x)
        next_logits = logits[0, -1] / temperature  # >1 = more random, <1 = more confident/greedy

        if top_k is not None:
            top_vals, top_idx = torch.topk(next_logits, top_k)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered[top_idx] = top_vals
            next_logits = filtered  # only the k most likely characters are even eligible

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

torch.save(model.state_dict(), "gpt/tiny_gpt.pt")
print("saved model weights to gpt/tiny_gpt.pt")
