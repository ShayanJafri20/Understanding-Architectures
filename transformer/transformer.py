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
HEAD_DIM = EMBED_DIM // NUM_HEADS


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
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.shape

        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)

        # split embed_dim into (num_heads, head_dim), move heads next to batch
        # so each head runs attention independently:
        # (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (batch, num_heads, seq_len, seq_len)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ V  # (batch, num_heads, seq_len, head_dim)

        # merge heads back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        return self.output(attended)


# shape-tracing sanity check before building the full model
mha = MultiHeadAttention(EMBED_DIM, NUM_HEADS, SEQ_LEN)
dummy_input = torch.randn(2, SEQ_LEN, EMBED_DIM)  # fake embeddings, batch=2
mha_out = mha(dummy_input)
print("multi-head attention input shape: ", dummy_input.shape)
print("multi-head attention output shape:", mha_out.shape, "(should match input shape)")

FF_HIDDEN_DIM = EMBED_DIM * 4


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)  # applied independently to every position - no mixing across positions


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, ff_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm residual around attention (mixes across positions)
        x = x + self.ff(self.ln2(x))    # pre-norm residual around feedforward (per-position computation)
        return x


# shape-tracing sanity check
block = TransformerBlock(EMBED_DIM, NUM_HEADS, SEQ_LEN, FF_HIDDEN_DIM)
block_out = block(dummy_input)
print("transformer block input shape: ", dummy_input.shape)
print("transformer block output shape:", block_out.shape, "(should match input shape)")
print("transformer block parameters:", sum(p.numel() for p in block.parameters()))

NUM_LAYERS = 2


class Transformer(nn.Module):
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

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        h = self.embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        logits = self.output(h)
        return logits


model = Transformer(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN, FF_HIDDEN_DIM)

sample_logits = model(X_train[:5])
print("sample output logits shape:", sample_logits.shape)
print("total trainable parameters:", sum(p.numel() for p in model.parameters()))

LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_STEPS = 2000

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for step in range(NUM_STEPS):
    batch_idx = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
    Xb, Yb = X_train[batch_idx], Y_train[batch_idx]

    logits = model(Xb)
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
plt.title("Transformer training loss")
plt.savefig("transformer/loss_curve.png")
print("saved loss curve to transformer/loss_curve.png")


@torch.no_grad()
def perplexity(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
    return torch.exp(loss).item()


@torch.no_grad()
def generate(model, start_text, length):
    context = [char_to_idx[ch] for ch in start_text]
    result = list(start_text)
    for _ in range(length - len(start_text)):
        x = torch.tensor([context[-SEQ_LEN:]])
        logits = model(x)
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
