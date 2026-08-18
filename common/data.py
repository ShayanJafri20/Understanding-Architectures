import os


def load_books(data_dir="data/harry_potter_txt"):
    book_files = sorted(os.listdir(data_dir))
    books = []
    for fname in book_files:
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            books.append(f.read())
    return books


def load_data(data_dir="data/harry_potter_txt", val_fraction=0.1):
    """Shared across every milestone: same text, same split, same vocab."""
    books = load_books(data_dir)
    full_text = "".join(books)

    split_idx = int(len(full_text) * (1 - val_fraction))
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    # vocab built from train_text ONLY, not full_text - val must stay "unseen"
    vocab = sorted(set(train_text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    return {
        "train_text": train_text,
        "val_text": val_text,
        "vocab": vocab,
        "vocab_size": len(vocab),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }
