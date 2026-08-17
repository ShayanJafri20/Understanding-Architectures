import os

DATA_DIR = "data/harry_potter_txt"

book_files = sorted(os.listdir(DATA_DIR))

books = []
for fname in book_files:
    path = os.path.join(DATA_DIR, fname)
    print(fname)
    with open(path, "r", encoding="utf-8") as f:
        books.append(f.read())


for fname, text in zip(book_files, books):
    print(fname, len(text), "chars")
