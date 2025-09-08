import json
from pathlib import Path

from tokenizer import SimpleTokenizer
from paths import ASSETS_DIR, DATA_DIR


if __name__ == "__main__":

    with open(DATA_DIR / "harry_potter" / "train.json", "r", encoding="utf-8") as f:
        texts = json.load(f)

    corpus = " ".join(texts)

    tokenizer = SimpleTokenizer()
    tokenizer.fit(corpus, token_min_frequency=1)
    print("Tokenizer ready")
    print(tokenizer.describe())

    save_path = Path(ASSETS_DIR / "hp" / "tokenizer.json")
    tokenizer.save(save_path)
    tokenizer = SimpleTokenizer.load(save_path)
    print(f"Tokenizer saved at {save_path}")
