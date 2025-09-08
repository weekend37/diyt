import re
import json

from pathlib import Path


def split_into_chapters(text: str) -> list[str]:
    chapter_pattern = re.compile(r'^\s*(CHAPTER\s+\w+)\s*$', re.MULTILINE)  # Matches "CHAPTER <whildcard>"
    matches = list(chapter_pattern.finditer(text))
    start_indices = [match.start() for match in matches]
    end_indices = start_indices[1:] + [len(text)]
    chapters = [text[i:j].strip() for i, j in zip(start_indices, end_indices)]
    return chapters


def split_into_chunks(text: str) -> str:
    text = text.replace("\n\n\n", "\n\n")
    chunks = text.split("\n\n")
    chunks = chunks[2:]  # skip chapter number and title
    chunks = [chunk.strip() for chunk in chunks]
    return chunks


if __name__ == "__main__":

    current_directory = Path(__file__).parent
    with open(current_directory / "corpus.txt", encoding="utf8") as f:
        corpus = f.read()

    chapters = split_into_chapters(text=corpus)
    chunks = []
    for chapter in chapters:
        chapter_chunks = split_into_chunks(chapter)
        chunks.extend(chapter_chunks)

    train_proportion = 0.2
    n_train = round(len(chunks) * train_proportion)

    with open(current_directory / "train.json", "w", encoding="utf-8") as f:
        json.dump(chunks[:n_train], f, ensure_ascii=False, indent=4)
