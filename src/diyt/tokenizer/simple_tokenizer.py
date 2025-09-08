from __future__ import annotations

import json
import torch
import typing

from collections import Counter
from pathlib import Path
from pydantic import BaseModel, ConfigDict


class Encoded(BaseModel):
    token_ids: torch.Tensor
    attention_mask: torch.Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self) -> int:
        return len(self.token_ids)


class SimpleTokenizer:

    unknown_token = "<unk>"
    pad_token = "<pad>"
    end_of_string_token = "<eos>"
    period_token = "."

    def __init__(self, vocab: list[str] = None) -> None:
        self.special_tokens = [self.unknown_token, self.pad_token, self.end_of_string_token, self.period_token]
        self.vocab = vocab
        self.token_to_id: dict[str, int] = None
        if vocab is not None:
            for special_token in self.special_tokens:
                if special_token not in vocab:
                    vocab.insert(0, special_token)
            self.token_to_id = dict(zip(self.vocab, range(len(self.vocab))))

    @property
    def vocab_size(self) -> int | None:
        return len(self.vocab) if self.vocab is not None else None

    def fit(self, corpus: str, token_min_frequency: int = 1) -> None:
        corpus = corpus.replace(".", " .")
        corpus_tokens = corpus.split(" ")
        token_counts = Counter(corpus_tokens)
        filtered_tokens = [token for token, frequency in token_counts.items() if frequency >= token_min_frequency]
        unique_tokens = [token for token in list(set(filtered_tokens)) if token not in self.special_tokens]
        self.vocab = self.special_tokens + unique_tokens
        self.token_to_id = dict(zip(self.vocab, range(len(self.vocab))))

    def describe(self) -> dict[str, typing.Any]:
        description = {
            "special_tokens": self.special_tokens
        }
        if self.vocab is not None:
            description["num_tokens"] = len(self.vocab)
        return description

    def encode(self, sequence: str, max_seq_length: int | None = None) -> Encoded:
        sequence = sequence.replace(".", " .")
        tokens = [token for token in sequence.split(" ") if token not in [""]]
        padding = []
        if max_seq_length is not None:
            tokens = tokens[:max_seq_length]
            padding = [self.pad_token] * (max_seq_length - len(tokens))
            tokens += padding

        token_ids_list = [self.token_to_id.get(token, self.token_to_id[self.unknown_token]) for token in tokens]
        token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        attention_mask = torch.concat(
            (
                torch.ones(len(tokens) - len(padding), dtype=bool),
                torch.zeros(len(padding), dtype=bool)
            )
        )
        encoded = Encoded(token_ids=token_ids, attention_mask=attention_mask)
        return encoded

    def batch_encode(self, sequences: list[str], max_seq_length: int) -> Encoded:
        all_token_ids = []
        all_attention_masks = []
        for sequence in sequences:
            encoded = self.encode(sequence, max_seq_length=max_seq_length)
            all_token_ids.append(encoded.token_ids)
            all_attention_masks.append(encoded.attention_mask)
        batch_token_ids = torch.stack(all_token_ids)
        batch_attention_masks = torch.stack(all_attention_masks)
        batch_encoded = Encoded(token_ids=batch_token_ids, attention_mask=batch_attention_masks)
        return batch_encoded

    def decode(self, token_ids: list[int]) -> str:
        sequence = " ".join([self.vocab[token_id] for token_id in token_ids])
        sequence = sequence.replace(" .", ".")
        return sequence

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        return [self.decode(sequence_token_ids) for sequence_token_ids in token_ids]

    def save(self, save_path: Path) -> None:
        data = {"vocab": self.vocab}
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, save_path: Path) -> SimpleTokenizer:
        with open(save_path, "r") as f:
            data = json.load(f)
        return SimpleTokenizer(vocab=data["vocab"])

    __call__ = batch_encode
