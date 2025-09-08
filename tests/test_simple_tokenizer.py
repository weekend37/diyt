import pytest

from diyt.tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


def test_tokenizer(tokenizer: SimpleTokenizer) -> None:
    tokenizer.fit("This is the full corpus.")
    sequences = [
        "This is the",
        "This is the full",
        "This is the full corpus.",
        "This is new",
        ""
    ]
    expected_decoded_sequences = [
        "This is the <pad>",
        "This is the full",
        "This is the full",
        "This is <unk> <pad>",
        "<pad> <pad> <pad> <pad>"
    ]
    encoded_sequences = tokenizer.batch_encode(sequences, max_seq_length=4)
    decoded_sequences = tokenizer.batch_decode(encoded_sequences.token_ids)
    assert decoded_sequences == expected_decoded_sequences
