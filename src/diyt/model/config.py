from pydantic import BaseModel


class ModelConfig(BaseModel):
    context_length: int
    token_embedding_size: int
    decoder_dim: int
    n_heads: int
    n_decoder_blocks: int
    dropout_ratio: float
