import torch

from .self_attention import SelfAttention
from .decoder_block import DecoderBlock


class Transformer(torch.nn.Module):

    def __init__(
        self,
        context_length: int = 512,
        vocab_size: int = 10_000,
        token_embedding_size: int = 128,
        decoder_dim: int = 256,
        n_heads: int = 8,
        n_decoder_blocks: int = 4,
        dropout_ratio: float = 0.1
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.semantic_vocab_embeddings = torch.nn.Embedding(vocab_size, token_embedding_size)
        self.positional_embeddings = torch.nn.Embedding(context_length, token_embedding_size)
        self.register_buffer("position_ids", torch.arange(context_length).long())

        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(n_decoder_blocks):
            self_attention = SelfAttention(
                context_length=context_length,
                input_dim=token_embedding_size if i == 0 else decoder_dim,
                hidden_dim=decoder_dim,
                output_dim=decoder_dim,
                n_heads=n_heads,
                dropout_ratio=dropout_ratio
            )
            decoder_block = DecoderBlock(
                self_attention=self_attention,
                feed_forward_hidden_dim=decoder_dim,
                dropout_ratio=dropout_ratio
            )
            self.decoder_blocks.append(decoder_block)
        self.layer_norm = torch.nn.LayerNorm(decoder_dim)
        self.linear = torch.nn.Linear(decoder_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, C = token_ids.shape
        x_semantic = self.semantic_vocab_embeddings(token_ids)  # [B, C, I]
        batch_position_ids = self.position_ids.unsqueeze(0).expand(B, -1)  # [B, C]
        x_positional = self.positional_embeddings(batch_position_ids)  # [B, C, I]
        x = x_semantic + x_positional  # [B, C, I]
        for decoder in self.decoder_blocks:
            x = decoder(x, attention_mask=attention_mask)  # [B, C, O]
        x = self.linear(self.layer_norm(x))  # [B, C, V]
        return x
