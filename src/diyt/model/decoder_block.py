import torch

from .self_attention import SelfAttention


class DecoderBlock(torch.nn.Module):

    def __init__(
        self,
        self_attention: SelfAttention,
        feed_forward_hidden_dim: int,
        dropout_ratio: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = torch.nn.Sequential(
            torch.nn.LayerNorm(self_attention.output_dim),
            torch.nn.Linear(self_attention.output_dim, feed_forward_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(feed_forward_hidden_dim, self_attention.input_dim),
            torch.nn.Dropout(dropout_ratio)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(x, attention_mask)
        x = x + self.feed_forward(x)
        return x
