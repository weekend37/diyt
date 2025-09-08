import torch


class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        context_length: int,  # C
        input_dim: int,  # I
        hidden_dim: int,  # H
        output_dim: int,  # O
        n_heads: int,  # N
        dropout_ratio: float = 0.1
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.attention_normalizer = hidden_dim ** 0.5
        causal_mask = torch.tril(torch.ones((1, 1, context_length, context_length), dtype=torch.bool))  # [1, 1, C, C]
        self.register_buffer("causal_mask", causal_mask)  # for device management and saving/loading weights

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.W_Q = torch.nn.Linear(input_dim, hidden_dim * n_heads)  # [I, N * H]
        self.W_K = torch.nn.Linear(input_dim, hidden_dim * n_heads)  # [I, N * H]
        self.W_V = torch.nn.Linear(input_dim, hidden_dim * n_heads)  # [I, N * H]
        self.W = torch.nn.Linear(hidden_dim * n_heads, output_dim)  # [N * H, O]
        self.dropout = torch.nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        B, C, _ = x.shape
        assert attention_mask.shape == (B, C), f"x: ({B}, {C}), attention_mask: ({attention_mask.shape})"
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]

        x = self.layer_norm(x)
        # This could be vectorized ...
        Q = self.W_Q(x)  # [B, C, N * H]
        K = self.W_K(x)  # [B, C, N * H]
        V = self.W_V(x)  # [B, C, N * H]
        Q = Q.view(B, C, self.n_heads, self.hidden_dim).transpose(-2, -3)  # [B, N, C, H]
        K = K.view(B, C, self.n_heads, self.hidden_dim).transpose(-2, -3)  # [B, N, C, H]
        V = V.view(B, C, self.n_heads, self.hidden_dim).transpose(-2, -3)  # [B, N, C, H]
        attention_scores = Q @ K.transpose(-2, -1)  # [B, N, C, H] @ [B, N, H, C] -> [B, N, C, C]
        normalized_attention_scores = attention_scores / self.attention_normalizer
        mask = self.causal_mask & attention_mask  # [1, 1, C, C] & [B, 1, 1, C] -> [B, 1, C, C]
        normalized_attention_scores_masked = normalized_attention_scores.masked_fill(~mask, float('-inf'))
        A = torch.softmax(normalized_attention_scores_masked, dim=-1)  # [B, N, C, C]
        AV = A @ V  # [B, N, C, C] @ [B, N, C, H] -> [B, N, C, H]
        AV_t = AV.transpose(-2, -3).reshape(B, C, self.n_heads * self.hidden_dim)  # [B, C, N * H]
        Z = self.W(AV_t)  # [B, C, N * H] @ [N * H, O] [B, C, O]
        return self.dropout(Z)
