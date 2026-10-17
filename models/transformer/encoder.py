import torch.nn as nn

from models.transformer.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.mha(self.norm1(x), mask)
        x = x + self.dropout(h)

        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)

        return x