import math
import torch
import torch.nn as nn


class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, att_mask):
        d_k = q.size(-1)
        
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        
        if att_mask is not None:
            scores = scores.masked_fill(att_mask[:, None, None, :]==0, -1e9)
        
        att = torch.softmax(scores, dim=-1)
        att = self.dropout(att)
        out = att @ v
        return out, att
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.att = ScaledDotAttention(dropout)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        out, _ = self.att(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)

        return self.W_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        return self.dropout(x + self.pos_emb(pos))