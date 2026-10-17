import torch
import torch.nn as nn

from text_cls import PAD_INDX
from models.transformer.attention import PositionalEncoding
from models.transformer.encoder import EncoderBlock


def collate_fn(batch):
    seqs, labels = zip(*batch)
    length = torch.tensor([len(s) for s in seqs])
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_INDX)
    max_len = padded.size(1)
    mask = (torch.arange(max_len).unsqueeze(0) < length.unsqueeze(1)).long()
    labels = torch.stack(labels)
    return padded, mask, labels


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_indx=0, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_indx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=128, dropout=dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, hidden_mult=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            self.embedding.weight[pad_indx].zero_()

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, mask):
        h = self.embedding(x)
        h = self.pos_enc(h)

        for block in self.blocks:
            h = block(h, mask)

        h = self.norm(h)

        mask_f = mask.unsqueeze(-1).float()
        summed = (h * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1)
        pooled = summed / count

        pooled = self.dropout(pooled)
        return self.classifier(pooled)