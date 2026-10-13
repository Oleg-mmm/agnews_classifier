import torch
import torch.nn as nn

from text_cls import PAD_INDX


def collate_fn(batch):
    seqs, labels = zip(*batch)
    length = torch.tensor([len(s) for s in seqs])
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_INDX)
    max_len = padded.size(1)
    mask = (torch.arange(max_len).unsqueeze(0) < length.unsqueeze(1)).long()
    labels = torch.stack(labels)
    return padded, mask, labels


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_indx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_indx)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True) # BiLSTM
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(2 * embed_dim, num_classes)

        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            self.embedding.weight[pad_indx].zero_()

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, mask):
        emb = self.embedding(x)

        h_all, _ = self.lstm(emb)

        lengths = mask.sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(x.size(0), device=x.device)

        H = h_all.size(-1) // 2

        last_fwd = h_all[batch_idx, lengths - 1, :H]
        first_bwd = h_all[batch_idx, 0, H:]

        h = torch.cat([last_fwd, first_bwd], dim=-1)

        h = self.dropout(h)
        logits = self.classifier(h)
        return logits