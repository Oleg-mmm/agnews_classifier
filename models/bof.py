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


class BagOfEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_indx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_indx)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            self.embedding.weight[0].zero_()
            
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x, mask):
        emb = self.embedding(x)
        mask_f = mask.unsqueeze(-1).float()
        emb = emb * mask_f
        summed = emb.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True)
        count = count.clamp(min=1)
        pooled = summed/count
        logits = self.classifier(pooled)
        return logits