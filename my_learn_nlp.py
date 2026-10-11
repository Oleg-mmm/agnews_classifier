from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

PAD_INDX = 0


def tokenizer(str):
    return str.lower().split()


class Vocab:
    def __init__(self, tokens, min_fq=1):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        
        counter = Counter(tokens)
        itos = [self.pad_token, self.unk_token]
        for tok, freq in counter.most_common():
            if freq >= min_fq and tok not in itos:
                itos.append(tok)
        
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        
    def __len__(self):
        return len(self.itos)
    
    def encode(self, tokens):
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(t, unk) for t in tokens]
    
    def decode(self, indx):
        return [self.itos[i] for i in indx]
    

class TextClsDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        super().__init__()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.samples = []
        for text, label in data:
            ids = vocab.encode(tokenizer(text))
            self.samples.append([ids, label])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indx):
        ids, label = self.samples[indx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    
def collate_fn(batch):
    seqs, labels = zip(*batch)
    length = torch.tensor([len(s) for s in seqs])
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_INDX)
    max_len = padded.size(1)
    mask = (torch.arange(max_len).unsqueeze(0) < length.unsqueeze(1)).long()
    labels = torch.stack(labels)
    return padded, mask, labels