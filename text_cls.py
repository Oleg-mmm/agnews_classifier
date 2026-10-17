from collections import Counter

import torch
from torch.utils.data import Dataset


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
    def __init__(self, data, vocab, tokenizer, max_len=None):
        super().__init__()
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.samples = []
        for text, label in data:
            ids = vocab.encode(tokenizer(text))
            if max_len is not None:
                ids = ids[:max_len]
            self.samples.append([ids, label])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indx):
        ids, label = self.samples[indx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
        
    return total_loss/total, correct/total


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


ID_LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci-Tech"}


@torch.no_grad()
def predict(text, model, vocab, device):
    model.eval()

    ids = vocab.encode(tokenizer(text))
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(x)

    logits = model(x, mask)
    probs = torch.softmax(logits, dim=-1)

    pred = probs.argmax(dim=-1).item()
    return ID_LABEL[pred], probs.squeeze(0).tolist()


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, cur_loss):
        if cur_loss < self.best_loss - self.delta:
            self.best_loss = cur_loss
            self.counter = 0
            return True # improved
            
        self.counter += 1
        if self.counter == self.patience:
            self.early_stop = True
            return False
        return False