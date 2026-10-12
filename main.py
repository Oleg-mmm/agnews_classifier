import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from text_cls import (
    PAD_INDX,
    EarlyStopping,
    tokenizer,
    Vocab,
    TextClsDataset,
    train_one_epoch,
    validate,
)
from models.birnn import BiRNNClassifier, collate_fn


def main():
    random.seed(123)

    raw_train = load_dataset("fancyzhx/ag_news", split="train")
    raw_test  = load_dataset("fancyzhx/ag_news", split="test")

    def hf_to_pairs(hf_split):
        return [(row["text"], row["label"]) for row in hf_split]

    train_data = hf_to_pairs(raw_train)
    test_data  = hf_to_pairs(raw_test)

    MIN_FQ = 2

    all_tokens = []
    for text, _ in train_data:
        all_tokens.extend(tokenizer(text))

    vocab = Vocab(all_tokens, min_fq=MIN_FQ)

    V = len(vocab)
    BATCH_SIZE = 64
    NUM_WORKERS = 0

    random.shuffle(train_data)

    split = int(0.9*len(train_data))
    train_data, val_data = train_data[:split], train_data[split:]

    train_dataset = TextClsDataset(train_data, vocab, tokenizer)
    val_dataset   = TextClsDataset(val_data,   vocab, tokenizer)
    test_dataset  = TextClsDataset(test_data,  vocab, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    NUM_CLASSES = 4   # AG News: World; Sports; Business; Sci-Tech
    EMBED_DIM = 128
    LR = 1e-3
    EPOCHS = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = BiRNNClassifier(V, EMBED_DIM, NUM_CLASSES, PAD_INDX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    loss_fn = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(f"epoch {epoch}: train loss  = {train_loss:.4f}; acc = {train_acc:.4f}")
        print(f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        print()
        
        improved = early_stopping(val_loss)
        if improved:
            torch.save(model.state_dict(), 'birnn_agnews.pt')
        if early_stopping.early_stop:
            print(f'early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load('birnn_agnews.pt', map_location=device))
    test_loss, test_acc = validate(model, test_loader, loss_fn, device)
    print(f"TEST: loss = {test_loss:.4f}; acc = {test_acc:.4f}")
    

if __name__ == "__main__":
    main()