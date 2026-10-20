import random
import torch
from datasets import load_dataset

from text_cls import PAD_INDX, tokenizer, Vocab
from models.bof import BagOfEmbedding
from models.rnn import RNNClassifier
from models.birnn import BiRNNClassifier
from models.lstm import LSTMClassifier
from models.transformer.transformer_model import TransformerClassifier


NUM_CLASSES = 4
MIN_FQ = 2


def build_vocab():
    raw_train = load_dataset("fancyzhx/ag_news", split="train")
    all_tokens = []
    for row in raw_train:
        all_tokens.extend(tokenizer(row["text"]))
    
    return Vocab(all_tokens, min_fq=MIN_FQ)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    vocab = build_vocab()
    V = len(vocab)
    
    print(f"Vocab Size: {V}\n")
    print(f"| {'Model Name':<25} | {'Parameters':>15} |")
    print(f"|{'-'*27}|{'-'*17}|")

    models = {
        "Bag of Embeddings": BagOfEmbedding(V, 128, NUM_CLASSES, PAD_INDX),
        "RNN (E=128)": RNNClassifier(V, 128, NUM_CLASSES, PAD_INDX),
        "BiRNN (E=64)": BiRNNClassifier(V, 64, NUM_CLASSES, PAD_INDX),
        "BiLSTM (E=64)": LSTMClassifier(V, 64, NUM_CLASSES, PAD_INDX),
        "Transformer (d=128)": TransformerClassifier(V, 128, NUM_CLASSES, PAD_INDX, num_heads=4, num_layers=2, dropout=0.2),
    }

    for name, model in models.items():
        params = count_parameters(model)
        print(f"| {name:<25} | {params:>15,} |")


if __name__ == "__main__":
    main()