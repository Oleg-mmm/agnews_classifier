import random

from datasets import load_dataset

from text_cls import PAD_INDX, tokenizer, Vocab
from models.bof import BagOfEmbedding
from models.rnn import RNNClassifier
from models.birnn import BiRNNClassifier
from models.lstm import LSTMClassifier


NUM_CLASSES = 4
MIN_FQ = 2


def build_vocab():
    random.seed(123)
    raw_train = load_dataset("fancyzhx/ag_news", split="train")
    train_data = [(row["text"], row["label"]) for row in raw_train]

    all_tokens = []
    for text, _ in train_data:
        all_tokens.extend(tokenizer(text))

    vocab = Vocab(all_tokens, min_fq=MIN_FQ)
    random.shuffle(train_data)
    split = int(0.9 * len(train_data))
    train_data = train_data[:split]

    all_tokens = []
    for text, _ in train_data:
        all_tokens.extend(tokenizer(text))
    vocab = Vocab(all_tokens, min_fq=MIN_FQ)
    return vocab


def count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    vocab = build_vocab()
    V = len(vocab)
    print(f"vocab_size = {V}\n")

    models = {
        "BoF (E=128)":              BagOfEmbedding(V, 128, NUM_CLASSES, PAD_INDX),
        "RNN (E=128)":              RNNClassifier(V, 128, NUM_CLASSES, PAD_INDX),
        "BiRNN (E=64)":             BiRNNClassifier(V, 64, NUM_CLASSES, PAD_INDX),
        "BiLSTM (E=64)":            LSTMClassifier(V, 64, NUM_CLASSES, PAD_INDX),
    }

    for name, model in models.items():
        print(f"{name:<20} {count(model):>12,}")


if __name__ == "__main__":
    main()