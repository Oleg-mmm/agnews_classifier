from datasets import load_dataset
from transformers import AutoTokenizer

import config as cfg


def build_tokenizer():
    return AutoTokenizer.from_pretrained(cfg.MODEL_NAME)


def build_dataset(tokenizer):
    ds = load_dataset(cfg.DATASET_NAME)

    if cfg.DEBUG_SUBSET:
        ds["train"] = ds["train"].select(range(cfg.DEBUG_SUBSET))
        ds["test"] = ds["test"].select(range(cfg.DEBUG_SUBSET // 4))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.MAX_LENGTH)

    ds = ds.map(tokenize, batched=True, num_proc=2)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds