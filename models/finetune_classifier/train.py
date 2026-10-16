import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

import config as cfg
from data import build_tokenizer, build_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    set_seed(cfg.SEED)

    tokenizer = build_tokenizer()
    ds = build_dataset(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        id2label=cfg.ID2LABEL,
        label2id=cfg.LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        learning_rate=cfg.LR,
        per_device_train_batch_size=cfg.TRAIN_BS,
        per_device_eval_batch_size=cfg.EVAL_BS,
        num_train_epochs=cfg.EPOCHS,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_steps=cfg.WARMUP_STEPS,
        logging_steps=cfg.LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=cfg.FP16,
        report_to="none",
        seed=cfg.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Final metrics:", trainer.evaluate())

    trainer.save_model(cfg.FINAL_DIR)
    tokenizer.save_pretrained(cfg.FINAL_DIR)
    print(f"Saved to {cfg.FINAL_DIR}")


if __name__ == "__main__":
    main()