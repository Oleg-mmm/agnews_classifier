MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "ag_news"
NUM_LABELS = 4
ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

MAX_LENGTH = 64
LR = 1e-3
EPOCHS = 12
TRAIN_BS = 32
EVAL_BS = 64
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0.06
LOGGING_STEPS = 100
FP16 = True
SEED = 123

OUTPUT_DIR = "./distilbert_agnews"
FINAL_DIR = "./distilbert_agnews_final"
DEBUG_SUBSET = 0