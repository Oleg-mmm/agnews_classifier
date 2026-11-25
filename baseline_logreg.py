import time
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


dataset = load_dataset("fancyzhx/ag_news")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

X_train, Y_train = train_df["text"], train_df['label']
X_test, Y_test = test_df['text'], test_df['label']

target_names = ["World", "Sports", "Business", "Sci/Tech"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        stop_words="english" 
    )),
    ("clf", LogisticRegression(
        C=2.0, 
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])

start_time = time.perf_counter()
pipeline.fit(X_train, Y_train)
train_time = time.perf_counter() - start_time
print(f"\nВремя обучения: {train_time:.2f} сек")

y_pred = pipeline.predict(X_test)

print(f"\nAccuracy на тесте: {accuracy_score(Y_test, y_pred):.4f}\n")
print("Подробный отчет по классам:")
print(classification_report(Y_test, y_pred, target_names=target_names, digits=4))