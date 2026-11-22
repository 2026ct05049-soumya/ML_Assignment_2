"""Run the Adult income experiment and save the holdout, models, and metrics."""
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from model.classifiers import SEED, build_models

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
TARGET = "income"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", TARGET,
]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def load_data():
    parts = [
        pd.read_csv(ROOT / "data" / name, names=COLUMNS, skipinitialspace=True, na_values="?", comment="|")
        for name in ("adult.data", "adult.test")
    ]
    data = pd.concat(parts, ignore_index=True)
    labels = data[TARGET].str.rstrip(".")
    if not labels.isin(["<=50K", ">50K"]).all():
        raise ValueError("Unexpected or missing income label in the source data.")
    return data.assign(income=labels)


def save_chart(results, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    colors = ["#235789", "#20796c", "#8a507a", "#b05528", "#4d6740"]
    for metric, ax in zip(METRICS, axes.flat):
        bars = ax.barh(results.index, results[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlim(-1 if metric == "MCC" else 0, 1.12)
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("UCI Adult income classification on the 20% holdout", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    data = load_data()
    unique = data.drop_duplicates()
    train, test = train_test_split(unique, test_size=0.20, stratify=unique[TARGET], random_state=SEED)
    test.to_csv(ROOT / "test_data.csv", index=False)

    audit = {
        "raw_rows": len(data),
        "duplicates_removed": len(data) - len(unique),
        "missing_before_deduplication": data.isna().sum().to_dict(),
        "class_counts": {
            name: frame[TARGET].value_counts().to_dict()
            for name, frame in (("raw", data), ("deduplicated", unique), ("train", train), ("test", test))
        },
    }
    (results_dir / "dataset_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    x_train, x_test = (frame.drop(columns=TARGET) for frame in (train, test))
    y_train, y_test = ((frame[TARGET] == ">50K").astype(int) for frame in (train, test))
    rows = []
    with threadpool_limits(limits=4):
        for key, name, model in build_models():
            print(f"Training and evaluating {name}", flush=True)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)
            predictions = model.classes_[probabilities.argmax(axis=1)]
            positive_index = list(model.classes_).index(1)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, probabilities[:, positive_index]),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            })
            joblib.dump(model, ROOT / "model" / f"{key}.joblib", compress=3)
    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(results_dir / "metrics.csv", float_format="%.10f")
    save_chart(results, results_dir / "metrics.png")
    print(f"Training rows {len(train):,} | Test rows {len(test):,} | Seed {SEED}")
    print(results.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Overall winner by MCC: {results['MCC'].idxmax()}")


if __name__ == "__main__":
    main()
