"""
Metrics computation and comparison plots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)


def evaluate_model(model, X_test, y_test, average="binary"):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }
    return metrics, y_pred


def evaluate_all(fitted_models, X_test, y_test, average="binary"):
    results, predictions = {}, {}
    for name, model in fitted_models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, average=average)
        results[name] = metrics
        predictions[name] = y_pred
    return pd.DataFrame(results).T, predictions


def plot_model_comparison(results_df, out_path):
    ax = results_df.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(y_test, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
