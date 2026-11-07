from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from pipeline import make_dataset

MODELS = {"cnn": "models/cnn.keras", "mobilenet": "models/mobilenet_stage1.keras"}
val_df = pd.read_csv("data/splits/val.csv")
test_df = pd.read_csv("data/splits/test.csv")
val_ds = make_dataset("data/splits/val.csv")    # no shuffle, so order matches the CSV
test_ds = make_dataset("data/splits/test.csv")
fig_dir = Path("reports/figures")

def pick_threshold(y, p):
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [fbeta_score(y, p >= t, beta=0.5, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])

rows = []
fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4))

for ax_cm, (name, path) in zip(axes_cm, MODELS.items()):
    model = tf.keras.models.load_model(path)

    # 1. Choose the threshold on VALIDATION data
    p_val = model.predict(val_ds, verbose=0).ravel()
    t = pick_threshold(val_df["label"].values, p_val)

    # 2. Evaluate on TEST data once
    y = test_df["label"].values
    p_test = model.predict(test_ds, verbose=0).ravel()
    pred = (p_test >= t).astype(int)
    auc = roc_auc_score(y, p_test)
    rows.append({"model": name, "threshold": round(t, 2),
                 "accuracy": accuracy_score(y, pred),
                 "precision": precision_score(y, pred, zero_division=0),
                 "recall": recall_score(y, pred),
                 "f1": f1_score(y, pred),
                 "roc_auc": auc})

    fpr, tpr, _ = roc_curve(y, p_test)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ConfusionMatrixDisplay(confusion_matrix(y, pred),
                           display_labels=["Real", "AI"]).plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{name} (threshold {t:.2f})")

    out = test_df.copy()
    out["prob_ai"], out["pred"] = p_test, pred
    out.to_csv(f"reports/{name}_test_predictions.csv", index=False)  # used in Task 2.6

ax_roc.plot([0, 1], [0, 1], "k--", label="random guess")
ax_roc.set(xlabel="False positive rate (real called AI)",
           ylabel="True positive rate (AI caught)", title="ROC curve (test set)")
ax_roc.legend()
fig_roc.tight_layout()
fig_roc.savefig(fig_dir / "roc_curves.png", dpi=150)
fig_cm.tight_layout()
fig_cm.savefig(fig_dir / "confusion_matrices.png", dpi=150)

table = pd.DataFrame(rows).round(3)
table.to_csv("reports/test_metrics.csv", index=False)
print(table.to_string(index=False))