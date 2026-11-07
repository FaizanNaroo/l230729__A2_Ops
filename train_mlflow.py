import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without opening windows
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from pipeline import make_dataset, IMG_SIZE, BATCH
from models import build_cnn, build_transfer, compile_model

SEED = 42
p = argparse.ArgumentParser()
p.add_argument("--backbone", choices=["cnn", "mobilenet", "efficientnet"], default="mobilenet")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--unfrozen", type=int, default=0)
p.add_argument("--augment", type=int, choices=[0, 1], default=1)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
p.add_argument("--run_name", default=None)
args = p.parse_args()

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("visionguard")
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def dvc_data_version():
    with open("data/raw.dvc") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]

def pick_threshold(y, prob):
    ts = np.arange(0.05, 0.96, 0.01)
    scores = [fbeta_score(y, prob >= t, beta=0.5, zero_division=0) for t in ts]
    i = int(np.argmax(scores))
    return float(ts[i]), float(scores[i])

class MLflowLogger(tf.keras.callbacks.Callback):
    """Manual approach: log every Keras metric at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)

train_ds = make_dataset("data/splits/train.csv", training=True, use_augment=bool(args.augment))
val_ds = make_dataset("data/splits/val.csv")
test_ds = make_dataset("data/splits/test.csv")
val_y = pd.read_csv("data/splits/val.csv")["label"].values
test_y = pd.read_csv("data/splits/test.csv")["label"].values

if args.backbone == "cnn":
    model = build_cnn(args.dropout)
else:
    model = build_transfer(args.backbone, args.dropout, args.unfrozen)
compile_model(model, args.lr)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                     restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=3, min_lr=1e-7),
]

if args.tracking == "autolog":
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

with mlflow.start_run(run_name=args.run_name):
    mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
                     "seed": SEED, "tracking": args.tracking})

    if args.tracking == "manual":
        mlflow.log_params({"backbone": args.backbone, "lr": args.lr,
                           "dropout": args.dropout, "unfrozen_layers": args.unfrozen,
                           "augment": args.augment, "max_epochs": args.epochs,
                           "patience": args.patience, "img_size": IMG_SIZE,
                           "batch_size": BATCH, "optimizer": "adam",
                           "loss": "binary_crossentropy", "seed": SEED})
        callbacks.append(MLflowLogger())

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=callbacks, verbose=2)

    if args.tracking == "manual":
        # Choose the threshold on validation, then evaluate on test
        p_val = model.predict(val_ds, verbose=0).ravel()
        t, val_f05 = pick_threshold(val_y, p_val)
        p_test = model.predict(test_ds, verbose=0).ravel()
        pred = (p_test >= t).astype(int)

        mlflow.log_metrics({
            "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
            "best_val_loss": float(min(hist.history["val_loss"])),
            "val_roc_auc": roc_auc_score(val_y, p_val),
            "val_f05": val_f05,
            "threshold": t,
            "test_accuracy": accuracy_score(test_y, pred),
            "test_precision": precision_score(test_y, pred, zero_division=0),
            "test_recall": recall_score(test_y, pred),
            "test_f1": f1_score(test_y, pred),
            "test_roc_auc": roc_auc_score(test_y, p_test),
        })

        # Task 3.4 artifacts: confusion matrix, ROC curve, split lists, packages, model
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix(test_y, pred),
                               display_labels=["Real", "AI"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Test confusion matrix (t={t:.2f})")
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        fpr, tpr, _ = roc_curve(test_y, p_test)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(test_y, p_test):.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="Test ROC curve")
        ax.legend()
        mlflow.log_figure(fig, "plots/roc_curve.png")
        plt.close(fig)

        mlflow.log_artifacts("data/splits", artifact_path="splits")
        with tempfile.TemporaryDirectory() as d:
            req = Path(d) / "requirements.txt"
            req.write_text(subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True))
            mlflow.log_artifact(str(req), artifact_path="environment")

        example = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.float32)
        signature = infer_signature(example, model.predict(example, verbose=0))
        mlflow.tensorflow.log_model(model, name="model", signature=signature)

        print(f"Done: threshold {t:.2f}, best_val_loss {min(hist.history['val_loss']):.4f}")