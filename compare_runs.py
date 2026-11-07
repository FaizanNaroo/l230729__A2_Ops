import os
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

runs = mlflow.search_runs(experiment_names=["visionguard"],
                          filter_string="tags.tracking = 'manual'")
cols = {"tags.mlflow.runName": "run", "params.backbone": "backbone",
        "params.lr": "lr", "params.dropout": "dropout", "params.augment": "augment",
        "params.unfrozen_layers": "unfrozen", "metrics.best_epoch": "best_epoch",
        "metrics.best_val_loss": "val_loss", "metrics.val_f05": "val_f05",
        "metrics.val_roc_auc": "val_auc", "metrics.test_accuracy": "test_acc",
        "metrics.test_precision": "test_prec", "metrics.test_recall": "test_rec",
        "metrics.test_f1": "test_f1", "metrics.test_roc_auc": "test_auc",
        "run_id": "run_id"}
table = runs[list(cols)].rename(columns=cols).sort_values("val_loss").round(4)
table.to_csv("reports/run_comparison.csv", index=False)
print(table.drop(columns="run_id").to_string(index=False))
print("\nBest run by validation loss:", table.iloc[0]["run"], table.iloc[0]["run_id"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(table["run"], table["val_loss"])
ax.invert_yaxis()  # best run on top
ax.set(xlabel="Best validation loss (lower is better)",
       title="VisionGuard experiment comparison")
plt.tight_layout()
plt.savefig("reports/figures/run_comparison.png", dpi=150)