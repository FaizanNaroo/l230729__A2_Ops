from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("audit.csv")
out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)
classes = sorted(df["class"].unique())

# Sample images: one row per class
n = 6
fig, axes = plt.subplots(len(classes), n, figsize=(15, 5))
for r, cls in enumerate(classes):
    sample = df[df["class"] == cls].sample(n, random_state=42)
    for c, path in enumerate(sample["path"]):
        axes[r, c].imshow(Image.open(path).convert("RGB"))
        axes[r, c].axis("off")
    axes[r, 0].set_title(cls, loc="left")
plt.tight_layout()
plt.savefig(out_dir / "sample_grid.png", dpi=150)
plt.close()

# Histograms: width, height and file size per class
columns = ["width", "height", "file_size"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, columns):
    bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 30)
    for cls in classes:
        ax.hist(df[df["class"] == cls][col], bins=bins, alpha=0.5, label=cls)
    ax.set_xscale("log")
    ax.set(title=col, xlabel=f"{col} (log scale)", ylabel="Images")
    ax.legend()
plt.tight_layout()
plt.savefig(out_dir / "histograms.png", dpi=150)
plt.close()

# Extra checks
print("Duplicate files:")
print(df[df.duplicated("sha256", keep=False)][["class", "path"]].to_string())
print("\nFormat per class:\n", pd.crosstab(df["class"], df["format"]))
print("\nColour mode per class:\n", pd.crosstab(df["class"], df["mode"]))