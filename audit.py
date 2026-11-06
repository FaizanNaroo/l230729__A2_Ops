from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw")
rows = []

for path in DATA_DIR.rglob("*"):
    if not path.is_file():
        continue

    row = {
        "path": str(path),
        "class": path.relative_to(DATA_DIR).parts[0],
        "file_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "corrupt": False,
    }

    try:
        with Image.open(path) as img:
            img.load()  # fully decode the image
            row.update(width=img.width, height=img.height,
                       mode=img.mode, format=img.format)
    except Exception:
        row["corrupt"] = True

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("audit.csv", index=False)

print("Counts per class:\n", df["class"].value_counts())
print("\nStats per class:\n",
      df.groupby("class")[["width", "height", "file_size"]]
        .agg(["min", "median", "max"]).T)
print("\nFormats:\n", df["format"].value_counts(dropna=False))
print("\nColour modes:\n", df["mode"].value_counts(dropna=False))
print("\nCorrupt files:", df["corrupt"].sum())
print("Extra duplicate copies:", df.duplicated("sha256").sum())
print("Hashes found in both classes:",
      (df.groupby("sha256")["class"].nunique() > 1).sum())