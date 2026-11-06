from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("audit.csv")
df = df[~df["corrupt"]].drop_duplicates("sha256")      # remove the duplicate copy
df["label"] = (df["class"] == "AiArtData").astype(int)  # 1 = AI, 0 = real
df["path"] = df["path"].str.replace("\\", "/")          # works on Linux and Docker too

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

out = Path("data/splits")
out.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label"]].to_csv(out / f"{name}.csv", index=False)
    print(name, len(part), part["label"].value_counts().to_dict())