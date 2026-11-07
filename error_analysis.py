import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

pred = pd.read_csv("reports/mobilenet_test_predictions.csv")
audit = pd.read_csv("audit.csv")
audit["path"] = audit["path"].str.replace("\\", "/")
df = pred.merge(audit[["path", "width", "height", "format", "mode"]], on="path")

wrong = df[df["label"] != df["pred"]].copy()
wrong["error"] = (wrong["label"] - wrong["prob_ai"]).abs()  # how confidently wrong
fp = wrong[wrong["label"] == 0].nlargest(5, "error")  # real called AI
fn = wrong[wrong["label"] == 1].nlargest(5, "error")  # AI called real
errors = pd.concat([fp, fn]).reset_index(drop=True)
errors.index += 1

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for ax, (i, r) in zip(axes.ravel(), errors.iterrows()):
    ax.imshow(Image.open(r["path"]).convert("RGB"))
    true, guess = ("Real", "AI") if r["label"] == 0 else ("AI", "Real")
    ax.set_title(f"#{i} True: {true} | Pred: {guess}\n"
                 f"P(AI)={r['prob_ai']:.2f} | {r['width']}x{r['height']} {r['format']}",
                 fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("reports/figures/error_grid.png", dpi=150)

errors.to_csv("reports/error_analysis.csv", index_label="id")
print(errors[["path", "label", "prob_ai", "width", "height", "format", "mode"]].to_string())
