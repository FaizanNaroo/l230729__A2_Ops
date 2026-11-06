import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pipeline import make_dataset

tf.keras.utils.set_random_seed(42)

# 1. Check that TensorFlow can decode every image
bad = []
for split in ["train", "val", "test"]:
    for p in pd.read_csv(f"data/splits/{split}.csv")["path"]:
        try:
            tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        except Exception:
            bad.append(p)
print("Files TF cannot decode:", bad)

# 2. Show one augmented training batch
images, labels = next(iter(make_dataset("data/splits/train.csv", training=True)))
print("Batch shape:", images.shape,
      "| pixel range:", float(tf.reduce_min(images)), "-", float(tf.reduce_max(images)))

plt.figure(figsize=(12, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title("AI" if labels[i] == 1 else "Real")
    plt.axis("off")
plt.tight_layout()
plt.savefig("reports/figures/augmented_batch.png", dpi=150)