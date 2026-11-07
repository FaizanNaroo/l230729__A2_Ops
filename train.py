
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import tensorflow as tf
from pipeline import make_dataset
from models import build_cnn, build_mobilenet, unfreeze_last_block, compile_model

SEED = 42
tf.keras.utils.set_random_seed(SEED)            # Python, NumPy and TF seeds
tf.config.experimental.enable_op_determinism()  # same results on every run

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--ft_lr", type=float, default=1e-5)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--ft_epochs", type=int, default=10)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--patience", type=int, default=5)
args = p.parse_args()

train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")
Path("models").mkdir(exist_ok=True)
fig_dir = Path("reports/figures")

def callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"models/{name}.keras", monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                             patience=2, min_lr=1e-7),
    ]

def plot(history, name):
    h = pd.DataFrame(history.history)
    h.index += 1  # count epochs from 1
    h.to_csv(f"reports/{name}_history.csv", index_label="epoch")
    best = h["val_loss"].idxmin()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(h.index, h[m], label="train")
        ax.plot(h.index, h[f"val_{m}"], label="validation")
        ax.axvline(best, color="red", ls="--", label=f"best epoch {best}")
        ax.set(title=f"{name}: {m}", xlabel="epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_curves.png", dpi=150)
    plt.close()
    print(f"{name}: best epoch {best}, val_loss {h['val_loss'][best]:.4f}, "
          f"val_accuracy {h['val_accuracy'][best]:.4f}")

def trainable_report(model, base, path):
    names = [l.name for l in base.layers if l.trainable]
    head = [l.name for l in model.layers if l.trainable and l is not base]
    text = (f"Trainable base layers ({len(names)}): {names}\n"
            f"Trainable head layers: {head}\n"
            f"Trainable params: {sum(w.numpy().size for w in model.trainable_weights):,}\n")
    Path(path).write_text(text)
    print(text)

if args.model == "cnn":
    model = compile_model(build_cnn(args.dropout), args.lr)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=callbacks("cnn"), verbose=2)
    plot(hist, "cnn")
else:
    # Stage 1: frozen base, train only the new head
    model, base = build_mobilenet(args.dropout)
    compile_model(model, args.lr)
    trainable_report(model, base, "reports/stage1_trainable.txt")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=callbacks("mobilenet_stage1"), verbose=2)
    plot(hist, "mobilenet_stage1")

    # Stage 2: unfreeze the last block, train with a much smaller learning rate
    unfreeze_last_block(base)
    compile_model(model, args.ft_lr)  # recompile so the change takes effect
    trainable_report(model, base, "reports/stage2_trainable.txt")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs,
                     callbacks=callbacks("mobilenet_stage2"), verbose=2)
    plot(hist, "mobilenet_stage2")
