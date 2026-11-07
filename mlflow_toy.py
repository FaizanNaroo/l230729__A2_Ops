import argparse
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-3)
args = p.parse_args()

from dotenv import load_dotenv
load_dotenv()  # copies the values from .env into environment variables
# Local database by default; Task 3.2 switches to DagsHub via an environment variable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("fashion_toy")

class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

tf.keras.utils.set_random_seed(42)
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x, y = x[:20000] / 255.0, y[:20000]

with mlflow.start_run(run_name=f"lr_{args.lr}"):
    mlflow.log_params({"lr": args.lr, "units": 96, "epochs": 5, "train_size": len(x)})
    mlflow.set_tag("purpose", "toy")

    model = tf.keras.Sequential([
        layers.Input((28, 28)),
        layers.Flatten(),
        layers.Dense(96, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(x, y, epochs=5, validation_split=0.2, verbose=2,
                     callbacks=[MLflowLogger()])

    fig, ax = plt.subplots()
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="validation")
    ax.set(title=f"Loss (lr={args.lr})", xlabel="epoch")
    ax.legend()
    mlflow.log_figure(fig, "loss_curve.png")

    mlflow.tensorflow.log_model(model, name="model")