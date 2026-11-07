import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x, y = x[:20000] / 255.0, y[:20000]  # a subset keeps it fast on CPU

def build(hp):
    model = tf.keras.Sequential([
        layers.Input((28, 28)),
        layers.Flatten(),
        layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=8, seed=42,
                        directory="tuner", project_name="fashion_toy", overwrite=True)
tuner.search(x, y, epochs=3, validation_split=0.2, verbose=2)

rows = [{**t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.trials.values()]
table = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
table.to_csv("reports/tuner_toy_results.csv", index=False)
print(table.to_string(index=False))
print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)