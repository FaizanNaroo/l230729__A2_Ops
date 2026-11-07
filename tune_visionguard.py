import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from pipeline import make_dataset, IMG_SIZE
from models import compile_model

tf.keras.utils.set_random_seed(42)
train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")

def build(hp):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    n = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
    base.trainable = n > 0
    if n > 0:
        for layer in base.layers[:-n]:  # freeze everything except the last n layers
            layer.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hp.Choice("dropout", [0.2, 0.3, 0.5]))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return compile_model(model, hp.Choice("lr", [1e-3, 3e-4, 1e-4]))

tuner = kt.RandomSearch(build, objective=kt.Objective("val_loss", "min"),
                        max_trials=10, seed=42, directory="tuner",
                        project_name="visionguard", overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=8, verbose=2,
             callbacks=[tf.keras.callbacks.EarlyStopping(
                 monitor="val_loss", patience=3, restore_best_weights=True)])

rows = [{**t.hyperparameters.values, "val_loss": t.score}
        for t in tuner.oracle.trials.values()]
table = pd.DataFrame(rows).sort_values("val_loss")
table.to_csv("reports/tuner_visionguard_results.csv", index=False)
print(table.to_string(index=False))
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)

# Which setting mattered most? Compare the average val_loss for each value.
for col in ["lr", "dropout", "unfrozen_layers"]:
    print(f"\nMean val_loss by {col}:\n", table.groupby(col)["val_loss"].mean().round(4))