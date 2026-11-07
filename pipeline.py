import pandas as pd
import tensorflow as tf
from preprocessing import decode_and_resize

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="augment")

# def load_image(path, label):
#     img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
#     img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
#     return img, label

def load_image(path, label):
    return decode_and_resize(tf.io.read_file(path), IMG_SIZE), label

# def make_dataset(csv_path, training=False):
#     df = pd.read_csv(csv_path)
#     ds = tf.data.Dataset.from_tensor_slices(
#         (df["path"].values, df["label"].values.astype("float32")))
#     ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
#     if training:
#         ds = ds.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
#     ds = ds.batch(BATCH)
#     if training:
#         ds = ds.map(lambda x, y: (augment(tf.cast(x, tf.float32), training=True), y),
#                     num_parallel_calls=AUTOTUNE)
#     else:
#         ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
#                     num_parallel_calls=AUTOTUNE)
#     return ds.prefetch(AUTOTUNE)

def make_dataset(csv_path, training=False, use_augment=True):
    df = pd.read_csv(csv_path)
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32")))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH)
    if training and use_augment:
        ds = ds.map(lambda x, y: (augment(tf.cast(x, tf.float32), training=True), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)