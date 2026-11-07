import tensorflow as tf
from tensorflow.keras import layers
from pipeline import IMG_SIZE

def build_cnn(dropout=0.3):
    return tf.keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1 / 255.0, name="rescale"),

        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(momentum=0.9, name="bn1"),
        layers.MaxPooling2D(name="pool1"),

        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(momentum=0.9, name="bn2"),
        layers.MaxPooling2D(name="pool2"),

        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(momentum=0.9, name="bn3"),
        layers.MaxPooling2D(name="pool3"),

        layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4"),
        layers.BatchNormalization(momentum=0.9, name="bn4"),
        layers.MaxPooling2D(name="pool4"),

        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dropout(dropout, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="baseline_cnn")

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model



def build_mobilenet(dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_tl"), base

def unfreeze_last_block(base, start="block_16_expand"):
    base.trainable = True
    names = [l.name for l in base.layers]
    for layer in base.layers[:names.index(start)]:
        layer.trainable = False
    return [l.name for l in base.layers if l.trainable]



def build_transfer(backbone="mobilenet", dropout=0.3, unfrozen=0):
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    if backbone == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
        x = layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")(inputs)  # to [-1, 1]
    else:
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
        x = inputs  # EfficientNet rescales 0-255 inputs itself

    base.trainable = unfrozen > 0
    if unfrozen > 0:
        for layer in base.layers[:-unfrozen]:
            layer.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=f"{backbone}_tl")