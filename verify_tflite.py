import sys
import pandas as pd
import tensorflow as tf
from preprocessing import decode_and_resize

sys.path.insert(0, "vercel_api/api")
import index as api   # noqa: E402

keras_model = tf.keras.models.load_model("models/champion/model.keras")
diffs = []
for path in pd.read_csv("data/splits/test.csv")["path"].head(20):
    data = open(path, "rb").read()
    x = tf.cast(decode_and_resize(data), tf.float32)[tf.newaxis]
    p_keras = float(keras_model(x, training=False)[0][0])
    p_tflite = api.predict_prob(api.preprocess(data))
    diffs.append(abs(p_keras - p_tflite))
    print(f"{path[-40:]:>40}  keras {p_keras:.4f}  tflite {p_tflite:.4f}")
print(f"\nMax difference: {max(diffs):.5f}  Mean: {sum(diffs) / len(diffs):.5f}")
