import shutil
from pathlib import Path
import tensorflow as tf

src = Path("models/champion")
out = Path("vercel_api/model")
out.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(src / "model.keras")
model.export(str(src / "saved_model"))          # Keras 3 -> TensorFlow SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(str(src / "saved_model"))
(out / "model.tflite").write_bytes(converter.convert())
shutil.copy(src / "meta.json", out / "meta.json")
print("TFLite size: %.1f MB" % ((out / "model.tflite").stat().st_size / 1e6))