import os
import sys
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from mlflow import MlflowClient
from dotenv import load_dotenv
from pipeline import load_image

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
NAME, ALIAS = "visionguard", "champion"

mv = MlflowClient().get_model_version_by_alias(NAME, ALIAS)
threshold = float(mv.tags.get("threshold", 0.5))
model = mlflow.tensorflow.load_model(f"models:/{NAME}@{ALIAS}")
print(f"Loaded {NAME}@{ALIAS} -> version {mv.version} ({mv.tags.get('run_name')}), "
      f"threshold {threshold:.2f}")

path = sys.argv[1]
img, _ = load_image(path, 0.0)                    # same decode + resize as training
x = tf.cast(img, tf.float32)[tf.newaxis, ...]     # add a batch dimension
prob_ai = float(model.predict(x, verbose=0)[0][0])
label = "AI-generated" if prob_ai >= threshold else "Real"
confidence = prob_ai if label == "AI-generated" else 1 - prob_ai
print(f"Image: {path}\nP(AI) = {prob_ai:.3f} -> {label} (confidence {confidence:.1%})")