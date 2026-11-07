import json
import os
import mlflow
import mlflow.tensorflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
NAME = "visionguard"

mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
model = mlflow.tensorflow.load_model(f"models:/{NAME}@champion")

os.makedirs("models/champion", exist_ok=True)
model.save("models/champion/model.keras")
meta = {"name": NAME, "version": mv.version, "run_name": mv.tags.get("run_name"),
        "threshold": float(mv.tags["threshold"]), "img_size": 224}
with open("models/champion/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved champion:", meta)