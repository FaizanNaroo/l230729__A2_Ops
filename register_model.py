import os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
NAME = "visionguard"
client = MlflowClient()

runs = mlflow.search_runs(experiment_names=["visionguard"],
                          filter_string="tags.tracking = 'manual'")
runs = runs.set_index("tags.mlflow.runName")

def register(run_name, alias, note):
    run = runs.loc[run_name]
    run_id = run["run_id"]
    run_info = client.get_run(run_id).info
    logged_models = client.search_logged_models(
        experiment_ids=[run_info.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )
    logged_model = next((model for model in logged_models if model.name == "model"), None)
    if logged_model is None:
        raise RuntimeError(f"No logged model named 'model' found for run {run_id}")

    mv = mlflow.register_model(f"models:/{logged_model.model_id}", NAME)
    client.set_model_version_tag(NAME, mv.version, "threshold", str(run["metrics.threshold"]))
    client.set_model_version_tag(NAME, mv.version, "run_name", run_name)
    client.update_model_version(NAME, mv.version, description=note)
    client.set_registered_model_alias(NAME, alias, mv.version)
    print(f"{run_name} -> version {mv.version}, alias '{alias}', "
          f"threshold {run['metrics.threshold']:.2f}")

register("dropout_0.5", "challenger", "MobileNetV2, dropout 0.5 (runner-up)")
register("efficientnet", "champion",
         "EfficientNetB0, frozen base (best validation loss 0.518)")