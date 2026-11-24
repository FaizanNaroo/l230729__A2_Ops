"""Inference back-ends for VisionGuard.

Classifier        - loads the champion Keras model once and predicts locally (Task 4).
RemoteClassifier  - same interface, but sends the image to the model API deployed
                    on Vercel and returns its JSON verdict (Task 7 / Task 8).

The Flask app picks one of them from the INFERENCE_MODE environment variable.
"""
import json
import sys
import time
from pathlib import Path

import requests


class Classifier:
    def __init__(
            self,
            model_dir="models/champion",
            threshold=None,
            uncertain_margin=0.10):
        # TensorFlow is imported here so that remote mode does not need to load it
        import tensorflow as tf
        from preprocessing import decode_and_resize
        self._tf, self._decode = tf, decode_and_resize

        model_dir = Path(model_dir)
        self.model = tf.keras.models.load_model(model_dir / "model.keras")
        meta = json.loads((model_dir / "meta.json").read_text())
        # Environment variable wins; otherwise use the threshold saved with the model
        self.threshold = round(
            float(threshold) if threshold else meta["threshold"], 2)
        self.img_size = meta["img_size"]
        self.model_name = f"{meta['name']} v{meta['version']} ({meta['run_name']})"
        self.margin = uncertain_margin
        self.mode = "local"

    def predict(self, image_bytes):
        tf = self._tf
        img = self._decode(image_bytes, self.img_size)       # same as training
        x = tf.cast(img, tf.float32)[tf.newaxis, ...]
        prob_ai = float(self.model(x, training=False)[0][0])
        is_ai = prob_ai >= self.threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "prob_ai": round(prob_ai, 4),
            "confidence": round(prob_ai if is_ai else 1 - prob_ai, 4),
            "uncertain": abs(prob_ai - self.threshold) < self.margin
            or (prob_ai < self.threshold and prob_ai >= 0.5),
            "threshold": self.threshold,
            "model": self.model_name,
        }

    def status(self):
        return {"mode": "local", "reachable": True}


class RemoteError(Exception):
    """The model API could not be used (network, auth, server error)."""


class BadImageError(ValueError):
    """The model API rejected the image itself (HTTP 400 / 413)."""


class RemoteClassifier:
    """Same interface as Classifier, but the model runs on Vercel.

    The image bytes are forwarded unchanged; the API does the decoding and resizing
    with the same maths as training. The API key travels in the X-API-Key header.
    """

    def __init__(self, url, api_key, timeout=30, session=None):
        if not url or not api_key:
            raise RuntimeError("Remote mode needs API_URL and API_KEY in the environment.")
        self.url = url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.timeout = timeout
        self.http = session or requests.Session()
        self.mode = "remote"
        self.model_name, self.threshold = "remote model (not reached yet)", None
        self._refresh_info()

    def _refresh_info(self):
        """Read model name and threshold from the API's /health. Never raises."""
        try:
            r = self.http.get(f"{self.url}/health", timeout=self.timeout)
            r.raise_for_status()
            info = r.json()
            self.model_name = f"{info['model']} @ Vercel"
            self.threshold = info["threshold"]
            return True
        except Exception:
            return False

    def status(self):
        reachable = self._refresh_info()
        return {"mode": "remote", "api_url": self.url, "reachable": reachable}

    @staticmethod
    def _error_text(r):
        try:
            return r.json().get("error", r.text[:200])
        except ValueError:
            return r.text[:200]

    def predict(self, image_bytes):
        try:
            r = self.http.post(f"{self.url}/predict", headers=self.headers,
                               files={"image": ("upload", image_bytes)},
                               timeout=self.timeout)
        except requests.Timeout:
            raise RemoteError("The model service took too long to answer. Please try again.")
        except requests.RequestException:
            raise RemoteError("The model service could not be reached. Please try again.")

        if r.status_code in (400, 413):
            raise BadImageError(self._error_text(r))
        if r.status_code == 401:
            raise RemoteError("The model service refused the API key (check API_KEY).")
        if r.status_code != 200:
            raise RemoteError(f"The model service returned an error ({r.status_code}).")
        try:
            result = r.json()
        except ValueError:
            raise RemoteError("The model service sent an invalid response.")

        result["model"] = f"{result['model']} @ Vercel"
        # keep the footer/health info in sync with what the API actually used
        self.model_name, self.threshold = result["model"], result["threshold"]
        return result


if __name__ == "__main__":
    image_paths = [Path(path) for path in sys.argv[1:]]
    if not image_paths:
        raise SystemExit(
            "Usage: python inference.py <image-path> [<image-path> ...]")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "Image file not found. Replace the placeholder with an actual image path:\n" +
            "\n".join(f"  {path}" for path in missing))

    t0 = time.perf_counter()
    clf = Classifier()
    print(
        f"Model loaded once in {time.perf_counter() - t0:.2f} s: {clf.model_name}")
    for path in image_paths:
        t0 = time.perf_counter()
        result = clf.predict(path.read_bytes())
        print(f"{path}\n  -> {result}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
