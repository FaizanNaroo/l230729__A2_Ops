import json
import sys
import time
from pathlib import Path
import tensorflow as tf
from preprocessing import decode_and_resize


class Classifier:
    def __init__(
            self,
            model_dir="models/champion",
            threshold=None,
            uncertain_margin=0.10):
        model_dir = Path(model_dir)
        self.model = tf.keras.models.load_model(model_dir / "model.keras")
        meta = json.loads((model_dir / "meta.json").read_text())
        # self.threshold = meta["threshold"]
        # Environment variable wins; otherwise use the threshold saved with the
        # model
        self.threshold = round(
            float(threshold) if threshold else meta["threshold"], 2)
        self.img_size = meta["img_size"]
        self.model_name = f"{meta['name']} v{meta['version']} ({meta['run_name']})"
        self.margin = uncertain_margin

    def predict(self, image_bytes):
        img = decode_and_resize(image_bytes,
                                self.img_size)       # same as training
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


if __name__ == "__main__":
    image_paths = [Path(path) for path in sys.argv[1:]]
    if not image_paths:
        raise SystemExit(
            "Usage: python inference.py <image-path> [<image-path> ...]")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "Image file not found. Replace the placeholder with an actual image path:\n" +
            "\n".join(
                f"  {path}" for path in missing))

    t0 = time.perf_counter()
    clf = Classifier()
    print(
        f"Model loaded once in {time.perf_counter() - t0:.2f} s: {clf.model_name}")
    for path in image_paths:
        t0 = time.perf_counter()
        result = clf.predict(path.read_bytes())
        print(f"{path}\n  -> {result}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
