import hmac
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from urllib.parse import parse_qsl, urlencode

try:
    from ai_edge_litert.interpreter import Interpreter      # small runtime (Vercel)
except ImportError:
    import tensorflow as tf                                 # fallback for local testing
    Interpreter = tf.lite.Interpreter

BASE = Path(__file__).resolve().parent.parent
META = json.loads((BASE / "model" / "meta.json").read_text())
IMG_SIZE = META["img_size"]
THRESHOLD = round(float(os.getenv("THRESHOLD") or META["threshold"]), 2)
MODEL_NAME = f"{META['name']} v{META['version']} ({META['run_name']})"
API_KEY = os.getenv("API_KEY", "")
MARGIN = 0.10

# Load ONCE per instance (cold start), then reuse for every request
interpreter = Interpreter(model_path=str(BASE / "model" / "model.tflite"))
interpreter.allocate_tensors()
IN_IDX = interpreter.get_input_details()[0]["index"]
OUT_IDX = interpreter.get_output_details()[0]["index"]
LOCK = threading.Lock()   # the interpreter is not thread-safe


def _axis(in_size, out_size):
    x = (np.arange(out_size) + 0.5) * (in_size / out_size) - 0.5
    x_floor = np.floor(x)
    lo = np.maximum(x_floor, 0).astype(np.int64)
    hi = np.minimum(np.ceil(x), in_size - 1).astype(np.int64)
    return lo, hi, (x - x_floor).astype(np.float32)


def tf_bilinear_resize(img, size):
    """Same maths as tf.image.resize(bilinear): half-pixel centres, no antialiasing."""
    y0, y1, wy = _axis(img.shape[0], size)
    x0, x1, wx = _axis(img.shape[1], size)
    img = img.astype(np.float32)
    wx = wx[None, :, None]
    top = img[y0][:, x0] + (img[y0][:, x1] - img[y0][:, x0]) * wx
    bot = img[y1][:, x0] + (img[y1][:, x1] - img[y1][:, x0]) * wx
    return top + (bot - top) * wy[:, None, None]


def preprocess(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in ("JPEG", "PNG", "MPO"):
            raise ValueError(f"Unsupported format {im.format}")
        img = np.asarray(im.convert("RGB"))             # drop alpha, like channels=3
    x = np.clip(tf_bilinear_resize(img, IMG_SIZE), 0, 255).astype(np.uint8)  # like tf.cast
    return x.astype(np.float32)[None, ...]


def predict_prob(x):
    with LOCK:
        interpreter.set_tensor(IN_IDX, x)
        interpreter.invoke()
        return float(interpreter.get_tensor(OUT_IDX)[0][0])


predict_prob(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.float32))   # warm-up

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024               # below Vercel's 4.5 MB

class VercelPathFix:
    """Vercel's rewrite sends every request to /api/index; restore the real path."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        params = parse_qsl(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        original = [v for k, v in params if k == "__path"]
        if original:
            environ["PATH_INFO"] = original[0] or "/"
            environ["QUERY_STRING"] = urlencode([(k, v) for k, v in params if k != "__path"])
        return self.wsgi_app(environ, start_response)


app.wsgi_app = VercelPathFix(app.wsgi_app)



@app.get("/")
@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, threshold=THRESHOLD, runtime="tflite")


@app.post("/predict")
def predict():
    if not API_KEY:
        return jsonify(error="Server API key not configured"), 500
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return jsonify(error="Missing or invalid API key"), 401
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Send the image as form-data field 'image'"), 400
    try:
        x = preprocess(file.read())
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return jsonify(error=f"Not a readable JPG/PNG image: {e}"), 400
    t0 = time.perf_counter()
    p = predict_prob(x)
    is_ai = p >= THRESHOLD
    return jsonify(label="AI-generated" if is_ai else "Real", prob_ai=round(p, 4),
                   confidence=round(p if is_ai else 1 - p, 4),
                   uncertain=abs(p - THRESHOLD) < MARGIN, threshold=THRESHOLD,
                   model=MODEL_NAME, inference_ms=int((time.perf_counter() - t0) * 1000))


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="Image larger than 4 MB"), 413


@app.route("/<path:anything>", methods=["GET", "POST"])
def debug_path(anything):
    # TEMPORARY: shows which path Flask actually receives on Vercel
    return jsonify(error="no matching route", path=request.path,
                   script_root=request.script_root, url=request.url), 404