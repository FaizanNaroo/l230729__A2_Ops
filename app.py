import hashlib
import io
import os
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from sqlalchemy import text
from werkzeug.utils import secure_filename

from db import Prediction, db
from inference import BadImageError, Classifier, RemoteClassifier, RemoteError

load_dotenv()
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO"}   # MPO = multi-picture JPEG from phones
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").strip().lower()
VERCEL_BODY_LIMIT_MB = 4                   # Vercel functions reject bodies over 4.5 MB
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
if INFERENCE_MODE == "remote":
    MAX_MB = min(MAX_MB, VERCEL_BODY_LIMIT_MB)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
    MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,   # Flask rejects bigger uploads with 413
)
db.init_app(app)
with app.app_context():
    db.create_all()


def build_classifier():
    """Created ONCE at start-up, never per request."""
    if INFERENCE_MODE == "remote":
        return RemoteClassifier(os.getenv("API_URL", ""), os.getenv("API_KEY", ""),
                                timeout=int(os.getenv("API_TIMEOUT", "30")))
    return Classifier(os.getenv("MODEL_DIR", "models/champion"),
                      threshold=os.getenv("THRESHOLD"))


classifier = build_classifier()


class UploadError(Exception):
    pass


def validate_upload(file):
    """Return (safe_name, bytes) or raise UploadError. Nothing is written to disk."""
    if file is None or file.filename == "":
        raise UploadError("Please choose an image to upload.")
    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(
            f"Only JPG and PNG images are accepted (you sent '.{ext or '?'}').")
    data = file.read()
    if not data:
        raise UploadError("The file is empty.")
    try:
        # check the real content, not the name
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise UploadError(
            "This file has an image name but could not be read as an image.")
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"The file content is {fmt}, not JPG or PNG.")
    return name, data


def error_page(title, message, code):
    return render_template("error.html", title=title, message=message), code


@app.context_processor
def inject_model():
    return {"model_name": classifier.model_name, "inference_mode": INFERENCE_MODE}


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/predict")
def predict():
    try:
        name, data = validate_upload(request.files.get("image"))
    except UploadError as e:
        return error_page("Upload rejected", str(e), 400)

    t0 = time.perf_counter()
    try:
        result = classifier.predict(data)
    except RemoteError as e:
        app.logger.warning("model API failure: %s", e)
        return error_page("Model service unavailable", str(e), 503)
    except BadImageError as e:
        return error_page("Upload rejected", f"The model service rejected the image: {e}", 400)
    except Exception:
        app.logger.exception("prediction failed")
        return error_page("Upload rejected", "The image could not be decoded for analysis.", 400)
    latency = int((time.perf_counter() - t0) * 1000)

    row = Prediction(filename=name, file_size=len(data),
                     image_sha256=hashlib.sha256(data).hexdigest(),
                     label=result["label"], prob_ai=result["prob_ai"],
                     confidence=result["confidence"], uncertain=result["uncertain"],
                     threshold=result["threshold"], model_name=result["model"],
                     latency_ms=latency)
    db.session.add(row)
    db.session.commit()
    app.logger.info("prediction #%s %s -> %s (%.3f) via %s in %d ms",
                    row.id, name, row.label, row.prob_ai, INFERENCE_MODE, latency)
    return redirect(url_for("result", pred_id=row.id))   # Post/Redirect/Get


@app.get("/result/<int:pred_id>")
def result(pred_id):
    return render_template("result.html", p=db.get_or_404(Prediction, pred_id))


def recent_predictions(limit=100):
    return db.session.execute(
        db.select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    ).scalars().all()


@app.get("/history")
def history():
    return render_template("history.html", rows=recent_predictions())


@app.get("/api/history")
def history_json():
    """Same database trail as /history, as JSON (handy for Postman and the demo script)."""
    limit = min(request.args.get("limit", 20, type=int), 100)
    return jsonify([{
        "id": r.id,
        "created_at": r.created_at.isoformat(),
        "filename": r.filename,
        "label": r.label,
        "prob_ai": r.prob_ai,
        "confidence": r.confidence,
        "uncertain": r.uncertain,
        "threshold": r.threshold,
        "model": r.model_name,
        "latency_ms": r.latency_ms,
    } for r in recent_predictions(limit)])


@app.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    inference = classifier.status()
    ok = db_ok and inference["reachable"]
    body = {
        "status": "ok" if ok else "degraded",
        "model": classifier.model_name,
        "threshold": classifier.threshold,
        "database": "ok" if db_ok else "unreachable",
        "inference": inference}
    # Only the database decides the HTTP code: an outage of the external model API
    # should show as "degraded", not make Docker restart a healthy web container.
    return jsonify(body), 200 if db_ok else 503


@app.errorhandler(413)
def too_large(e):
    return error_page("File too large", f"The maximum upload size is {MAX_MB} MB.", 413)


@app.errorhandler(404)
def not_found(e):
    return error_page("Page not found", "That page or result does not exist.", 404)


@app.errorhandler(500)
def server_error(e):
    return error_page("Something went wrong",
                      "An unexpected error occurred. Please try again.", 500)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")
