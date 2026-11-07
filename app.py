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
from inference import Classifier

load_dotenv()
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
    MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,   # Flask rejects bigger uploads with 413
)
db.init_app(app)
with app.app_context():
    db.create_all()

#classifier = Classifier(os.getenv("MODEL_DIR", "models/champion"))
  # loaded ONCE
classifier = Classifier(os.getenv("MODEL_DIR", "models/champion"),
                        threshold=os.getenv("THRESHOLD"))  # loaded ONCE

class UploadError(Exception):
    pass

def validate_upload(file):
    """Return (safe_name, bytes) or raise UploadError. Nothing is written to disk."""
    if file is None or file.filename == "":
        raise UploadError("Please choose an image to upload.")
    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Only JPG and PNG images are accepted (you sent '.{ext or '?'}').")
    data = file.read()
    if not data:
        raise UploadError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:   # check the real content, not the name
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise UploadError("This file has an image name but could not be read as an image.")
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"The file content is {fmt}, not JPG or PNG.")
    return name, data

@app.context_processor
def inject_model():
    return {"model_name": classifier.model_name}

@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)

@app.post("/predict")
def predict():
    try:
        name, data = validate_upload(request.files.get("image"))
        t0 = time.perf_counter()
        try:
            result = classifier.predict(data)
        except Exception:
            raise UploadError("The image could not be decoded for analysis.")
        latency = int((time.perf_counter() - t0) * 1000)
    except UploadError as e:
        return render_template("error.html", title="Upload rejected", message=str(e)), 400

    row = Prediction(filename=name, file_size=len(data),
                     image_sha256=hashlib.sha256(data).hexdigest(),
                     label=result["label"], prob_ai=result["prob_ai"],
                     confidence=result["confidence"], uncertain=result["uncertain"],
                     threshold=result["threshold"], model_name=result["model"],
                     latency_ms=latency)
    db.session.add(row)
    db.session.commit()
    return redirect(url_for("result", pred_id=row.id))   # Post/Redirect/Get

@app.get("/result/<int:pred_id>")
def result(pred_id):
    return render_template("result.html", p=db.get_or_404(Prediction, pred_id))

@app.get("/history")
def history():
    rows = db.session.execute(
        db.select(Prediction).order_by(Prediction.created_at.desc()).limit(100)
    ).scalars().all()
    return render_template("history.html", rows=rows)

@app.get("/health")
def health():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    body = {"status": "ok" if db_ok else "degraded", "model": classifier.model_name,
            "threshold": classifier.threshold, "database": "ok" if db_ok else "unreachable"}
    return jsonify(body), 200 if db_ok else 503

@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", title="File too large",
                           message=f"The maximum upload size is {MAX_MB} MB."), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", title="Page not found",
                           message="That page or result does not exist."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", title="Something went wrong",
                           message="An unexpected error occurred. Please try again."), 500

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")