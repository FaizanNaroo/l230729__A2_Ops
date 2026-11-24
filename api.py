import hmac
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from inference import Classifier

load_dotenv()
API_KEY = os.environ["API_KEY"]                       # required, never hard-coded
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
classifier = Classifier(os.getenv("MODEL_DIR", "models/champion"),
                        threshold=os.getenv("THRESHOLD"))   # loaded once


@app.get("/health")
def health():
    return jsonify(status="ok", model=classifier.model_name, threshold=classifier.threshold)


@app.post("/predict")
def predict():
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return jsonify(error="invalid or missing API key"), 401
    file = request.files.get("image")
    if file is None:
        return jsonify(error="send the image as form field 'image'"), 400
    try:
        return jsonify(classifier.predict(file.read()))
    except Exception:
        return jsonify(error="could not decode image"), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify(error=f"file larger than {MAX_MB} MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)