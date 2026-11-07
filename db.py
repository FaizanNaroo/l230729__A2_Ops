from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))      # when
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    image_sha256 = db.Column(db.String(64), nullable=False)                # which image
    label = db.Column(db.String(20), nullable=False)                       # what was decided
    prob_ai = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)                       # how confidently
    uncertain = db.Column(db.Boolean, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)                 # by which model
    latency_ms = db.Column(db.Integer, nullable=False)