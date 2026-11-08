import io
import json
import os
import sys

import pytest
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.save(model_dir / "model.keras")
    (
        model_dir /
        "meta.json").write_text(
        json.dumps(
            {
                "name": "test",
                "version": "0",
                "run_name": "dummy",
                "threshold": 0.5,
                "img_size": 224}))
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["DATABASE_URL"] = "sqlite:///" + \
        (tmp_path_factory.mktemp("db") / "t.db").as_posix()
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
    return buf.getvalue()


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_rejects_wrong_extension(client):
    assert upload(client, b"hello", "notes.txt").status_code == 400


def test_rejects_fake_image(client):
    assert upload(client, b"not an image", "fake.jpg").status_code == 400


def test_valid_png_is_saved(client):
    assert upload(client, png_bytes(), "red.png").status_code == 302
    assert b"red.png" in client.get("/history").data


def test_preprocessing_shape():
    from preprocessing import decode_and_resize
    out = decode_and_resize(png_bytes(), 224)
    assert out.shape == (224, 224, 3) and out.dtype == tf.uint8
