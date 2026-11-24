"""Task 8: the Flask-side client for the Vercel model API, tested with a fake HTTP
session so CI never calls the real API or needs the secret key."""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import BadImageError, RemoteClassifier, RemoteError  # noqa: E402


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


class FakeSession:
    def __init__(self, post_response=None, post_exc=None, health_ok=True):
        self.post_response, self.post_exc, self.health_ok = post_response, post_exc, health_ok
        self.sent_headers = None

    def get(self, url, timeout):
        if not self.health_ok:
            raise requests.ConnectionError("down")
        return FakeResponse(200, {"status": "ok", "model": "visionguard v4", "threshold": 0.74})

    def post(self, url, headers, files, timeout):
        self.sent_headers = headers
        if self.post_exc:
            raise self.post_exc
        return self.post_response


OK_BODY = {"label": "Real", "prob_ai": 0.2, "confidence": 0.8, "uncertain": False,
           "threshold": 0.74, "model": "visionguard v4"}


def make(**kw):
    return RemoteClassifier("https://example.test/", "secret", session=FakeSession(**kw))


def test_reads_model_info_at_startup():
    clf = make()
    assert clf.model_name == "visionguard v4 @ Vercel" and clf.threshold == 0.74
    assert clf.url == "https://example.test"


def test_startup_survives_unreachable_api():
    clf = make(health_ok=False)
    assert clf.threshold is None and clf.status()["reachable"] is False


def test_successful_prediction_sends_key():
    clf = make(post_response=FakeResponse(200, dict(OK_BODY)))
    out = clf.predict(b"img")
    assert out["label"] == "Real" and out["model"].endswith("@ Vercel")
    assert clf.http.sent_headers == {"X-API-Key": "secret"}


@pytest.mark.parametrize("status", [400, 413])
def test_bad_image_is_client_error(status):
    clf = make(post_response=FakeResponse(status, {"error": "nope"}))
    with pytest.raises(BadImageError):
        clf.predict(b"img")


@pytest.mark.parametrize("response,exc", [
    (FakeResponse(401, {"error": "bad key"}), None),
    (FakeResponse(500, ValueError("not json")), None),
    (FakeResponse(200, ValueError("not json")), None),
    (None, requests.Timeout()),
    (None, requests.ConnectionError()),
])
def test_service_failures_are_remote_errors(response, exc):
    clf = make(post_response=response, post_exc=exc)
    with pytest.raises(RemoteError):
        clf.predict(b"img")


def test_missing_config_fails_fast():
    with pytest.raises(RuntimeError):
        RemoteClassifier("", "")
