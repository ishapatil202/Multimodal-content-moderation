import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from moderation.api import create_api
from tests.fakes import make_moderator


@pytest.fixture
def client():
    return TestClient(create_api(make_moderator()))


def png(color=(0, 0, 255), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format=fmt)
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_text(client):
    r = client.post("/api/moderate/text", json={"text": "you idiot"})
    body = r.json()
    assert r.status_code == 200
    assert body["flagged"] and body["top_category"] == "Toxic"
    assert body["latency_ms"] >= 0


def test_text_threshold(client):
    r = client.post("/api/moderate/text", json={"text": "you idiot", "threshold": 0.99})
    assert r.json()["flagged"] is False


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "hi", "threshold": 0}, {}])
def test_text_errors(client, payload):
    assert client.post("/api/moderate/text", json=payload).status_code in (400, 422)


def test_batch(client):
    r = client.post("/api/moderate/batch", json={"texts": ["hi", "you idiot"]})
    assert r.status_code == 200
    assert r.json()["count"] == 2
    assert [x["flagged"] for x in r.json()["results"]] == [False, True]


def test_batch_empty_item(client):
    assert client.post("/api/moderate/batch", json={"texts": ["ok", ""]}).status_code == 400


def test_image(client):
    r = client.post("/api/moderate/image", files={"image": ("a.png", png((255, 0, 0)), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["flagged"] and body["modalities"] == ["image"]
    assert body["ai_generated"] is not None


def test_image_jpeg_with_threshold(client):
    r = client.post(
        "/api/moderate/image",
        files={"image": ("a.jpg", png((255, 0, 0), "JPEG"), "image/jpeg")},
        data={"threshold": "0.99"},
    )
    assert r.status_code == 200 and r.json()["flagged"] is False


def test_bad_image(client):
    r = client.post("/api/moderate/image", files={"image": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_image_too_large(client, monkeypatch):
    from moderation import config

    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 10)
    r = client.post("/api/moderate/image", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 413


def test_multimodal(client):
    r = client.post(
        "/api/moderate",
        data={"text": "nice pic"},
        files={"image": ("a.png", png((255, 0, 0)), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["modalities"] == ["text", "image"]
    assert r.json()["top_category"] == "NSFW image"


def test_multimodal_text_only(client):
    r = client.post("/api/moderate", data={"text": "you idiot"})
    assert r.status_code == 200 and r.json()["modalities"] == ["text"]


def test_multimodal_nothing(client):
    assert client.post("/api/moderate", data={}).status_code == 400
