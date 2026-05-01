import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "sessions_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(api, "pipe", None)
    os.makedirs(api.sessions_dir, exist_ok=True)
    with TestClient(api.app) as test_client:
        yield test_client


def create_session(test_client):
    response = test_client.post(
        "/v1/avatar/sessions",
        files={"source_image": ("avatar.jpg", b"fake-image", "image/jpeg")},
        data={"animal": "false"},
    )
    assert response.status_code == 201
    return response.json()


def test_healthz_reports_service_ready_without_loading_models(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["engine_loaded"] is False


def test_readyz_returns_503_until_engine_loaded(client):
    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == "avatar engine is not loaded"


def test_index_page_serves_live_avatar_console(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Live Avatar Console" in response.text
    assert "/v1/avatar/sessions" in response.text
    assert "/static/app.js" in response.text


def test_static_app_js_served(client):
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "createSession" in response.text
    assert "WebSocket" in response.text


def test_create_avatar_session_requires_source_image(client):
    response = client.post("/v1/avatar/sessions")

    assert response.status_code == 422


def test_create_avatar_session_stores_source_image_and_json_metadata(client):
    payload = create_session(client)

    assert payload["id"]
    assert payload["source_filename"] == "avatar.jpg"
    assert payload["animal"] is False
    assert payload["status"] == "ready"

    metadata_path = os.path.join(api.sessions_dir, payload["id"], "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as fin:
        metadata = json.load(fin)
    assert metadata == payload


def test_rejects_path_traversal_session_id(client):
    response = client.delete("/v1/avatar/sessions/..%2Foutside")

    assert response.status_code == 404


def test_delete_avatar_session_removes_session(client):
    payload = create_session(client)

    response = client.delete(f"/v1/avatar/sessions/{payload['id']}")

    assert response.status_code == 204
    assert not os.path.exists(os.path.join(api.sessions_dir, payload["id"]))


def test_render_session_requires_driving_media(client):
    payload = create_session(client)

    response = client.post(f"/v1/avatar/sessions/{payload['id']}/render")

    assert response.status_code == 422
    assert "driving_video or driving_pickle is required" in response.text


def test_render_returns_503_when_engine_cannot_load(client, monkeypatch):
    payload = create_session(client)

    def fail_load():
        raise RuntimeError("missing checkpoints")

    monkeypatch.setattr(api, "load_avatar_engine", fail_load)
    response = client.post(
        f"/v1/avatar/sessions/{payload['id']}/render",
        files={"driving_video": ("drive.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 503
    assert "missing checkpoints" in response.text


def test_stream_avatar_session_returns_jpeg_frames(client, monkeypatch):
    payload = create_session(client)

    class FakePipe:
        src_imgs = [np.zeros((8, 8, 3), dtype=np.uint8)]
        src_infos = [{"fake": True}]

        def init_vars(self):
            return None

        def prepare_source(self, source_path, realtime=False):
            return os.path.exists(source_path) and realtime

        def run(self, frame, img_src, src_info, first_frame=False):
            output = np.full((8, 8, 3), 255, dtype=np.uint8)
            return frame, output, output, ({}, [], [])

    monkeypatch.setattr(api, "pipe", FakePipe())
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok

    with client.websocket_connect(f"/v1/avatar/sessions/{payload['id']}/stream") as websocket:
        websocket.send_bytes(encoded.tobytes())
        frame = websocket.receive_bytes()

    assert frame.startswith(b"\xff\xd8")
