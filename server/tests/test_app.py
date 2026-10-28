import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

flask = pytest.importorskip("flask")

from ocr.modelD import ModelD
from ocr.training_data import WIDTH
from ocr.train import CONTEXT_LENGTH, END, NUM_CLASSES, TOKEN_DIM, alphabet
from server.app import create_app


@pytest.fixture
def client(tmp_path):
    # A small untrained model and one photo: a dark bar on white
    np.random.seed(0)
    params = {"depth": 1, "emb_dim": 8}
    ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
           num_classes=NUM_CLASSES, blank=END, **params).save_weights(
               str(tmp_path / "model.npz"))
    (tmp_path / "photos" / "abka").mkdir(parents=True)
    image = Image.new("L", (40, 120), 255)
    image.paste(0, (15, 10, 25, 110))
    image.save(tmp_path / "photos" / "abka" / "1.png")
    (tmp_path / "photos" / "manifest.json").write_text(json.dumps(
        [{"label": "abka", "manchu": "ᠠᠪᡴᠠ", "path": "abka/1.png"}]))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "*",
        "photos": "photos",
        "models": [{"name": "Test", "model": "Untrained ModelD",
                    "class": "ModelD", "weights": "model.npz",
                    "params": params}],
    }))
    return create_app(str(tmp_path / "config.json")).test_client()


def test_awake(client):
    assert client.get("/awake").status_code == 200


def test_random(client):
    response = client.get("/random")
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    body = response.get_json()
    assert body["label"] == "abka"
    assert body["manchu"] == "ᠠᠪᡴᠠ"
    assert body["image"].startswith("data:image/png;base64,")
    # The preprocessed photo is greyscale and WIDTH pixels wide
    header, encoded = body["preprocessed-image"].split(",")
    assert header == "data:image/png;base64"
    preprocessed = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert preprocessed.mode == "L"
    assert preprocessed.width == WIDTH
    [result] = body["results"]
    assert result["name"] == "Test"
    assert result["model"] == "Untrained ModelD"
    # One CTC token and probability for each real token of the photo
    assert len(result["ctc"]) == len(result["probs"]) > 0
    assert all(token is None or token in alphabet for token in result["ctc"])
    assert all(0 <= prob <= 1 for prob in result["probs"])
    assert 0 <= result["score"] <= 1
