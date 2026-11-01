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
from ocr.photos import manchu_to_abkai, manchu_to_roman
from ocr.train import character_accuracy
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
    # And one synthetic word, a lighter bar on grey paper
    (tmp_path / "synthetic").mkdir()
    image = Image.new("L", (64, 200), 200)
    image.paste(60, (24, 10, 40, 190))
    image.save(tmp_path / "synthetic" / "000.png")
    (tmp_path / "synthetic" / "manifest.json").write_text(json.dumps(
        # Labelled in the dataset's romanization, which spells ᠴ c, not q
        [{"label": "acambi", "manchu": "ᠠᠴᠠᠮᠪᡳ", "font": "SomeFont", "seed": 7,
          "path": "000.png"}]))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "*",
        "photos": "photos",
        "synthetic": "synthetic",
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
    assert body["source"] == "photo"
    assert body["font"] is None
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


def bar_image(size=(40, 120), mode="L", format="PNG"):
    # A dark bar on white, like the fixture's photo, as file bytes
    image = Image.new(mode, size, 255 if mode == "L" else (255, 255, 255))
    width, height = size
    image.paste(0 if mode == "L" else (0, 0, 0),
                (width * 3 // 8, height // 12, width * 5 // 8,
                 height * 11 // 12))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def check_upload_body(body, label, manchu):
    assert body["label"] == label
    assert body["manchu"] == manchu
    assert body["image"].startswith("data:image/png;base64,")
    header, encoded = body["preprocessed-image"].split(",")
    preprocessed = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert preprocessed.mode == "L"
    assert preprocessed.width == WIDTH
    [result] = body["results"]
    assert result["name"] == "Test"
    assert len(result["ctc"]) == len(result["probs"]) > 0
    assert body["source"] == "upload"
    assert body["font"] is None
    if label is None:
        assert result["score"] is None
    else:
        assert 0 <= result["score"] <= 1


def test_upload_file(client):
    response = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image()), "word.png")})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    check_upload_body(response.get_json(), None, None)


def test_upload_file_with_label(client):
    response = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image(format="JPEG")), "word.jpg"),
        "label": " abka "})
    assert response.status_code == 200
    check_upload_body(response.get_json(), "abka", "ᠠᠪᡴᠠ")


def test_upload_reads_like_random(client):
    # The same photo read through either endpoint gives the same results
    random_body = client.get("/random").get_json()
    data = base64.b64decode(random_body["image"].split(",")[1])
    upload_body = client.post("/upload/single", json={
        "image": base64.b64encode(data).decode("ascii"),
        "label": "abka"}).get_json()
    assert upload_body["results"] == random_body["results"]
    assert (upload_body["preprocessed-image"]
            == random_body["preprocessed-image"])


def test_upload_json_data_url(client):
    encoded = base64.b64encode(bar_image()).decode("ascii")
    response = client.post("/upload/single", json={
        "image": f"data:image/png;base64,{encoded}"})
    assert response.status_code == 200
    check_upload_body(response.get_json(), None, None)


def test_upload_transparent_background_is_white(client):
    image = Image.new("RGBA", (40, 120), (0, 0, 0, 0))
    image.paste((0, 0, 0, 255), (15, 10, 25, 110))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    body = client.post("/upload/single", data={
        "image": (io.BytesIO(buffer.getvalue()), "word.png")}).get_json()
    encoded = body["image"].split(",")[1]
    uploaded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert uploaded.getpixel((0, 0)) == (255, 255, 255)


def test_upload_follows_exif_orientation(client):
    # Saved sideways, with an EXIF tag saying to turn it upright
    image = Image.open(io.BytesIO(bar_image())).rotate(90, expand=True)
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", exif=exif)
    body = client.post("/upload/single", data={
        "image": (io.BytesIO(buffer.getvalue()), "word.jpg")}).get_json()
    uploaded = Image.open(io.BytesIO(base64.b64decode(
        body["image"].split(",")[1])))
    assert uploaded.height > uploaded.width


def test_upload_shrinks_big_images(client):
    body = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image(size=(1000, 3000))), "big.png")
    }).get_json()
    uploaded = Image.open(io.BytesIO(base64.b64decode(
        body["image"].split(",")[1])))
    assert max(uploaded.size) == 2000


@pytest.mark.parametrize("kwargs", [
    {},
    {"json": {"label": "abka"}},
    {"json": {"image": "not base64!"}},
    {"data": {"image": (io.BytesIO(b"not an image"), "word.png")}},
    {"data": {"image": (io.BytesIO(b""), "word.png")}},
    {"json": {"image": base64.b64encode(bar_image()).decode("ascii"),
              "label": "abka?"}},
    {"json": {"image": base64.b64encode(bar_image()).decode("ascii"),
              "label": 3}},
])
def test_upload_bad_request(client, kwargs):
    response = client.post("/upload/single", **kwargs)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_upload_too_large(client, monkeypatch):
    import server.app
    monkeypatch.setattr(server.app, "MAX_UPLOAD_BYTES", 100)
    response = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image()), "word.png")})
    assert response.status_code == 413
    assert response.get_json()["error"]


def test_upload_preflight(client):
    response = client.options("/upload/single", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "Content-Type" in response.headers["Access-Control-Allow-Headers"]


def test_random_photo_is_the_default(client):
    for query in ("", "?source=", "?source=photo"):
        assert client.get(f"/random{query}").get_json()["source"] == "photo"


def test_random_synthetic(client):
    response = client.get("/random?source=synthetic")
    assert response.status_code == 200
    body = response.get_json()
    assert body["source"] == "synthetic"
    assert body["font"] == "SomeFont"
    # Shown in Abkai, but scored against the manifest's romanization
    assert body["label"] == "aqambi"
    assert body["manchu"] == "ᠠᠴᠠᠮᠪᡳ"
    assert body["image"].startswith("data:image/png;base64,")
    [result] = body["results"]
    assert 0 <= result["score"] <= 1


def test_random_unknown_source(client):
    response = client.get("/random?source=drawing")
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_upload_label_in_abkai(client):
    response = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image()), "word.png"), "label": "aqambi"})
    assert response.status_code == 200
    body = response.get_json()
    check_upload_body(body, "aqambi", "ᠠᠴᠠᠮᠪᡳ")
    # Scored in the dataset's romanization, as in the run reports
    [result] = body["results"]
    guess = manchu_to_roman(result["prediction-manchu"])
    assert result["score"] == pytest.approx(
        character_accuracy(guess, "acambi"))
    assert result["prediction-abkai"] == manchu_to_abkai(
        result["prediction-manchu"])


def test_upload_label_marks_are_normalized(client):
    body = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image()), "word.png"),
        "label": "k'amduri"}).get_json()
    assert body["label"] == "k\u2018amduri"
    assert body["manchu"] == "ᠺᠠᠮᡩᡠᡵᡳ"
