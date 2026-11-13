import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

flask = pytest.importorskip("flask")

from ocr.box_classifier import make_classifier
from ocr.modelD import ModelD
from ocr.training_data import WIDTH
from ocr.train import CONTEXT_LENGTH, END, NUM_CLASSES, TOKEN_DIM, alphabet
from ocr.photos import manchu_to_abkai, manchu_to_norman, manchu_to_roman
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
    # And an untrained box classifier, for /upload/multiple, and a real
    # page with its transcription and source
    make_classifier().save_weights(str(tmp_path / "classifier.npz"))
    (tmp_path / "dictionary.json").write_text(json.dumps(
        [["ᠠᠪᡴᠠ", ["kaikki", "norman"]], ["ᠨᠠ", ["kaikki"]],
         ["ᠪᡝ", ["norman"]], ["ᠠᠴᠠᠮᠪᡳ", ["norman"]]], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "page001.png").write_bytes(page_image())
    (tmp_path / "pages" / "page001.txt").write_text(
        "abka , na\nbe\n", encoding="utf-8")
    (tmp_path / "pages" / "page002.png").write_bytes(page_image())
    (tmp_path / "pages" / "page002.txt").write_text("be\n", encoding="utf-8")
    (tmp_path / "pages" / "sources.json").write_text(json.dumps(
        {"page001": {"credit": "Somewhere", "link": "https://example.org"}}))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "*",
        "photos": "photos",
        "synthetic": "synthetic",
        "box_classifier": "classifier.npz",
        "dictionary": "dictionary.json",
        "pages": "pages",
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
    assert body["label-norman"] == "abka"
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
    assert result["prediction-norman"] == manchu_to_norman(
        result["prediction-manchu"])
    # ᠴ is q in Abkai and c in Norman
    assert body["label-norman"] == "acambi"


def test_upload_label_marks_are_normalized(client):
    body = client.post("/upload/single", data={
        "image": (io.BytesIO(bar_image()), "word.png"),
        "label": "k'amduri"}).get_json()
    assert body["label"] == "k\u2018amduri"
    assert body["manchu"] == "ᠺᠠᠮᡩᡠᡵᡳ"


def page_image(format="PNG"):
    # Three columns of dark bars on white, like words in print
    image = Image.new("L", (300, 400), 255)
    for left in (40, 130, 220):
        for top in (30, 160, 290):
            image.paste(0, (left, top, left + 40, top + 90))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()

def test_upload_multiple(client):
    response = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")})
    assert response.status_code == 200
    body = response.get_json()
    assert body["model"] == "Test"
    assert body["image"].startswith("data:image/jpeg;base64,")
    page = Image.open(io.BytesIO(base64.b64decode(
        body["image"].partition(",")[2])))
    assert page.mode == "RGB" and page.format == "JPEG"
    # The untrained classifier may call bars other text, and split them
    # into square characters
    assert len(body["boxes"]) >= 9
    for box in body["boxes"]:
        assert box["kind"] in ("manchu", "other")
        # The image in the box, cut from the page
        crop = Image.open(io.BytesIO(base64.b64decode(
            box["image"].partition(",")[2])))
        assert crop.size == (box["w"], box["h"])
        assert box["x"] + box["w"] <= page.width
        assert box["y"] + box["h"] <= page.height
        assert {"x", "y", "w", "h", "line", "position", "low-confidence",
                "manchu-probability", "prediction-manchu",
                "prediction-abkai", "confidence"} <= set(box)
        if box["kind"] == "manchu":
            assert box["prediction-abkai"] == manchu_to_abkai(
                box["prediction-manchu"])
            assert box["prediction-norman"] == manchu_to_norman(
                box["prediction-manchu"])
        else:
            assert box["prediction-manchu"] is None
    # Reading order: columns left to right, top to bottom in each
    order = [(box["x"], box["y"]) for box in body["boxes"]]
    assert [x for x, _ in order] == sorted(x for x, _ in order)
    # One line per column with Manchu in it, a word or □ per box
    lines = body["manchu"].split("\n") if body["manchu"] else []
    assert len(lines) == len({box["line"] for box in body["boxes"]
                              if box["line"] is not None})
    assert len(body["abkai"].split("\n") if body["abkai"] else []) == \
        len(lines)

def test_upload_multiple_json_and_model(client):
    encoded = base64.b64encode(page_image("JPEG")).decode("ascii")
    response = client.post("/upload/multiple?model=Test",
                           json={"image": f"data:image/jpeg;base64,{encoded}"})
    assert response.status_code == 200
    assert response.get_json()["model"] == "Test"

def test_upload_multiple_unknown_model(client):
    response = client.post("/upload/multiple?model=Nope", data={
        "image": (io.BytesIO(page_image()), "page.png")})
    assert response.status_code == 400
    assert "Nope" in response.get_json()["error"]

def test_upload_multiple_blank_page(client):
    buffer = io.BytesIO()
    Image.new("L", (200, 300), 240).save(buffer, format="PNG")
    response = client.post("/upload/multiple", data={
        "image": (io.BytesIO(buffer.getvalue()), "blank.png")})
    assert response.status_code == 200
    body = response.get_json()
    assert body["boxes"] == [] and body["manchu"] == "" and \
        body["abkai"] == ""

def test_upload_multiple_bad_image(client):
    response = client.post("/upload/multiple", data={
        "image": (io.BytesIO(b"not an image"), "page.png")})
    assert response.status_code == 400


def test_random_page(client):
    response = client.get("/random?source=page&name=page001")
    assert response.status_code == 200
    body = response.get_json()
    assert body["source"] == "page" and body["name"] == "page001"
    assert body["model"] == "Test"
    assert body["transcription"] == "abka , na\nbe"
    assert body["credit"] == "Somewhere"
    assert body["link"] == "https://example.org"
    assert body["original-image"].startswith("data:image/jpeg;base64,")
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert 0 <= body["score"]["words"] <= 1
    assert body["score"]["characters"] <= 1
    # Everything /upload/multiple gives, too
    assert {"angle", "manchu", "abkai", "boxes"} <= set(body)

def test_random_page_unknown_model(client):
    response = client.get("/random?source=page&model=Nope")
    assert response.status_code == 400

def test_random_page_needs_pages(tmp_path):
    config = {"allowed_origin": "*", "models": []}
    np.random.seed(0)
    params = {"depth": 1, "emb_dim": 8}
    ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
           num_classes=NUM_CLASSES, blank=END, **params).save_weights(
               str(tmp_path / "model.npz"))
    config["models"] = [{"name": "Test", "model": "Untrained ModelD",
                         "class": "ModelD", "weights": "model.npz",
                         "params": params}]
    (tmp_path / "config.json").write_text(json.dumps(config))
    client = create_app(str(tmp_path / "config.json")).test_client()
    response = client.get("/random?source=page")
    assert response.status_code == 400

def test_random_page_by_name(client):
    for name, text in (("page001", "abka , na\nbe"), ("page002", "be")):
        body = client.get(f"/random?source=page&name={name}").get_json()
        assert body["name"] == name and body["transcription"] == text
    # Without a name, either
    names = {client.get("/random?source=page").get_json()["name"]
             for _ in range(20)}
    assert names <= {"page001", "page002"}

def test_random_page_unknown_name(client):
    response = client.get("/random?source=page&name=page999")
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "page999" in error and "page001" in error and "page002" in error

def test_pages(client):
    response = client.get("/pages")
    assert response.status_code == 200
    names = response.get_json()["pages"]
    assert names == ["page001", "page002"]
    # Each one can be asked for
    for name in names:
        assert client.get(f"/random?source=page&name={name}").status_code \
            == 200

def test_pages_without_pages(tmp_path):
    np.random.seed(0)
    params = {"depth": 1, "emb_dim": 8}
    ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
           num_classes=NUM_CLASSES, blank=END, **params).save_weights(
               str(tmp_path / "model.npz"))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "*",
        "models": [{"name": "Test", "model": "Untrained ModelD",
                    "class": "ModelD", "weights": "model.npz",
                    "params": params}]}))
    client = create_app(str(tmp_path / "config.json")).test_client()
    assert client.get("/pages").get_json() == {"pages": []}


def test_lexigraphs_photo_is_the_default(client):
    for query in ("", "?source=", "?source=photo"):
        response = client.get(f"/lexigraphs{query}")
        assert response.status_code == 200
        body = response.get_json()
        assert body["source"] == "photo"
        assert body["model"] == "Test"
        assert (body["page"], body["per-page"]) == (1, 50)
        assert (body["total"], body["pages"]) == (1, 1)
        [word] = body["words"]
        assert word["index"] == 0
        assert word["label"] == "abka" and word["manchu"] == "ᠠᠪᡴᠠ"
        assert word["font"] is None
        assert word["image"].startswith("data:image/png;base64,")
        assert set(word) == {"index", "label", "label-norman", "manchu",
                             "font", "image", "prediction-manchu",
                             "prediction-abkai", "prediction-norman",
                             "score"}


def test_lexigraphs_read_like_random(client):
    # The same reading, and score, as /random gives the same word
    [word] = client.get("/lexigraphs?source=synthetic").get_json()["words"]
    [result] = client.get("/random?source=synthetic").get_json()["results"]
    assert word["label"] == "aqambi" and word["font"] == "SomeFont"
    for key in ("prediction-manchu", "prediction-abkai", "score"):
        assert word[key] == result[key]


def test_lexigraphs_pages(client, tmp_path):
    # Three photos, two to a page
    manifest = json.loads((tmp_path / "photos" / "manifest.json").read_text())
    manifest = [dict(manifest[0], label=label) for label in ("a", "b", "c")]
    (tmp_path / "photos" / "manifest.json").write_text(json.dumps(manifest))
    client = create_app(str(tmp_path / "config.json")).test_client()
    first = client.get("/lexigraphs?per_page=2").get_json()
    second = client.get("/lexigraphs?per_page=2&page=2").get_json()
    past = client.get("/lexigraphs?per_page=2&page=3").get_json()
    assert (first["total"], first["pages"]) == (3, 2)
    assert [w["index"] for w in first["words"]] == [0, 1]
    assert [w["index"] for w in second["words"]] == [2]
    assert past["words"] == []


def test_lexigraphs_model(client):
    assert client.get("/lexigraphs?model=Test").status_code == 200
    response = client.get("/lexigraphs?model=Nope")
    assert response.status_code == 400
    assert "Test" in response.get_json()["error"]


@pytest.mark.parametrize("query", ["source=drawing", "page=0", "page=x",
                                   "per_page=0", "per_page=101"])
def test_lexigraphs_bad_request(client, query):
    response = client.get(f"/lexigraphs?{query}")
    assert response.status_code == 400
    assert response.get_json()["error"]

def straight_page(client):
    # The straightened page /upload/multiple returns, and its boxes
    body = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")}).get_json()
    return body["image"], body["boxes"]

def test_upload_multiple_boxes(client):
    image, _ = straight_page(client)
    boxes = [
        # Out of reading order, one reaching past the image
        {"x": 220, "y": 30, "w": 40, "h": 90, "kind": "manchu"},
        {"x": 40, "y": 160, "w": 40, "h": 220, "kind": "other"},
        {"x": 35, "y": 25, "w": 50, "h": 100},
        {"x": 280, "y": 350, "w": 100, "h": 100, "kind": None},
    ]
    response = client.post("/upload/multiple/boxes?model=Test",
                           json={"image": image, "boxes": boxes})
    assert response.status_code == 200
    body = response.get_json()
    assert body["model"] == "Test" and body["angle"] == 0
    page = Image.open(io.BytesIO(base64.b64decode(image.partition(",")[2])))
    returned = Image.open(io.BytesIO(base64.b64decode(
        body["image"].partition(",")[2])))
    assert returned.size == page.size
    # The boxes as sent, cut to the image, in reading order
    assert [(box["x"], box["y"], box["w"], box["h"])
            for box in body["boxes"]] == [
        (35, 25, 50, 100), (40, 160, 40, 220), (220, 30, 40, 90),
        (280, 350, page.width - 280, page.height - 350)]
    assert [box["position"] for box in body["boxes"]] == [0, 1, 0, 0]
    assert body["boxes"][1]["kind"] == "other"
    assert body["boxes"][1]["prediction-manchu"] is None
    # Marked Manchu by hand, so in the text however unsure the model is
    marked = body["boxes"][2]
    assert marked["kind"] == "manchu" and marked["line"] is not None
    for box in body["boxes"]:
        assert 0 <= box["manchu-probability"] <= 1

def test_upload_multiple_boxes_form(client):
    image, boxes = straight_page(client)
    data = base64.b64decode(image.partition(",")[2])
    response = client.post("/upload/multiple/boxes", data={
        "image": (io.BytesIO(data), "page.png"),
        "boxes": json.dumps([{key: box[key] for key in
                              ("x", "y", "w", "h", "kind")}
                             for box in boxes])})
    assert response.status_code == 200
    # Sending back the boxes found reads them in the same boxes
    key = lambda box: (box["x"], box["y"], box["w"], box["h"], box["kind"])
    assert sorted(map(key, response.get_json()["boxes"])) == \
        sorted(map(key, boxes))

def test_upload_multiple_boxes_none(client):
    image, _ = straight_page(client)
    response = client.post("/upload/multiple/boxes",
                           json={"image": image, "boxes": []})
    assert response.status_code == 200
    body = response.get_json()
    assert body["boxes"] == [] and body["manchu"] == ""

@pytest.mark.parametrize("boxes", [
    None, "nope", [1], [{"x": 0, "y": 0, "w": 10}],
    [{"x": 0, "y": 0, "w": 0, "h": 10}],
    [{"x": 0, "y": 0, "w": True, "h": 10}],
    [{"x": 5000, "y": 0, "w": 10, "h": 10}],
    [{"x": 0, "y": 0, "w": 10, "h": 10, "kind": "chinese"}],
    # A reading needs both its text and confidence
    [{"x": 0, "y": 0, "w": 10, "h": 10, "prediction-manchu": "ᠠ"}],
    [{"x": 0, "y": 0, "w": 10, "h": 10, "confidence": 0.9}],
    [{"x": 0, "y": 0, "w": 10, "h": 10, "prediction-manchu": "ᠠ",
      "confidence": 2}],
])
def test_upload_multiple_boxes_bad_boxes(client, boxes):
    image, _ = straight_page(client)
    body = {"image": image}
    if boxes is not None:
        body["boxes"] = boxes
    response = client.post("/upload/multiple/boxes", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_upload_multiple_boxes_too_many(client, monkeypatch):
    import server.app as app
    monkeypatch.setattr(app, "MAX_BOXES", 1)
    image, _ = straight_page(client)
    box = {"x": 0, "y": 0, "w": 10, "h": 10}
    response = client.post("/upload/multiple/boxes",
                           json={"image": image, "boxes": [box, box]})
    assert response.status_code == 400

def test_upload_multiple_boxes_is_not_shrunk(client):
    # A page a little over MAX_UPLOAD_SIDE, as turning one can make it
    from server.app import MAX_STRAIGHT_SIDE, MAX_UPLOAD_SIDE
    buffer = io.BytesIO()
    Image.new("L", (MAX_UPLOAD_SIDE + 50, 100), 255).save(buffer, "PNG")
    box = {"x": MAX_UPLOAD_SIDE, "y": 0, "w": 50, "h": 100,
           "kind": "other"}
    response = client.post("/upload/multiple/boxes", data={
        "image": (io.BytesIO(buffer.getvalue()), "page.png"),
        "boxes": json.dumps([box])})
    assert response.status_code == 200
    assert response.get_json()["boxes"][0]["x"] == MAX_UPLOAD_SIDE
    buffer = io.BytesIO()
    Image.new("L", (MAX_STRAIGHT_SIDE + 1, 10), 255).save(buffer, "PNG")
    response = client.post("/upload/multiple/boxes", data={
        "image": (io.BytesIO(buffer.getvalue()), "page.png"),
        "boxes": "[]"})
    assert response.status_code == 413

def test_upload_multiple_boxes_keeps_readings_sent_back(client):
    first = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")}).get_json()
    # Every box sent back as it came, with a made-up reading for one
    # Manchu box, as if the frontend kept it
    sent = [dict(box) for box in first["boxes"]]
    manchu = [box for box in sent if box["kind"] == "manchu"]
    if not manchu:
        sent[0].update(kind="manchu")
        manchu = [sent[0]]
    manchu[0].update({"prediction-manchu": "ᠠᠪᡴᠠ", "confidence": 0.9})
    response = client.post("/upload/multiple/boxes",
                           json={"image": first["image"], "boxes": sent})
    assert response.status_code == 200
    key = lambda box: (box["x"], box["y"], box["w"], box["h"])
    returned = {key(box): box for box in response.get_json()["boxes"]}
    kept = returned[key(manchu[0])]
    assert kept["prediction-manchu"] == "ᠠᠪᡴᠠ"
    assert kept["prediction-abkai"] == "abka" and kept["confidence"] == 0.9
    for box in sent:
        assert returned[key(box)]["prediction-manchu"] == \
            box["prediction-manchu"]

def test_page_boxes_have_closest_words(client):
    first = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")}).get_json()
    # A Manchu box sent back with a reading, to know what it'll match
    box = dict(first["boxes"][0], kind="manchu",
               **{"prediction-manchu": "ᠠᠪᡴᠠ", "confidence": 0.9})
    other = dict(first["boxes"][1], kind="other")
    body = client.post("/upload/multiple/boxes", json={
        "image": first["image"], "boxes": [box, other]}).get_json()
    by_kind = {box["kind"]: box for box in body["boxes"]}
    assert by_kind["manchu"]["closest-words"] == [
        {"manchu": "ᠠᠪᡴᠠ", "abkai": "abka", "norman": "abka", "distance": 0,
         "sources": ["kaikki", "norman"], "lemma": None, "affixes": None},
        # A form of ᠠᠴᠠᠮᠪᡳ (acambi), closer than ᠨᠠ and ᠪᡝ (3 edits)
        {"manchu": "ᠠᠴᠠᡴᠠ", "abkai": "aqaka", "norman": "acaka", "distance": 2,
         "sources": ["norman"],
         "lemma": {"manchu": "ᠠᠴᠠᠮᠪᡳ", "abkai": "aqambi", "norman": "acambi"},
         "affixes": ["-ka"]}]
    assert by_kind["other"]["closest-words"] is None
    for box in first["boxes"]:
        if box["kind"] == "manchu":
            assert len(box["closest-words"]) == (
                2 if box["prediction-manchu"].strip(" ᠈᠉") else 0)
        else:
            assert box["closest-words"] is None

def test_page_boxes_closest_verb_forms(client):
    first = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")}).get_json()
    box = dict(first["boxes"][0], kind="manchu",
               **{"prediction-manchu": "ᠠᠴᠠᠪᡠᡥᠠ", "confidence": 0.9})
    body = client.post("/upload/multiple/boxes", json={
        "image": first["image"], "boxes": [box]}).get_json()
    assert body["boxes"][0]["closest-words"][0] == {
        "manchu": "ᠠᠴᠠᠪᡠᡥᠠ", "abkai": "aqabuha", "norman": "acabuha",
        "distance": 0, "sources": ["norman"],
        "lemma": {"manchu": "ᠠᠴᠠᠮᠪᡳ", "abkai": "aqambi", "norman": "acambi"},
        "affixes": ["-bu-", "-ha"]}

def test_page_text_in_norman(client):
    first = client.post("/upload/multiple", data={
        "image": (io.BytesIO(page_image()), "page.png")}).get_json()
    # Two columns: a Manchu word with ᡧ and ᡡ above other text, and one word
    boxes = [
        dict(first["boxes"][0], kind="manchu", x=40, y=30, w=40, h=90,
             **{"prediction-manchu": "ᡧᡠᠨ ᠠᡴᡡ", "confidence": 0.9}),
        dict(first["boxes"][1], kind="other", x=40, y=160, w=40, h=90),
        dict(first["boxes"][2], kind="manchu", x=130, y=30, w=40, h=90,
             **{"prediction-manchu": "ᠠᠴᠠᠮᠪᡳ", "confidence": 0.9}),
    ]
    body = client.post("/upload/multiple/boxes", json={
        "image": first["image"], "boxes": boxes}).get_json()
    assert body["abkai"] == "xun akv □\naqambi"
    assert body["norman"] == "šun akū □\nacambi"
    assert [box["prediction-norman"] for box in body["boxes"]] == [
        "šun akū", None, "acambi"]
    for box in body["boxes"]:
        for match in box["closest-words"] or []:
            assert match["norman"] == manchu_to_norman(match["manchu"])

def test_lexigraphs_have_norman(client):
    [word] = client.get("/lexigraphs").get_json()["words"]
    assert word["label-norman"] == manchu_to_norman("ᠠᠪᡴᠠ")
    assert word["prediction-norman"] == manchu_to_norman(
        word["prediction-manchu"])


def skewed_page_image():
    # page_image turned 3 degrees, so the layout straightens it
    page = Image.open(io.BytesIO(page_image())).rotate(
        3, expand=True, fillcolor=255)
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()

def test_page_crops_can_be_left_out(client):
    for query, crops in (("", True), ("?crops=true", True),
                         ("?crops=false", False), ("?crops=0", False)):
        body = client.post(f"/upload/multiple{query}", data={
            "image": (io.BytesIO(page_image()), "page.png")}).get_json()
        assert body["boxes"]
        for box in body["boxes"]:
            assert (box["image"].startswith("data:image/png;base64,")
                    if crops else box["image"] is None)
    body = client.get("/random?source=page&crops=false").get_json()
    assert all(box["image"] is None for box in body["boxes"])
    response = client.post("/upload/multiple?crops=maybe", data={
        "image": (io.BytesIO(page_image()), "page.png")})
    assert response.status_code == 400

def test_upload_multiple_boxes_with_the_upload_and_angle(client):
    data = skewed_page_image()
    first = client.post("/upload/multiple", data={
        "image": (io.BytesIO(data), "page.png")}).get_json()
    assert first["angle"] != 0
    boxes = [{key: box[key] for key in ("x", "y", "w", "h", "kind")}
             for box in first["boxes"]]
    encoded = base64.b64encode(data).decode("ascii")
    response = client.post("/upload/multiple/boxes", json={
        "image": f"data:image/png;base64,{encoded}",
        "angle": first["angle"], "boxes": boxes})
    assert response.status_code == 200
    body = response.get_json()
    # Turned as before: the same page, the same boxes, and the same angle
    # to send next time
    assert body["angle"] == first["angle"]
    size = lambda body: Image.open(io.BytesIO(base64.b64decode(
        body["image"].partition(",")[2]))).size
    assert size(body) == size(first)
    key = lambda box: (box["x"], box["y"], box["w"], box["h"], box["image"])
    assert sorted(map(key, body["boxes"])) == sorted(map(key, first["boxes"]))
    # And as a form
    response = client.post("/upload/multiple/boxes", data={
        "image": (io.BytesIO(data), "page.png"),
        "angle": str(first["angle"]), "boxes": json.dumps(boxes)})
    assert response.status_code == 200
    assert response.get_json()["angle"] == first["angle"]

def test_upload_multiple_boxes_by_page_name(client):
    first = client.get("/random?source=page&name=page001").get_json()
    boxes = [{key: box[key] for key in ("x", "y", "w", "h", "kind")}
             for box in first["boxes"]]
    response = client.post("/upload/multiple/boxes", json={
        "name": "page001", "angle": first["angle"], "boxes": boxes})
    assert response.status_code == 200
    body = response.get_json()
    key = lambda box: (box["x"], box["y"], box["w"], box["h"], box["image"])
    assert sorted(map(key, body["boxes"])) == sorted(map(key, first["boxes"]))
    response = client.post("/upload/multiple/boxes", json={
        "name": "page999", "boxes": boxes})
    assert response.status_code == 400
    assert "page001" in response.get_json()["error"]

@pytest.mark.parametrize("angle", [90, -46, "3", True, float("nan")])
def test_upload_multiple_boxes_bad_angle(client, angle):
    encoded = base64.b64encode(page_image()).decode("ascii")
    response = client.post("/upload/multiple/boxes", data=json.dumps({
        "image": encoded, "angle": angle, "boxes": []}),
        content_type="application/json")
    assert response.status_code == 400


def cors_client(tmp_path, monkeypatch, on_heroku):
    # The fixture's server, allowing only the frontend's origin
    if on_heroku:
        monkeypatch.setenv("DYNO", "web.1")
    else:
        monkeypatch.delenv("DYNO", raising=False)
    np.random.seed(0)
    ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
           num_classes=NUM_CLASSES, blank=END, depth=1,
           emb_dim=8).save_weights(str(tmp_path / "model.npz"))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "https://frontend.example",
        "models": [{"name": "Test", "model": "Untrained ModelD",
                    "class": "ModelD", "weights": "model.npz",
                    "params": {"depth": 1, "emb_dim": 8}}],
    }))
    return create_app(str(tmp_path / "config.json")).test_client()

def allowed(client, origin):
    headers = {"Origin": origin} if origin else {}
    response = client.get("/awake", headers=headers)
    assert "Origin" in response.headers.get("Vary", "")
    return response.headers.get("Access-Control-Allow-Origin")

LOCAL_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:5173",
                 "http://[::1]:8080", "https://localhost"]
OTHER_ORIGINS = ["https://example.com", "http://localhost.example.com",
                 "https://frontend.example.evil", "null"]

def test_cors_on_heroku_allows_only_the_frontend(tmp_path, monkeypatch):
    client = cors_client(tmp_path, monkeypatch, on_heroku=True)
    assert allowed(client, "https://frontend.example") == \
        "https://frontend.example"
    for origin in LOCAL_ORIGINS + OTHER_ORIGINS + [None]:
        assert allowed(client, origin) is None
    # Preflights too
    response = client.options("/upload/multiple", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"})
    assert "Access-Control-Allow-Origin" not in response.headers

def test_cors_locally_allows_localhost_too(tmp_path, monkeypatch):
    client = cors_client(tmp_path, monkeypatch, on_heroku=False)
    assert allowed(client, "https://frontend.example") == \
        "https://frontend.example"
    for origin in LOCAL_ORIGINS:
        assert allowed(client, origin) == origin
    for origin in OTHER_ORIGINS + [None]:
        assert allowed(client, origin) is None

def test_only_frontend_requests_are_counted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYNO", "web.1")
    np.random.seed(0)
    ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
           num_classes=NUM_CLASSES, blank=END, depth=1,
           emb_dim=8).save_weights(str(tmp_path / "model.npz"))
    (tmp_path / "photos" / "abka").mkdir(parents=True)
    image = Image.new("L", (40, 120), 255)
    image.paste(0, (15, 10, 25, 110))
    image.save(tmp_path / "photos" / "abka" / "1.png")
    (tmp_path / "photos" / "manifest.json").write_text(json.dumps(
        [{"label": "abka", "manchu": "ᠠᠪᡴᠠ", "path": "abka/1.png"}]))
    (tmp_path / "config.json").write_text(json.dumps({
        "allowed_origin": "https://frontend.example", "photos": "photos",
        "models": [{"name": "Test", "model": "Untrained ModelD",
                    "class": "ModelD", "weights": "model.npz",
                    "params": {"depth": 1, "emb_dim": 8}}],
    }))
    (tmp_path / "visitors").mkdir()
    client = create_app(str(tmp_path / "config.json"),
                        visitors_folder=str(tmp_path / "visitors")
                        ).test_client()
    frontend = {"Origin": "https://frontend.example", "User-Agent": "Firefox"}
    # Not counted: no Origin (curl, scripts), another site, preflights
    client.get("/awake")
    client.get("/awake", headers={"Origin": "https://example.com"})
    client.options("/random", headers=frontend)
    assert "visitors" not in capsys.readouterr().out
    # Counted: one visitor, through Heroku's router, then reading a word
    forwarded = dict(frontend, **{"X-Forwarded-For": "6.6.6.6, 1.2.3.4"})
    client.get("/awake", headers=forwarded)
    assert capsys.readouterr().out.strip().endswith(": 1 (0 read something)")
    client.get("/random", headers=forwarded)
    assert capsys.readouterr().out.strip().endswith(": 1 (1 read something)")
    # The same browser at another address is another visitor; the address
    # the client claims (the first one) doesn't matter
    client.get("/awake", headers=dict(frontend, **{
        "X-Forwarded-For": "6.6.6.6, 5.6.7.8"}))
    assert capsys.readouterr().out.strip().endswith(": 2 (1 read something)")
    client.get("/awake", headers=dict(frontend, **{
        "X-Forwarded-For": "7.7.7.7, 1.2.3.4"}))
    assert "visitors" not in capsys.readouterr().out
    # A failed read isn't reading something
    client.get("/random?source=nope", headers=dict(frontend, **{
        "X-Forwarded-For": "5.6.7.8"}))
    assert "visitors" not in capsys.readouterr().out
