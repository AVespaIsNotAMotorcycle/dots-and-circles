# Server

A Flask server for the frontend, run on Heroku by the `Procfile` at the
repository root. Locally, from the repository root:

```
.venv/bin/python -m flask --app server.app run
```

## Endpoints

- `GET /awake`: an empty 200, so the frontend knows the dyno is up.
- `GET /random`: reads a random image with every model in `config.json`.
  `?source=photo` (the default) picks a photographed word from `photos/`,
  and `?source=synthetic` a rendered one from `synthetic/`. Any other source
  gets a 400.

  ```json
  {
    "source": "photo",
    "font": null,
    "label": "abka",
    "manchu": "ᠠᠪᡴᠠ",
    "image": "data:image/png;base64,...",
    "preprocessed-image": "data:image/png;base64,...",
    "results": [{
      "name": "Run 26",
      "model": "ModelD (8 transformers, ...)",
      "prediction-manchu": "ᠠᠪᡴᠠ",
      "prediction-abkai": "abka",
      "ctc": [null, "ᠠ", "ᠠ", null, "ᠪ", "..."],
      "probs": [0.99, 0.97, 0.88, 0.99, 0.95, "..."],
      "score": 1.0
    }]
  }
  ```

  `source` is where the image came from (`photo`, `synthetic` or, from
  `/upload/single`, `upload`), and `font` the font a synthetic word was
  rendered in (`null` otherwise). `label` is the word in Abkai, `manchu`
  the same in Manchu script, and `image` the photo before preprocessing.
  `preprocessed-image` is the photo as a model with the default
  preprocessing reads it: a greyscale PNG, whitened, cropped to the ink and
  scaled to the models' input width. `ctc` has the most likely class at
  each of the photo's tokens (`null` for the blank), and `probs` that
  class's probability.

  `label` and `prediction-abkai` are Abkai as in
  [Wikipedia's chart](https://en.wikipedia.org/wiki/Transliterations_of_Manchu):
  ᠴ is `q`, and loanword letters carry ‘ (U+2018) after k, g and h and ʻ
  (U+02BB) elsewhere (`k‘`, `rʻ`, `syʻ`). `score`, though, is computed in
  the photo dataset's own romanization (`ocr.photos.manchu_to_roman`), as
  in the run reports: the character accuracy of the prediction against the
  label, `max(0, 1 - edit distance / len(label))`.
- `GET /lexigraphs`: every word of a source, a page at a time, with one
  model's reading of each. `?source=photo` (the default) lists the
  photographed words in `photos/`, and `?source=synthetic` the rendered
  ones in `synthetic/`, both in the order of their `manifest.json`.
  `?page=` counts from 1 (the default), and `?per_page=` is 50 by default
  and at most 100; a page past the last has no `words`. `?model=` picks
  the model, as for `/upload/multiple` (`default_model`, Run 31, by
  default). An unknown source or model, or a bad page or `per_page`, gets
  a 400.

  ```json
  {
    "source": "photo",
    "model": "Run 31",
    "page": 1,
    "per-page": 50,
    "total": 300,
    "pages": 6,
    "words": [{
      "index": 0,
      "label": "abkai",
      "manchu": "ᠠᠪᡴᠠᡳ",
      "font": null,
      "image": "data:image/png;base64,...",
      "prediction-manchu": "ᠠᠪᡴᠠᡳ",
      "prediction-abkai": "abkai",
      "score": 1.0
    }]
  }
  ```

  The fields mean what they do in `/random`, and `index` is the word's
  place in the whole list. Each worker reads a word the first time it's
  asked for and keeps the reading, so a page is slow (about 0.6 s locally
  for 50) only the first time.
- `GET /pages`: the names of the real pages `/random?source=page&name=`
  takes, in order: `{"pages": ["page001", "page002", ...]}`. Each is the
  page's filename without its extension (`page001.png` and its
  transcription `page001.txt`). Empty if the server doesn't read pages
  (no `pages` or `box_classifier` in `config.json`).
- `GET /random?source=page`: reads a random real page, with its known
  transcription, from the folder `pages` names in `config.json` (the
  benchmark pages in `ocr/real_pages`), or the one `?name=` names (e.g.
  `?name=page003`; an unknown name gets a 400 listing them). `?model=`
  picks the model, as for `/upload/multiple`. It returns everything `/upload/multiple` does (with
  `source` `"page"`), and:

  ```json
  {
    "name": "page001",
    "original-image": "data:image/png;base64,...",
    "transcription": "genefi , jugvn asuru ...\ngemungge heqen i ...",
    "score": {"characters": 0.976, "words": 0.873},
    "credit": "Stabi Berlin (Public Domain Mark 1.0); transcription from manc.hu",
    "link": "https://digital.staatsbibliothek-berlin.de/werkansicht?..."
  }
  ```

  `original-image` is the page as saved, and `image` the page deskewed
  as it was read (the boxes' coordinates are in it). `transcription` is the page's transcription
  as written, a line per Manchu column, in Abkai with its punctuation.
  `score` compares the reading with it as `ocr.evaluate_real_pages` does
  (ignoring punctuation, each line matched to the closest line read).
  `credit` and `link` come from the folder's `sources.json` (`null` for a
  page it doesn't list); the Staatsbibliothek asks to be named as "Stabi
  Berlin" with the link wherever its scans are shown. Without `pages` or
  `box_classifier` in `config.json`, `?source=page` gets a 400.
- `POST /upload/single`: reads an uploaded image of one word, written top to
  bottom, and returns the same JSON as `/random`. Send it either as
  `multipart/form-data` with the file in an `image` field, or as a JSON body
  `{"image": "data:image/png;base64,..."}` (bare base64 works too). Either
  may also have a `label`, the word in Abkai. Any of `'`, `’`, `‘` or `ʻ`
  works for the marks, and the response gives it back in the standard
  form. Without one, `label`, `manchu` and every `score` are `null`.

  The image is turned upright by its EXIF orientation, given a white
  background if it's transparent, and shrunk to at most 2000 pixels on its
  longest side. `image` in the response is the upload after those steps, as
  a PNG. Images over 10 MB or 50 million pixels get a 413, and a missing or
  unreadable image or a label that isn't Abkai gets a 400. Both come back
  as `{"error": "..."}`.

- `POST /upload/multiple`: reads a page of print: columns of words written
  top to bottom, read left to right, with Chinese characters among them or
  in columns of their own. Send the image as for `/upload/single` (any
  `label` is ignored). `?model=` picks the model by its `name` in
  `config.json`, `default_model` by default; an unknown name gets a 400.

  ```json
  {
    "model": "Run 31",
    "image": "data:image/png;base64,...",
    "angle": -0.8,
    "manchu": "ᠠᠪᡴᠠ □ ᠨᠠ\nᠪᡝ",
    "abkai": "abka □ na\nbe",
    "boxes": [{
      "x": 120, "y": 40, "w": 46, "h": 210,
      "image": "data:image/png;base64,...",
      "line": 0, "position": 0, "kind": "manchu",
      "manchu-probability": 0.999,
      "prediction-manchu": "ᠠᠪᡴᠠ", "prediction-abkai": "abka",
      "confidence": 0.97, "low-confidence": false
    }]
  }
  ```

  The page is found by `ocr/layout.py` (deskewed, ruled lines removed,
  split into columns and words), each box is classed as Manchu or other
  text by the small classifier in `ocr/box_classifier.py`, and Manchu boxes
  are read by the model (see `ocr/page_reader.py`). `image` is the
  page in colour, deskewed by `angle` degrees anticlockwise as it was read,
  with the corners that adds filled with the paper's colour. Nothing is
  drawn on it: each box gives its top left corner (`x`, `y`) and size
  (`w`, `h`) in that image, and its own `image`, the part of the page it
  holds, as a PNG. `boxes` are in reading order. `manchu` and `abkai` have a line
  per column with Manchu in it, words separated by spaces and `□` for each
  other box; columns with no Manchu (such as a column of Chinese) are left
  out, and their boxes' `line` is `null`. `position` counts from the top of
  the box's column, and `manchu-probability` is the classifier's.
  `confidence` is the mean probability of the letters the model read
  (leaving out the blanks between them, which it's nearly always sure of,
  even reading junk). A Manchu box under 0.65 (`MIN_CONFIDENCE` in
  `ocr/page_reader.py`) is marked `low-confidence` and left out of
  `manchu` and `abkai`; a column of nothing else leaves no line, and its
  boxes' `line` is `null`. On the real pages every correct reading scored
  at least 0.7, and most junk (a page's edge read as Manchu, say) at most
  0.57. Other boxes' predictions, confidence and `low-confidence` are
  `null`.

## config.json

- `allowed_origin`: sent as `Access-Control-Allow-Origin`.
- `photos`: the folder of photos, with their `manifest.json`.
- `synthetic`: the folder of synthetic words, with their `manifest.json`.
  Optional: without it, `?source=synthetic` gets a 400.
- `default_model`: the `name` of the model `/upload/multiple` reads with
  unless asked for another. The last model by default.
- `pages`: the folder of real pages with transcriptions for
  `/random?source=page`, relative to this folder (`../ocr/real_pages`).
  Optional.
- `box_classifier`: the weights of the Manchu/other classifier for
  `/upload/multiple`, relative to this folder. Train them with
  `python -m ocr.train_box_classifier`. Optional: without it,
  `/upload/multiple` gets a 404.
- `models`: one entry per model, in the order they're returned:
  - `name`, `model`: the short and long names sent to the frontend.
  - `class`: a CTC model class in `ocr/`, e.g. `ModelD`.
  - `weights`: its `.npz` file, relative to this folder.
  - `params`: keyword arguments for the class besides the ones every model
    shares (`token_dim`, `context_length`, `num_classes`, `blank`), e.g.
    `depth` and `emb_dim`. They must match the weights.
  - `preprocess`: `crop` and `whiten` for `ocr.preprocess.prepare`, both
    true by default.
  - `stem_breaks`: whether `/upload/multiple` splits boxes at stem breaks
    when the model reads the parts more confidently (see
    `ocr.page_reader.split_at_stem_breaks`). True by default. Models
    trained on tight phrases (Run 31 on) write the spaces between close-set
    words themselves, and the stem breaks only cut lone letters off words,
    so they set it false.

## Photos

`photos/` holds one photo each of 300 random words from the photographed
print dataset (`ocr/non_synthetic_data`, which isn't in the repository).
They're all benchmark words (see `ocr/split.py`), so models trained on the
split have never seen them. Run `select_photos.py` again after the photo
split changes. It rebuilds the folder; each photo is saved under its label (the
dataset's romanization, which `score` uses), and its original path in the dataset is kept in `manifest.json` as `source`.
The photos are unmodified.

The photos are from a
[dataset published by Sun Haipeng, Tao Wenhao, and Bi Xiaojun](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

Haipeng SUN, Wenhao TAO, Xiaojun BI. A dataset of Manchu ancient book words
for OCR[J]. China Scientific Data, 2026, 11(1): 1-11. DOI:
10.11922/11-6035.csd.2025.0069.zh.

## Synthetic words

`synthetic/` holds 300 random words from the test split (see
`ocr/split.py`), so models trained on that split have never seen them. Each
one is in a random font and made to look like a photographed print, as in
training. They're saved before preprocessing, so the server prepares them
exactly as it does the photos. The fonts aren't in the repository or on the
server, so `select_synthetic.py` renders the set locally, and the PNGs are
committed. Each word's font and its training seed are in
`manifest.json`. Run it again after the lexicon, fonts or split change.
