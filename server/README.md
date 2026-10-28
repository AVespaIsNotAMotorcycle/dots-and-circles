# Server

A Flask server for the frontend, run on Heroku by the `Procfile` at the
repository root. Locally, from the repository root:

```
.venv/bin/python -m flask --app server.app run
```

## Endpoints

- `GET /awake`: an empty 200, so the frontend knows the dyno is up.
- `GET /random`: reads a random photo from `photos/` with every model in
  `config.json`:

  ```json
  {
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

  `label` is the word in Abkai, `manchu` the same in Manchu script, and
  `image` the photo before preprocessing. `preprocessed-image` is the photo
  as a model with the default preprocessing reads it: a greyscale PNG,
  whitened, cropped to the ink and scaled to the models' input width. `ctc` has the most likely class at
  each of the photo's tokens (`null` for the blank), and `probs` that class's
  probability. `score` is the character accuracy of `prediction-abkai`
  against `label`: `max(0, 1 - edit distance / len(label))`, as in the run
  reports.

## config.json

- `allowed_origin`: sent as `Access-Control-Allow-Origin`.
- `photos`: the folder of photos, with their `manifest.json`.
- `models`: one entry per model, in the order they're returned:
  - `name`, `model`: the short and long names sent to the frontend.
  - `class`: a CTC model class in `ocr/`, e.g. `ModelD`.
  - `weights`: its `.npz` file, relative to this folder.
  - `params`: keyword arguments for the class besides the ones every model
    shares (`token_dim`, `context_length`, `num_classes`, `blank`), e.g.
    `depth` and `emb_dim`. They must match the weights.
  - `preprocess`: `crop` and `whiten` for `ocr.preprocess.prepare`, both
    true by default.

## Photos

`photos/` holds one photo each of 300 random words from the photographed
print dataset (`ocr/non_synthetic_data`, which isn't in the repository).
`select_photos.py` rebuilds it; each photo is saved under its Abkai label,
and its original path in the dataset is kept in `manifest.json` as `source`.
The photos are unmodified.

The photos are from a
[dataset published by Sun Haipeng, Tao Wenhao, and Bi Xiaojun](https://www.scidb.cn/en/detail?dataSetId=b45491b63d694534a9323acf14846586),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

Haipeng SUN, Wenhao TAO, Xiaojun BI. A dataset of Manchu ancient book words
for OCR[J]. China Scientific Data, 2026, 11(1): 1-11. DOI:
10.11922/11-6035.csd.2025.0069.zh.
