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
    "label-norman": "abka",
    "manchu": "ᠠᠪᡴᠠ",
    "image": "data:image/png;base64,...",
    "preprocessed-image": "data:image/png;base64,...",
    "results": [{
      "name": "Run 26",
      "model": "ModelD (8 transformers, ...)",
      "prediction-manchu": "ᠠᠪᡴᠠ",
      "prediction-abkai": "abka",
      "prediction-norman": "abka",
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

  `label-norman` and `prediction-norman` are the same in Jerry Norman's
  romanization, as in his Comprehensive Manchu-English Dictionary
  (`ocr.photos.manchu_to_norman`): ᡧ `š`, ᡡ `ū`, ᠴ `c`, ᡰ `ž`, ᡮ `ts`,
  ᡯ `dz`, `'` after the loanword letters (`k'`, `g'`, `h'`), and the
  apical vowel as he prints it (`sy`, `cy`, `jy`, and `ts` and `dz` alone,
  as in `taidz`). ᡱ and ᡷ before another vowel, which his dictionary
  never has, are `c'` and `j'`. Every field with Abkai in any response
  has a Norman one beside it like this, except a real page's
  `transcription`, which stays as written. Uploaded labels are still
  typed in Abkai.
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
      "label-norman": "abkai",
      "manchu": "ᠠᠪᡴᠠᡳ",
      "font": null,
      "image": "data:image/png;base64,...",
      "prediction-manchu": "ᠠᠪᡴᠠᡳ",
      "prediction-abkai": "abkai",
      "prediction-norman": "abkai",
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
    "original-image": "data:image/jpeg;base64,...",
    "transcription": "genefi , jugvn asuru ...\ngemungge heqen i ...",
    "score": {"characters": 0.976, "words": 0.873},
    "credit": "Stabi Berlin (Public Domain Mark 1.0); transcription from manc.hu",
    "link": "https://digital.staatsbibliothek-berlin.de/werkansicht?..."
  }
  ```

  `original-image` is the page as saved (shrunk to 2000 pixels on its
  longest side, as a JPEG), and `image` the page deskewed as it was read
  (the boxes' coordinates are in it). `?crops=false` leaves out the
  boxes' images, as for `/upload/multiple`. `transcription` is the page's transcription
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

  The server shrinks pages to 2000 pixels on their longest side, so a
  frontend can save upload time by doing it first: a phone photo of
  3-5 MB is about 0.3 MB at 2000 pixels as a JPEG. Use a quality of 0.92
  or so. JPEG changes the readings a little (on the real pages, 87.9% of
  characters right from the PNGs, 87.6% from JPEGs at 92 and 87.5% at
  85, single pages moving a few points either way), so a page that's
  already small, or a PNG scan, is best sent as it is. Whatever is sent,
  keep it: `/upload/multiple/boxes` takes it back.

  ```json
  {
    "model": "Run 31",
    "image": "data:image/jpeg;base64,...",
    "angle": -0.8,
    "manchu": "ᠠᠪᡴᠠ □ ᠨᠠ\nᠪᡝ",
    "abkai": "abka □ na\nbe",
    "norman": "abka □ na\nbe",
    "boxes": [{
      "x": 120, "y": 40, "w": 46, "h": 210,
      "image": "data:image/png;base64,...",
      "line": 0, "position": 0, "kind": "manchu",
      "manchu-probability": 0.999,
      "prediction-manchu": "ᠠᠪᡴᠠ", "prediction-abkai": "abka",
      "prediction-norman": "abka",
      "confidence": 0.97, "low-confidence": false,
      "closest-words": [
        {"manchu": "ᠠᠪᡴᠠ", "abkai": "abka", "norman": "abka", "distance": 0,
         "sources": ["kaikki", "norman"], "lemma": null, "affixes": null},
        {"manchu": "ᠠᠪᠰᠠ", "abkai": "absa", "norman": "absa", "distance": 1,
         "sources": ["norman"], "lemma": null, "affixes": null}
      ]
    }]
  }
  ```

  The page is found by `ocr/layout.py` (deskewed, ruled lines removed,
  split into columns and words), each box is classed as Manchu or other
  text by the small classifier in `ocr/box_classifier.py`, and Manchu boxes
  are read by the model (see `ocr/page_reader.py`). `image` is the
  page in colour, deskewed by `angle` degrees anticlockwise as it was read,
  with the corners that adds filled with the paper's colour, as a JPEG
  (quality 85, `PAGE_JPEG_QUALITY` in `server/app.py`): it's for showing,
  a tenth of the size of a PNG. Nothing is drawn on it: each box gives its
  top left corner (`x`, `y`) and size (`w`, `h`) in that image, and its
  own `image`, the part of the page it holds, as a PNG. `?crops=false`
  leaves the boxes' images out (`null`): they're most of what's left of
  the response, and a frontend can cut them from the page itself (on a
  canvas, say). With them, a page's response is about 0.5-1 MB, and
  without, 80-360 kB. Anything but `true`, `false`, `1`, `0`, `yes` or
  `no` gets a 400. `boxes` are in reading order. `manchu`, `abkai` and
  `norman` (Norman's romanization, see `/random`) have a line per column
  with Manchu in it, words separated by spaces and `□` for each
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

  `closest-words` are the two dictionary entries closest to a Manchu
  box's reading, closest first, to show beside it (the reading itself is
  never changed: the models must read words in no dictionary). They come
  from Wiktionary (`kaikki`) and Norman's dictionary (`norman`), words
  and phrases but not affixes (see `dictionary` below). `distance` is the
  edit distance in the photo dataset's romanization, as `score` uses,
  ignoring punctuation at the end of a word (`aljara,` counts as
  `aljara`, but `al.jara` stays as it is), and a tie goes to the entry nearer
  the reading's length. A box read as nothing has none (`[]`), and other
  boxes, or every box without a `dictionary`, have `null`. The
  dictionaries list verbs by their `-mbi` form, so verbs' other forms are
  matched too: the stem with up to two derivational infixes and a suffix
  (`gene-fi`, `ara-bu-ha`), from the lists in `ocr/affixes.py`. Such a
  match is the form itself (`genefi`), with its verb's entry as `lemma`
  (`manchu`, `abkai` and `norman`, as `{"manchu": "ᡤᡝᠨᡝᠮᠪᡳ", "abkai":
  "genembi", "norman": "genembi"}`) and the affixes on its stem, in
  Norman's romanization, as `affixes` (`["-fi"]`); `sources` are the
  verb's. Entries have `null` for both, and a form spelled like an entry
  is that entry. Vowel harmony isn't checked, so `araho` matches as well
  as `araha`. Each verb gives at most one form, its closest, and on a tie
  entries come first, then forms with fewer infixes. Matching takes about
  5 ms a box (up to 20 ms for long readings).

- `POST /upload/multiple/boxes`: reads a page again in boxes the user has
  edited (removed, moved, resized or added), in the coordinates of the
  page as the last response gave it. Send the page one of three ways:

  - The image sent to `/upload/multiple`, again, with the `angle` its
    response gave. The server opens and turns it exactly as it did then,
    so the page is the same to the pixel.
  - For a page from `/random?source=page`, its `name` and `angle`, and no
    image: the server has the page already.
  - With no `angle`, an image already straightened, taken as it is (not
    turned or shrunk). This was the only way at first. Don't send back the
    response's `image` like this: it's a JPEG, which reads a little
    differently (about one box in seven on the real pages).

  ```json
  {
    "image": "data:image/jpeg;base64,...the upload...",
    "angle": -0.8,
    "boxes": [
      {"x": 120, "y": 40, "w": 46, "h": 210, "kind": "manchu",
       "prediction-manchu": "ᠠᠪᡴᠠ", "confidence": 0.97},
      {"x": 180, "y": 40, "w": 44, "h": 120, "kind": "manchu"},
      {"x": 230, "y": 60, "w": 40, "h": 40}
    ]
  }
  ```

  It can also be sent as `multipart/form-data`, with the file in `image`,
  the angle in `angle` and the boxes as a JSON string in a `boxes` field
  (`name` only works in JSON). `?model=` and `?crops=` work as for
  `/upload/multiple`. The response is the same as `/upload/multiple`'s,
  with the `angle` the page was turned by (the one sent, or 0), to send
  next time. The boxes stay where they were sent (cut to the page if they
  reach past it), in reading order. The server finds the columns itself: boxes overlapping by
  at least half the narrower one's width share one, read left to right,
  and each column is read top to bottom.

  Each box needs `x`, `y`, `w` and `h`, and may have:

  - `kind`: `"manchu"` or `"other"`. Left out or `null`, the classifier
    decides.
  - `prediction-manchu` and `confidence`, as the last response gave them:
    the box isn't read again, and keeps that reading. Send a box back
    unchanged with them, and leave them out for one the user drew, moved,
    resized or changed the kind of. Boxes `/upload/multiple` made by
    splitting one box at the spaces the model read come out differently
    when each part is read alone, so this keeps the words the user didn't
    touch from changing, and saves reading them again. `null` counts as
    left out, and any other keys are ignored, so boxes can be sent back
    just as they came.

  A box isn't split, even if the model reads spaces in it: its
  `prediction-manchu` then has the spaces. A box taller than a few words
  is squeezed to fit what the model reads at once, so a box over a whole
  column reads badly. A Manchu box read again with low confidence is
  still marked `low-confidence`, but if it was sent with `kind`
  `"manchu"` it stays in `manchu` and `abkai`: the user asked for it.
  One sent back with its reading follows the usual rule.

  An empty `boxes` gets a page with none. An already straightened page
  over 2829 pixels on a side (2000 turned by 45°, more than
  `/upload/multiple` ever returns) gets a 413. More than 2000 boxes, a box
  that isn't valid, an `angle` that isn't a number from -45 to 45, or a
  `name` that isn't one of `GET /pages` gets a 400, as
  `{"error": "..."}`.

## config.json

- `allowed_origin`: the one origin whose pages may read the server's
  responses (the frontend, `https://dots-and-circles.sasha-m-ebersole.com`),
  or `"*"` for any. A request from it gets it back as
  `Access-Control-Allow-Origin`, and one from anywhere else gets none, so
  the browser keeps the response from the page. Run locally (anywhere
  Heroku's `DYNO` variable isn't set), the server also answers pages on
  `localhost`, `127.0.0.1` or `[::1]`, on any port, so a frontend in
  development can use it. This only governs browsers: anything else
  (`curl`, a script) can still call the server from anywhere.
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
- `dictionary`: the entries page boxes' `closest-words` come from,
  relative to this folder (`../ocr/training_data/dictionary_words.json`).
  Rebuild it with `python -m ocr.training_data.list_dictionary_words`
  after either dictionary changes. Optional: without it, `closest-words`
  is `null`. Loading it takes about 13 MB a worker.
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

## Visitors

The server counts each day's unique visitors without keeping anything
about them (`server/visitors.py`), and prints the counts to the log:

```
visitors 2026-11-13: 5 (3 read something)
```

A line is printed whenever a count grows, so the day's latest is its
total so far. See it with:

```
heroku logs -a dots-and-circles | grep visitors | tail -1
```

Heroku keeps only about the last 1,500 lines of log, so this reaches back
a day or so at most.

Only requests from the frontend's page count (their `Origin` is
`allowed_origin`), so `curl`, scripts and local development don't; with
`allowed_origin` `"*"`, nothing is counted. "Read something" means a
successful `/random`, `/lexigraphs` or upload. A visitor is a hash of
their IP (the one Heroku's router saw) and browser (`User-Agent`) with a
random salt made each day. The salt and the day's hashes are kept only in
a file in the dyno's `/tmp`, shared by the workers, and deleted the next
day; no IP is written anywhere. Days are in UTC. A restart (Heroku's
daily one, or a deploy) wipes `/tmp`, so visitors after it count again
that day. People sharing an address and browser count as one, and one
person on two devices as two.


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
