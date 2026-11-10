# CLAUDE.md

Manchu OCR built on a hand-written numpy neural network library. See
README.md for the architecture.

## Layout

- `nn/`, `loss/`, `optim/`, `utils/`: the numpy library (layers, CTCLoss,
  Adam/SGD, softmax), each with its own `tests/`.
- `ocr/`: models (`modelA`–`modelD`), `train.py`, `evaluate_non_synthetic.py`,
  `photos.py`, `preprocess.py`. Reading whole pages: `layout.py` (columns
  and words, classical), `box_classifier.py` (Manchu or other, a small MLP;
  `train_box_classifier.py`), `page_reader.py`, `evaluate_pages.py`
  (scores them on synthetic pages) and `evaluate_real_pages.py` (on the
  transcribed real pages in `ocr/real_pages/`; see its README for sources).
- `ocr/training_data/`: lexicon, word rendering (`__init__.py`) and print
  augmentation (`augment.py`). `fonts/` is gitignored for licensing reasons:
  never commit font files, and check it holds `.ttf` files before training.
  `pages.py` renders synthetic pages (Manchu columns, sometimes with Chinese)
  with their true boxes, for the page reader; its Chinese fonts go in
  `fonts/chinese/`.
- `ocr/runs/runNNN/`: one folder per training run (log, graphs, reports).
  Weights (`*.npz`) are gitignored.
- `server/`: the Flask server Heroku runs (`Procfile`, `requirements.txt`
  at the root). Its models, weights and photos are listed in
  `server/config.json`; see `server/README.md`.

## Commands

Run from the repo root:

- Train: `python -m ocr.train`
- Evaluate on photographed words:
  `python -m ocr.evaluate_non_synthetic ocr/runs/runNNN/<weights>.npz`
- Test: `python -m pytest -q nn loss optim utils ocr/tests server/tests`
  (the server tests skip without Flask)
- Serve locally: `python -m flask --app server.app run`

## Lexicon

`lexicon.json` is built in two steps, run from `ocr/training_data/`:

1. `reformat.py` rebuilds it from scratch from the Wiktionary dump
   (`kaikki.org-dictionary-Manchu.jsonl`).
2. `add_norman.py` then adds the headwords of Norman's dictionary
   (`Jerry_Norman_Dict.txt`). Running `reformat.py` alone drops them.

Changing the lexicon changes which word each training seed maps to, so runs
before and after a lexicon change aren't directly comparable.

## Photographed data

`ocr/non_synthetic_data/` holds photographed print words, one folder per
word. It's gitignored, so check it's there before relying on it.

The photos are split by word (`ocr.split.split_photo_words`) into three
sets, listed in the committed `ocr/training_data/photo_words.json`:

- **train** (85% of photos): added to every epoch's training batches,
  prepared as for testing and not augmented.
- **monitor** (500 photos): scored after each epoch and logged as
  `photo`/`photo_exact`. If a checkpoint is ever chosen by that score, the
  benchmark is still clean.
- **benchmark** (the rest, ~3,150 photos): the held-out test.
  `evaluate_non_synthetic.py` uses it by default (`--set` picks another).

The folders are named in the dataset's romanization: Möllendorff in ASCII
with x and v, which isn't Abkai (it writes ᠴ as c). `ocr.photos.manchu_to_roman`
gives it, and all scoring and label matching uses it. `manchu_to_abkai` is
real Abkai, only for showing words to people.

Train and test are split by word (`ocr/split.py`): a word is in one set in
every font and as a photo. Lexicon words matching training photo words go
to train, and those matching monitor or benchmark words go to test.
`photo_words.json` is CC BY 4.0. Rebuild it with
`python -m ocr.training_data.list_photo_words` when the photos change, then
rebuild the demo sets (`server/select_photos.py`,
`server/select_synthetic.py`). Runs from run 30 on train on photos and on a
different lexicon split. Earlier runs never saw any photo word, so they can
be scored on the benchmark, but their synthetic test scores aren't
comparable.

## Memory use

The models keep their forward activations for the backward pass, so memory
grows with batch size × number of models or worker processes. Size parallel
jobs with that in mind.

## Standing decisions

- The model must read words that aren't in any lexicon (Chinese and Mongolian
  transliterations, unknown words). Don't use lexicon-constrained decoding.
  A "nearest match" shown beside the raw reading in a frontend is fine.
- Whether free variation selectors (FVS) stay in training labels is
  undecided. Don't assume either way.
