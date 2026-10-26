# CLAUDE.md

Manchu OCR built on a hand-written numpy neural network library. See
README.md for the architecture.

## Layout

- `nn/`, `loss/`, `optim/`, `utils/`: the numpy library (layers, CTCLoss,
  Adam/SGD, softmax), each with its own `tests/`.
- `ocr/`: models (`modelA`–`modelD`), `train.py`, `evaluate_non_synthetic.py`,
  `photos.py`, `preprocess.py`.
- `ocr/training_data/`: lexicon, fonts, word rendering (`__init__.py`) and
  print augmentation (`augment.py`).
- `ocr/runs/runNNN/`: one folder per training run (log, graphs, reports).
  Weights (`*.npz`) are gitignored.

## Commands

Run from the repo root:

- Train: `python -m ocr.train`
- Evaluate on photographed words:
  `python -m ocr.evaluate_non_synthetic ocr/runs/runNNN/<weights>.npz`
- Test: `python -m pytest -q nn loss optim utils ocr/tests`

## Lexicon

`lexicon.json` is built in two steps, run from `ocr/training_data/`:

1. `reformat.py` rebuilds it from scratch from the Wiktionary dump
   (`kaikki.org-dictionary-Manchu.jsonl`).
2. `add_norman.py` then adds the headwords of Norman's dictionary
   (`Jerry_Norman_Dict.txt`). Running `reformat.py` alone drops them.

Changing the lexicon changes which word each training seed maps to, so runs
before and after a lexicon change aren't directly comparable.

## Photographed test data

`ocr/non_synthetic_data/` holds photographed print words, one folder per
word, named in Abkai. It's gitignored, so check it's there before relying on
it. `train.py` scores a fixed 500-photo set (seed 1) after each epoch. If a
checkpoint is ever chosen by that score, evaluate on photos outside that set.

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
