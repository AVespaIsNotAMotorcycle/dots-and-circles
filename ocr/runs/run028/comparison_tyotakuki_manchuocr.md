# Photographed words: run 28 against tyotakuki/ManchuOCR

[tyotakuki/ManchuOCR](https://github.com/tyotakuki/ManchuOCR) (commit
7f1e910, June 2023) is a PyTorch CNN + BiLSTM encoder with an attention GRU
decoder, based on bai-shang/crnn_seq2seq_ocr_pytorch. It outputs Möllendorff
in capitals and ships two pretrained models: model 0 (trained on ~750k
images) and model 1 (~120k). The repository has no license, so this report
gives only aggregate results and a few short error examples, with
attribution. Its code, weights and per-photo guesses are not in this
repository.

Run 28 (D-8/64, Adam 3e-3, 10 epochs) uses its existing
`non_synthetic_eval_*.csv`. Both models were scored on the same 5,000
photographed words (seed 0), so every comparison is paired.

## Method

ManchuOCR reads whole pages. It segments columns and words by projection,
then runs word-level steps on each word. Since every photo is already one
word, only those word-level steps from `readmanchu.py` were applied, with its
default parameters:

1. Rotate 90° counter-clockwise.
2. Adaptive Gaussian inverse threshold (block 33, constant 32).
3. Crop to the ink by the blurred row and column sums (thresholds 800 and
   200).
4. Pad above and below so the spine is centred.
5. Scale to height 64 and pad as `readmanchu.py` does.
6. `ResizeNormalize(480, 64)`.

Decoding is greedy, as in `readmanchu.py`, batched 32 at a time. Each
pipeline was run on the raw photo and on the photo after our whitening +
cropping (`load_photo(path, crop=True, whiten=True)`). Spot checks of the
resulting inputs match the format of ManchuOCR's own examples.

Möllendorff was mapped to Abkai: lower case, š → x, ū → v, ž → z, with
hyphens dropped. Exact match and character accuracy are computed on Abkai
letters, the same way `ocr/evaluate_non_synthetic.py` computes them.

Their page-level segmentation might crop words slightly differently from step
3. Even so, a gap this large is unlikely to come from preprocessing alone.

## Headline

| Model | Preprocessing | Exact | Char |
|---|---|---|---|
| **Run 28** | whitened + cropped | **62.3%** | **90.5%** |
| Run 28 | plain | 49.3% | 85.3% |
| ManchuOCR model 1 | whitened + cropped, then theirs | 33.3% | 74.2% |
| ManchuOCR model 1 | theirs on the raw photo | 26.6% | 64.6% |
| ManchuOCR model 0 | whitened + cropped, then theirs | 29.4% | 70.7% |
| ManchuOCR model 0 | theirs on the raw photo | 26.1% | 64.8% |

Our whitening + cropping helps ManchuOCR too, by 3–7 points exact. Its best
setup, model 1 after our preprocessing, is the one compared below.

## Paired outcomes (whitened + cropped)

| | Photos |
|---|---|
| Exact in both | 1,147 |
| Exact only in run 28 | 1,970 |
| Exact only in ManchuOCR | 517 |
| Exact in neither | 1,366 |

Run 28 wins nearly 4 to 1 on the photos where the two disagree. The
either-model oracle is 72.7%, so the two models' errors only partly overlap.

## Word length (exact / char)

| Letters | Photos | Run 28 | ManchuOCR model 1 |
|---|---|---|---|
| 1–3 | 447 | 76.1% / 87.2% | 23.9% / 47.7% |
| 4–6 | 2,554 | 66.1% / 90.3% | 31.6% / 72.1% |
| 7–9 | 1,656 | 56.0% / 91.3% | 37.8% / 82.0% |
| 10+ | 343 | 47.5% / 92.1% | 35.9% / 86.6% |

The gap is widest on short words and narrowest on long ones, which are run
28's own weak spot.

## Words in and out of run 28's lexicon (exact / char)

| | Photos | Run 28 | ManchuOCR model 1 |
|---|---|---|---|
| in the 21,532-word lexicon | 2,669 | 65.8% / 90.6% | 33.4% / 72.5% |
| in no lexicon | 2,331 | 58.4% / 90.3% | 33.2% / 76.2% |

ManchuOCR is flat across the two because it was trained on a different word
list. Run 28 still leads by 25 points on words it has never seen, so the gap
isn't mainly about vocabulary.

## ManchuOCR's errors

- **u read as o** is the most common error: `obumbi → obombi`,
  `buki → boki`, `buci → boci`, `burakv → borakv`.
- **Insertions:** its guesses run 0.25 letters longer than the label on
  average (run 28: 0.00), a typical failure of attention decoders.
- **Short words:** `a → en`. Their 1–3 letter words are only 23.9% exact.

## Conclusions

1. Run 28 is clearly ahead: +29 points exact and +16 points char against
   ManchuOCR's best setup, in every length band and in and out of the
   lexicon.
2. On long words the difference is smaller (char 92.1% against 86.6%).
3. Our whitening + cropping isn't specific to our model. It helps a model
   trained independently on other data as well.
4. Showing ManchuOCR's guesses on the public site would need the author's
   permission, since the repository has no license.
