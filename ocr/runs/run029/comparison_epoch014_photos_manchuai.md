# Run 29 at epoch 14 against epoch 10, run 28 and ManchuOCR: all photos and the ManchuAI test sets

2026-11-03. Run 29 was resumed from epoch 10 and trained to epoch 14 at a
third of the learning rate: `get_learn_rate` returns `initial_rate / 3`
from the 11th epoch on, which makes it 1e-3 instead of 3e-3. The weight
files and the log still say `3e-3`, because the name comes from the run's
configuration.

## Checkpoints

| Name | Model | Weights | Learning rate |
|---|---|---|---|
| Run 28 | D-8/64 | `run028/weights_D-8-64_Adam_3e-3_epoch010.npz` | 3e-3 throughout |
| Run 29 ep 10 | D-12/64, D-8/64, C-8/64 | `run029/weights_*_epoch010.npz` | 3e-3 |
| Run 29 ep 14 | D-12/64, D-8/64, C-8/64 | `run029/weights_*_Adam_3e-3.npz` (latest) | 3e-3 to epoch 10, then 1e-3 |
| ManchuOCR model 1 | CNN + BiLSTM encoder, attention GRU decoder | [tyotakuki/ManchuOCR](https://github.com/tyotakuki/ManchuOCR) `model/*_1.pth` | — |
| ManchuOCR model 0 | the same | `model/*_0.pth` | — |

Epoch 14 is simply the latest checkpoint. It wasn't chosen by any photo
score, so the 500-photo set `train.py` monitors (seed 1) is included in the
photo results without bias.

ManchuOCR is a reference: an independent model trained on 750k (model 0)
and 120k (model 1) synthetic images of words from Zhang Zhuohui's
130,917-word list. Its weights date from April 2022 (last commit 7f1e910,
June 2023), years before the photo dataset was published, so it's very
unlikely to have seen our photos. The repository has no
license, so this report gives only aggregate results. Its code, weights and
guesses are not in this repository.

## Test sets

- **All photos:** every one of the 24,280 photos of 2,428 words in
  `ocr/non_synthetic_data/`. Earlier reports used a 5,000-photo sample
  (seed 0). Run 29 has seen none of these words, because of the word split.
  Run 28 predates the split and saw some of them, which favours run 28.
- **Paper set (218 handwritten words):** `mic7ch/test_data_long`, train and
  validation splits.
- **Repository set (753 handwritten words):** the `test` split of
  `mic7ch/manchu`.

The two handwritten sets are handled as in
`run028/comparison_manchuai_ocr.md`:

- Each image is rotated 90° clockwise back upright.
- The Manchu label is romanized with `ocr.photos.MANCHU_TO_ROMAN`, dropping
  unmapped punctuation.
- Both the label and the guess are normalized w → f before i/u/ū.

Run 28 reproduces its earlier figures exactly: 73.9% exact on the paper set
and 60.4% on the repository set, and 90.4% char on all photos against
90.5% on the 5,000-photo sample. Neither dataset states a license, so this
report gives aggregate results only.

## Metrics

- **Exact:** the share of words read exactly.
- **Char:** mean character accuracy, `max(0, 1 − edit distance / label
  length)` per word, as in `ocr/evaluate_non_synthetic.py`.
- **CER:** total edits over total label letters.

Intervals are 95% paired bootstrap intervals (2,000 resamples). Our models'
headline pass is **whitened + cropped**, as in earlier reports.

## How ManchuOCR was run

ManchuOCR reads whole pages. It cuts each page into lines and words, then
reads each word. Since every image here is one word, only the word-level
steps of `readmanchu.py` were applied, with its default parameters:

1. Rotate 90° counter-clockwise.
2. Adaptive Gaussian inverse threshold (block 33, constant 32), with a
   10-pixel border.
3. Crop along the word by the blurred column sums (threshold 200).
4. Pad across the word so the spine is centred.
5. Scale to height 64, pad as `readmanchu.py` does, then
   `ResizeNormalize(480, 64)`.

Decoding is greedy, as in `readmanchu.py`, batched 32 at a time. Its
Möllendorff output is mapped to the dataset romanization (lower case,
š → x, ū → v, ž → z, hyphens dropped). Each image goes through two
pipelines: **its own** steps only, and **ours first**, meaning our
whitening + cropping (`prepare(crop=True, whiten=True)`) before its steps.

**This differs from the October comparison in one step.** `readmanchu.py`
also crops across the word, by the line's blurred row sums (threshold 800).
On a page, those sums run over the whole line, so each word keeps the
line's full width. On a single word, they're the word's own and cut off the
outermost rows, clipping the tips of letters. October's comparison
(`run028/comparison_tyotakuki_manchuocr.md`) applied that crop and got
33.3% exact. Skipping it, as here, is closer to what ManchuOCR does on a
page:

| ManchuOCR model 1, 500 photos (seed-0 sample), ours first | Exact | Char |
|---|---|---|
| Crop across and along the word (as in October) | 25.2% | 67.5% |
| **Crop only along the word (this report)** | **70.4%** | **91.5%** |
| No crop | 62.2% | 87.5% |
| Crop only across the word | 19.6% | 62.7% |

The variant was picked on these 500 photos, which are part of the
all-photos test set. The choice follows from how the page reader works,
though, and the gap between variants (45 points) is far larger than any
selection effect. The handwritten sets weren't used to choose it.

The October numbers couldn't be reproduced exactly. With both crops, the
rebuilt pipeline gets 29.1% / 70.6% on the 5,000-photo sample, against
October's 33.3% / 74.2%. Its scripts weren't kept, so the remaining
difference can't be traced.

## Headline: exact / char

Our models are whitened + cropped. ManchuOCR is model 1 with our
preprocessing first, its best setup on photos.

| Test set | Run 28 | D-12 ep 10 | **D-12 ep 14** | D-8 ep 10 | **D-8 ep 14** | C-8 ep 10 | C-8 ep 14 | ManchuOCR |
|---|---|---|---|---|---|---|---|---|
| All photos (24,280) | 62.5% / 90.4% | 60.8% / 89.3% | **71.7% / 93.2%** | 61.6% / 89.7% | **69.9% / 92.4%** | 49.6% / 85.5% | 59.5% / 89.3% | 68.4% / 90.9% |
| Paper set (218) | 73.9% / 94.1% | 76.6% / 94.3% | **76.1% / 94.4%** | 63.8% / 91.1% | **78.4% / 95.1%** | 52.3% / 86.8% | 74.3% / 93.8% | 52.3% / 81.8% |
| Repository set (753) | 60.4% / 88.7% | 64.1% / 89.4% | **70.3% / 92.3%** | 59.1% / 88.1% | **69.9% / 91.4%** | 49.4% / 84.1% | 62.3% / 89.4% | 56.0% / 84.7% |

### CER

| Test set | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 | ManchuOCR |
|---|---|---|---|---|---|---|---|---|
| All photos | 0.092 | 0.101 | 0.065 | 0.096 | 0.072 | 0.139 | 0.104 | 0.082 |
| Paper set | 0.062 | 0.059 | 0.065 | 0.088 | 0.052 | 0.126 | 0.062 | 0.180 |
| Repository set | 0.108 | 0.098 | 0.072 | 0.112 | 0.081 | 0.150 | 0.100 | 0.154 |

### Paired differences (percentage points, exact / char)

| Comparison | All photos | Paper set | Repository set |
|---|---|---|---|
| D-12 ep 14 − run 28, exact | **+9.1** [+8.5, +9.8] | +2.3 [−4.1, +8.7] | **+9.8** [+6.6, +13.3] |
| D-12 ep 14 − run 28, char | **+2.7** [+2.6, +2.9] | +0.3 [−1.3, +1.9] | **+3.6** [+2.4, +4.6] |
| D-8 ep 14 − run 28, exact | **+7.4** [+6.8, +8.0] | +4.6 [−0.5, +10.1] | **+9.4** [+5.7, +12.7] |
| D-8 ep 14 − run 28, char | **+2.0** [+1.9, +2.2] | +1.0 [−0.3, +2.3] | **+2.7** [+1.6, +3.9] |
| D-12 ep 14 − ep 10, exact | **+10.9** [+10.3, +11.5] | −0.5 [−6.0, +5.0] | **+6.1** [+3.1, +9.4] |
| D-12 ep 14 − ep 10, char | **+3.9** [+3.7, +4.0] | +0.1 [−1.4, +1.7] | **+2.8** [+1.8, +3.8] |
| D-8 ep 14 − ep 10, exact | **+8.4** [+7.8, +8.9] | **+14.7** [+9.2, +20.2] | **+10.8** [+7.4, +14.2] |
| D-8 ep 14 − ep 10, char | **+2.7** [+2.5, +2.9] | **+4.1** [+2.4, +5.9] | **+3.3** [+2.2, +4.4] |
| C-8 ep 14 − ep 10, exact | **+9.9** [+9.3, +10.6] | **+22.0** [+15.1, +28.9] | **+12.9** [+9.2, +16.5] |
| C-8 ep 14 − ep 10, char | **+3.9** [+3.6, +4.1] | **+7.0** [+4.7, +9.4] | **+5.3** [+3.9, +6.5] |
| D-12 ep 14 − D-8 ep 14, exact | **+1.7** [+1.2, +2.3] | −2.3 [−7.3, +2.8] | +0.4 [−2.7, +3.3] |
| D-12 ep 14 − D-8 ep 14, char | **+0.7** [+0.5, +0.9] | −0.7 [−2.1, +0.7] | +0.9 [−0.1, +1.8] |
| D-12 ep 14 − ManchuOCR, exact | **+3.3** [+2.5, +3.9] | **+23.9** [+15.6, +31.7] | **+14.2** [+9.7, +18.7] |
| D-12 ep 14 − ManchuOCR, char | **+2.3** [+2.0, +2.5] | **+12.6** [+9.4, +16.1] | **+7.6** [+5.9, +9.4] |
| D-8 ep 14 − ManchuOCR, exact | **+1.5** [+0.8, +2.3] | **+26.1** [+18.3, +34.4] | **+13.8** [+9.3, +18.5] |
| D-8 ep 14 − ManchuOCR, char | **+1.6** [+1.3, +1.8] | **+13.3** [+10.0, +16.7] | **+6.7** [+5.0, +8.6] |
| Run 28 − ManchuOCR, exact | **−5.9** [−6.6, −5.1] | **+21.6** [+13.3, +29.8] | +4.4 [−0.4, +9.0] |
| Run 28 − ManchuOCR, char | **−0.4** [−0.7, −0.2] | **+12.3** [+9.2, +15.7] | **+4.0** [+2.1, +6.0] |

Bold marks intervals that exclude zero. "ManchuOCR" is model 1, ours
first.

On all photos the two models' errors overlap only in part. D-12/64 at
epoch 14 reads 4,804 photos that ManchuOCR misses, and ManchuOCR 4,014 that
D-12 misses.

### Against the ManchuAI-OCR models

The other models' figures are from `run028/comparison_manchuai_ocr.md`
(exact only).

| Model | Paper set (218) | Repository set (753) |
|---|---|---|
| CRNN v3 (trained on real handwriting, incl. our photos) | 97.7% | 97.3% |
| LLaMA-3.2-11B (60k synthetic, LoRA) | 93.1%* | 87.4% |
| **Run 29 D-12/64 ep 14** | 76.1% | **70.3%** |
| **Run 29 D-8/64 ep 14** | **78.4%** | 69.9% |
| Run 28 | 73.9% | 60.4% |
| ManchuOCR model 1, its own steps | 60.6% | 58.0% |
| CRNN-3M (60k synthetic) | 72.5%* | 56.8% |
| ManchuOCR model 1, ours first | 52.3% | 56.0% |

\* Checkpoint chosen on this test set, so the figure is optimistic.

On the repository set, run 29 is now about 13–14 points ahead of CRNN-3M,
up from run 28's 3.6. The gap to LLaMA-3.2-11B has shrunk from 27 to 17
points.

## ManchuOCR, all setups (exact / char)

| Test set | Model 1, ours first | Model 1, its own steps | Model 0, ours first | Model 0, its own steps |
|---|---|---|---|---|
| All photos (24,280) | **68.4% / 90.9%** | 51.4% / 81.1% | 66.6% / 90.6%† | 57.8% / 84.9%† |
| Paper set (218) | 52.3% / 81.8% | **60.6% / 86.9%** | 45.0% / 80.0% | 50.5% / 84.5% |
| Repository set (753) | 56.0% / 84.7% | **58.0% / 86.5%** | 51.4% / 82.9% | 54.6% / 86.1% |

† Model 0 was run on 500 photos only (the seed-0 sample's first 500), where
model 1 gets 70.4% / 91.5% ours first and 54.8% / 82.7% with its own
steps. On those 500 the two models are within noise of each other, so
model 0 wasn't run on all photos.

- **Model 1 is the better model** in every setup but one.
- **Our preprocessing helps it on print by 17 points** (68.4% against
  51.4%), but hurts it on handwriting by 2–8 points. The handwritten images
  are already binarized and tightly padded, and its own thresholding suits
  them better.
- **Run 29 leads either way.** Against ManchuOCR's best setup on each
  handwritten set, which is test-set selection in ManchuOCR's favour,
  D-12/64 at epoch 14 is still +15.5 points exact on the paper set and
  +12.3 on the repository set.

## Our models, all passes (exact / char)

### All photos (24,280)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 62.5% / 90.4% | 60.8% / 89.3% | 71.7% / 93.2% | 61.6% / 89.7% | 69.9% / 92.4% | 49.6% / 85.5% | 59.5% / 89.3% |
| whitened | 59.4% / 89.4% | 59.9% / 89.3% | 69.3% / 92.6% | 59.5% / 89.6% | 67.9% / 92.2% | 46.6% / 84.6% | 57.8% / 88.6% |
| cropped | 55.5% / 87.6% | 52.6% / 85.9% | 66.6% / 91.4% | 50.3% / 85.6% | 61.6% / 89.7% | 41.0% / 81.6% | 52.7% / 86.3% |
| plain | 49.5% / 85.4% | 48.4% / 84.5% | 63.3% / 90.4% | 46.5% / 84.3% | 58.7% / 89.0% | 38.5% / 81.0% | 52.0% / 86.3% |

Epoch 14 gains on every pass. D-12/64's plain score rose 15 points exact,
so the lower learning rate also made it less dependent on preprocessing.

### Paper set (218)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 73.9% / 94.1% | 76.6% / 94.3% | 76.1% / 94.4% | 63.8% / 91.1% | 78.4% / 95.1% | 52.3% / 86.8% | 74.3% / 93.8% |
| cropped | 76.6% / 94.7% | 76.6% / 94.3% | 77.1% / 94.7% | 63.8% / 91.3% | 77.1% / 94.9% | 53.2% / 87.1% | 74.3% / 93.7% |
| whitened | 46.8% / 83.1% | 9.6% / 72.3% | 5.0% / 70.8% | 42.2% / 81.1% | 15.6% / 75.2% | 3.2% / 55.0% | 15.6% / 74.4% |
| plain | 46.8% / 82.8% | 9.6% / 72.0% | 5.5% / 71.1% | 44.0% / 81.5% | 19.3% / 76.5% | 3.2% / 54.8% | 15.1% / 74.5% |

### Repository set (753)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 60.4% / 88.7% | 64.1% / 89.4% | 70.3% / 92.3% | 59.1% / 88.1% | 69.9% / 91.4% | 49.4% / 84.1% | 62.3% / 89.4% |
| cropped | 60.4% / 88.7% | 64.1% / 89.4% | 70.3% / 92.3% | 59.1% / 88.1% | 69.9% / 91.4% | 49.4% / 84.1% | 62.3% / 89.4% |
| whitened | 38.0% / 79.6% | 6.9% / 66.7% | 0.9% / 67.6% | 36.4% / 80.3% | 21.6% / 74.8% | 2.7% / 46.2% | 10.6% / 65.5% |
| plain | 38.0% / 79.6% | 6.9% / 66.7% | 0.9% / 67.6% | 36.4% / 80.3% | 21.6% / 74.8% | 2.7% / 46.2% | 10.6% / 65.5% |

## Uncropped handwriting: a trailing space

On the uncropped handwritten images, run 29 collapses in exact accuracy:
D-12/64 at epoch 14 gets 0.9% plain on the repository set, against run
28's 38.0%. Its character accuracy stays at 67.6%, so most letters are
still right. These images are padded to 480 × 64 with the word filling only
part of the frame, and run 29 reads the empty padding after the word as a
word gap, ending its guess with a space:

| Repository set, plain | Guesses with a leading or trailing space | Exact as scored | Exact with the space stripped |
|---|---|---|---|
| Run 28 | 7.7% | 38.0% | 40.8% |
| D-8 ep 14 | 64.4% | 21.6% | 54.4% |
| D-12 ep 10 | 86.7% | 6.9% | 49.5% |
| D-12 ep 14 | 97.9% | 0.9% | 58.2% |

The photos are tightly framed, so this never happens on them: 0.0% of
guesses have the extra space. With cropping, which the server always does,
it all but disappears on handwriting too (≤ 0.1% at epoch 14).

The cause isn't established. A plausible guess: in training, a long blank
stretch below the ink only occurs between the words of a phrase (38% of the
lexicon), since renders are padded above the word, never below. Run 28 had
the same data, so why run 29 does it so much more isn't known. It gets
worse with training (D-12 86.7% → 97.9% from epoch 10 to 14). Two
possible fixes are stripping the guess in the frontend and scoring, or
padding below some single-word renders in training. Neither is needed for
cropped input.

## Conclusions

1. **The lower learning rate paid off.** Epochs 11–14 at 1e-3 added 8–11
   points exact and 3–4 points char on all photos for every model. That's
   far more than the epoch-to-epoch noise at 3e-3 (±2–4 points exact). Run
   29 D-12/64 at epoch 14 is the new best on photos: 71.7% exact, 93.2%
   char, CER 0.065.
2. **Run 29 now beats run 28 clearly.** On all photos it leads by +9.1
   exact / +2.7 char (D-12) and +7.4 / +2.0 (D-8), even though run 28 saw
   some photo words in training. It leads by about +10 exact / +3 char on
   the 753-word handwritten set, a model that has never seen handwriting.
3. **ManchuOCR is a much stronger reference than October's report
   suggested.** With the crop corrected, model 1 gets 68.4% exact / 90.9%
   char on all photos, ahead of run 28 (−5.9 exact for run 28) and only 3.3
   / 2.3 points behind run 29 D-12/64. On handwriting run 29 is far ahead,
   by about 12–26 points exact in every setup.
4. **D-12 against D-8:** D-12 is ahead on photos (+1.7 exact / +0.7 char,
   intervals exclude zero). On both handwritten sets they can't be told
   apart.
5. **C-8/64 is still about 10 points behind the D models** on photos
   (3 points char). Dropout continues to help.
6. **The paper set is too small to separate the models.** At n = 218 the
   interval is about ±6 points exact.
7. **Always crop.** Uncropped, padded handwriting makes run 29 append a
   space to almost every guess (see above).

Still to do, if wanted: continue at 1e-3 (or lower) to see whether the
photo score keeps rising, and run the per-letter error breakdown for epoch
14.

Per-image results and the scripts are in the session scratchpad only.
