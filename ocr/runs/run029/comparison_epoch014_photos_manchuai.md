# Run 29 at epoch 14 against epoch 10 and run 28: all photos and the ManchuAI test sets

2026-11-03. Run 29 was resumed from epoch 10 and trained to epoch 14 at a
third of the learning rate: `get_learn_rate` returns `initial_rate / 3`,
which makes it 1e-3 instead of 3e-3. That change in `ocr/train.py` isn't
committed yet. The weight files and the log still say `3e-3`, because the
name comes from the run's configuration.

## Checkpoints

| Name | Model | Weights | Learning rate |
|---|---|---|---|
| Run 28 | D-8/64 | `run028/weights_D-8-64_Adam_3e-3_epoch010.npz` | 3e-3 throughout |
| Run 29 ep 10 | D-12/64, D-8/64, C-8/64 | `run029/weights_*_epoch010.npz` | 3e-3 |
| Run 29 ep 14 | D-12/64, D-8/64, C-8/64 | `run029/weights_*_Adam_3e-3.npz` (latest) | 3e-3 to epoch 10, then 1e-3 |

Epoch 14 is simply the latest checkpoint. It wasn't chosen by any photo
score, so the 500-photo set `train.py` monitors (seed 1) is included in the
photo results without bias.

## Test sets

- **All photos:** every one of the 24,280 photos of 2,428 words in
  `ocr/non_synthetic_data/`. Earlier reports used a 5,000-photo sample
  (seed 0). Run 29 has seen none of these words, because of the word split.
  Run 28 predates the split and saw some of them, which favours run 28.
- **Paper set (218 handwritten words):** `mic7ch/test_data_long`, train and
  validation splits.
- **Repository set (753 handwritten words):** the `test` split of
  `mic7ch/manchu`.

The method for the two handwritten sets is the same as in
`run028/comparison_manchuai_ocr.md`:

- Each image is rotated 90° clockwise back upright.
- The Manchu label is romanized with `ocr.photos.MANCHU_TO_ROMAN`, dropping
  unmapped punctuation.
- Both the label and the guess are normalized w → f before i/u/ū.

Run 28 reproduces its earlier figures exactly: 73.9% on the paper set and
60.4% on the repository set (whitened + cropped). The scoring is therefore
unchanged, even though Abkai has been replaced by the dataset romanization.
Neither dataset states a license, so this report gives aggregate results
only.

The headline pass is **whitened + cropped**, as in earlier reports. Exact is
the share of words read exactly; CER is total edits over total label
letters. Intervals are 95% paired bootstrap intervals (2,000 resamples) on
exact accuracy.

## Headline: whitened + cropped, exact (CER)

| Test set | Run 28 | D-12 ep 10 | **D-12 ep 14** | D-8 ep 10 | **D-8 ep 14** | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| All photos (24,280) | 62.5% (0.092) | 60.8% (0.101) | **71.7% (0.065)** | 61.6% (0.096) | **69.9% (0.072)** | 49.6% (0.139) | 59.5% (0.104) |
| Paper set (218) | 73.9% (0.062) | 76.6% (0.059) | **76.1% (0.065)** | 63.8% (0.088) | **78.4% (0.052)** | 52.3% (0.126) | 74.3% (0.062) |
| Repository set (753) | 60.4% (0.108) | 64.1% (0.098) | **70.3% (0.072)** | 59.1% (0.112) | **69.9% (0.081)** | 49.4% (0.150) | 62.3% (0.100) |

### Paired differences in exact accuracy (percentage points)

| Comparison | All photos | Paper set | Repository set |
|---|---|---|---|
| D-12 ep 14 − run 28 | **+9.1** [+8.5, +9.8] | +2.3 [−4.1, +8.7] | **+9.8** [+6.6, +13.3] |
| D-8 ep 14 − run 28 | **+7.4** [+6.8, +8.0] | +4.6 [−0.5, +10.1] | **+9.4** [+5.7, +12.7] |
| D-12 ep 14 − ep 10 | **+10.9** [+10.3, +11.5] | −0.5 [−6.0, +5.0] | **+6.1** [+3.1, +9.4] |
| D-8 ep 14 − ep 10 | **+8.4** [+7.8, +8.9] | **+14.7** [+9.2, +20.2] | **+10.8** [+7.4, +14.2] |
| C-8 ep 14 − ep 10 | **+9.9** [+9.3, +10.6] | **+22.0** [+15.1, +28.9] | **+12.9** [+9.2, +16.5] |
| D-12 ep 14 − D-8 ep 14 | **+1.7** [+1.2, +2.3] | −2.3 [−7.3, +2.8] | +0.4 [−2.7, +3.3] |

Bold marks intervals that exclude zero.

### Against the ManchuAI-OCR models

The other models' figures are from `run028/comparison_manchuai_ocr.md`.

| Model | Paper set (218) | Repository set (753) |
|---|---|---|
| CRNN v3 (trained on real handwriting, incl. our photos) | 97.7% | 97.3% |
| LLaMA-3.2-11B (60k synthetic, LoRA) | 93.1%* | 87.4% |
| **Run 29 D-12/64 ep 14** | 76.1% | **70.3%** |
| **Run 29 D-8/64 ep 14** | **78.4%** | 69.9% |
| Run 28 | 73.9% | 60.4% |
| CRNN-3M (60k synthetic) | 72.5%* | 56.8% |

\* Checkpoint chosen on this test set, so the figure is optimistic.

On the repository set, run 29 is now about 13–14 points ahead of CRNN-3M,
up from run 28's 3.6. The gap to LLaMA-3.2-11B has shrunk from 27 to 17
points.

## All passes, exact (CER)

### All photos (24,280)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 62.5% (0.092) | 60.8% (0.101) | 71.7% (0.065) | 61.6% (0.096) | 69.9% (0.072) | 49.6% (0.139) | 59.5% (0.104) |
| whitened | 59.4% (0.104) | 59.9% (0.105) | 69.3% (0.072) | 59.5% (0.103) | 67.9% (0.077) | 46.6% (0.150) | 57.8% (0.111) |
| cropped | 55.5% (0.120) | 52.6% (0.135) | 66.6% (0.081) | 50.3% (0.137) | 61.6% (0.100) | 41.0% (0.176) | 52.7% (0.132) |
| plain | 49.5% (0.144) | 48.4% (0.154) | 63.3% (0.095) | 46.5% (0.158) | 58.7% (0.111) | 38.5% (0.187) | 52.0% (0.134) |

Epoch 14 gains on every pass. D-12/64's plain score rose 15 points, so the
lower learning rate also made it less dependent on preprocessing.

### Paper set (218)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 73.9% (0.062) | 76.6% (0.059) | 76.1% (0.065) | 63.8% (0.088) | 78.4% (0.052) | 52.3% (0.126) | 74.3% (0.062) |
| cropped | 76.6% (0.057) | 76.6% (0.059) | 77.1% (0.062) | 63.8% (0.087) | 77.1% (0.055) | 53.2% (0.123) | 74.3% (0.062) |
| whitened | 46.8% (0.185) | 9.6% (0.276) | 5.0% (0.293) | 42.2% (0.199) | 15.6% (0.258) | 3.2% (0.449) | 15.6% (0.258) |
| plain | 46.8% (0.190) | 9.6% (0.279) | 5.5% (0.291) | 44.0% (0.195) | 19.3% (0.247) | 3.2% (0.449) | 15.1% (0.258) |

### Repository set (753)

| Pass | Run 28 | D-12 ep 10 | D-12 ep 14 | D-8 ep 10 | D-8 ep 14 | C-8 ep 10 | C-8 ep 14 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 60.4% (0.108) | 64.1% (0.098) | 70.3% (0.072) | 59.1% (0.112) | 69.9% (0.081) | 49.4% (0.150) | 62.3% (0.100) |
| cropped | 60.4% (0.108) | 64.1% (0.098) | 70.3% (0.072) | 59.1% (0.112) | 69.9% (0.081) | 49.4% (0.150) | 62.3% (0.100) |
| whitened | 38.0% (0.209) | 6.9% (0.315) | 0.9% (0.301) | 36.4% (0.201) | 21.6% (0.261) | 2.7% (0.518) | 10.6% (0.333) |
| plain | 38.0% (0.209) | 6.9% (0.315) | 0.9% (0.301) | 36.4% (0.201) | 21.6% (0.261) | 2.7% (0.518) | 10.6% (0.333) |

## Uncropped handwriting: a trailing space

On the uncropped handwritten images, run 29 collapses: D-12/64 at epoch 14
gets 0.9% plain on the repository set, against run 28's 38.0%. These images
are padded to 480 × 64 with the word filling only part of the frame. Run 29
reads the empty padding after the word as a word gap and ends its guess
with a space:

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
   points on all photos for every model, far more than the epoch-to-epoch
   noise at 3e-3 (±2–4 points). Run 29 D-12/64 at epoch 14 is the new best
   on photos: 71.7% exact, CER 0.065.
2. **Run 29 now beats run 28 clearly**, by +9.1 (D-12) and +7.4 (D-8)
   points on all photos. That's despite run 28 having seen some photo words
   in training. It also leads by about +10 on the 753-word handwritten
   set, a model that has never seen handwriting.
3. **D-12 against D-8:** D-12 is ahead on photos (+1.7, interval excludes
   zero). On both handwritten sets they can't be told apart.
4. **C-8/64 is still about 10 points behind the D models** on photos.
   Dropout continues to help.
5. **The paper set is too small to separate the models.** At n = 218 the
   interval is about ±6 points. D-12/64 gained nothing there from epochs
   11–14, while it gained 6 points on the 753-word set.
6. **Always crop.** Uncropped, padded handwriting makes run 29 append a
   space to almost every guess (see above).

Still to do, if wanted: continue at 1e-3 (or lower) to see whether the
photo score keeps rising, and run the per-letter error breakdown for epoch
14.

Per-image results and the scripts are in the session scratchpad only.
