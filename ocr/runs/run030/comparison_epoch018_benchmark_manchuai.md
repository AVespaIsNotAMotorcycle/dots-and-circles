# Run 30 at epoch 18 against runs 28 and 29, ManchuOCR and ManchuAI: photo benchmark and the ManchuAI test sets

2026-11-04. Run 30 is run 29 resumed from its epoch-14 weights and Adam
state. It trained epochs 15–18 at 1e-3 (`initial_rate / 3`; the filenames
still say `3e-3`), with the 20,630 training photos added to every epoch
(see CLAUDE.md, "Photographed data"). Runs 28 and 29 never trained on a
photo.

## Checkpoints

| Name | Model | Weights |
|---|---|---|
| **Run 30 ep 18** | D-12/64, D-8/64 | `run030/weights_*_Adam_3e-3.npz` (latest) |
| Run 30 ep 15 | D-12/64, D-8/64 | `run030/weights_*_epoch015.npz` |
| Run 29 ep 14 | D-12/64, D-8/64 | `run029/weights_*_Adam_3e-3.npz` |
| Run 28 | D-8/64 | `run028/weights_D-8-64_Adam_3e-3_epoch010.npz` |

Epoch 18 is simply the latest checkpoint. No checkpoint was chosen by a
benchmark or handwriting score.

## Test sets

- **Photo benchmark (3,150 photos of 315 words):** the `benchmark` set of
  `photo_words.json`, the evaluator's default. Run 30 trained on none of
  these words, as photos or as renders. Runs 28 and 29 trained on no photo at
  all. Run 28 predates the word split, so it rendered some benchmark words.
  The benchmark comes from the same books and the same photography as the
  training photos, though, so it's held out by word but not by source.
- **Paper set (218 handwritten words):** `mic7ch/test_data_long`, train and
  validation splits.
- **Repository set (753 handwritten words):** the `test` split of
  `mic7ch/manchu`. Our photos are the real part of that dataset's training
  splits, but the test split is separate handwriting with no byte-identical
  images in train (`run028/comparison_manchuai_ocr.md`). So run 30 has
  still never seen handwriting.

Handwritten images are handled as in the run 28 and 29 reports: rotated
90° clockwise, labels romanized with `ocr.photos.MANCHU_TO_ROMAN`, and w → f
before i/u/ū in both label and guess. Every earlier figure reproduces
exactly: run 29 D-12 ep 14 gets 72.1% / 93.3% on the benchmark, run 28 gets
73.9% and 60.4% on the handwritten sets, and the CRNN-3M pairing below
gives the same McNemar z as before (+1.73).

Metrics are as before. **Exact** is words read exactly. **Char** is mean
`max(0, 1 − edits / label length)`. **CER** is total edits over total
label letters. Intervals are 95% paired bootstrap intervals (2,000
resamples). The headline pass is **whitened + cropped**.

## Headline: exact / char

| Test set | **Run 30 D-12 ep 18** | **Run 30 D-8 ep 18** | Run 29 D-12 ep 14 | Run 29 D-8 ep 14 | Run 28 | ManchuOCR model 1 |
|---|---|---|---|---|---|---|
| Photo benchmark (3,150) | **92.9% / 98.5%** | 91.9% / 98.3% | 72.1% / 93.3% | 69.6% / 92.5% | 62.4% / 90.4% | 69.2% / 91.2% (52.1% / 81.6%) |
| Paper set (218) | **90.4% / 98.1%** | 89.0% / 97.4% | 76.1% / 94.4% | 78.4% / 95.1% | 73.9% / 94.1% | 52.3% / 81.7% (60.6% / 86.9%) |
| Repository set (753) | **85.5% / 96.8%** | 81.1% / 95.5% | 70.3% / 92.3% | 69.9% / 91.4% | 60.4% / 88.7% | 55.9% / 84.9% (58.0% / 86.5%) |

ManchuOCR is model 1, run as in `run029/comparison_epoch014_photos_manchuai.md`
(its word-level steps, cropping only along the word). The main figure has
our whitening + cropping applied first; its own preprocessing alone is in
brackets. The handwritten figures reproduce the run 29 report (55.9%
against 56.0% on the repository set, one image; the rest exactly).

### CER

| Test set | Run 30 D-12 ep 18 | Run 30 D-8 ep 18 | Run 29 D-12 ep 14 | Run 29 D-8 ep 14 | Run 28 | ManchuOCR (ours first / own) |
|---|---|---|---|---|---|---|
| Photo benchmark | 0.014 | 0.016 | 0.065 | 0.074 | 0.094 | 0.081 / 0.178 |
| Paper set | 0.023 | 0.024 | 0.065 | 0.052 | 0.062 | 0.181 / 0.133 |
| Repository set | 0.034 | 0.044 | 0.072 | 0.081 | 0.108 | 0.151 / 0.138 |

### Against ManchuOCR, paired (percentage points, exact / char)

ManchuOCR is compared in its best setup on each set: our preprocessing
first on photos, its own steps on handwriting. Choosing that per set
favours ManchuOCR.

| Comparison | Photo benchmark | Paper set | Repository set |
|---|---|---|---|
| Run 30 D-12 − ManchuOCR, exact | **+23.7** [+21.9, +25.4] | **+29.8** [+22.5, +37.6] | **+27.5** [+23.4, +31.5] |
| Run 30 D-12 − ManchuOCR, char | **+7.3** [+6.7, +7.8] | **+11.2** [+8.7, +13.8] | **+10.3** [+8.8, +11.8] |
| Run 30 D-8 − ManchuOCR, exact | **+22.8** [+21.0, +24.6] | **+28.4** [+21.6, +35.3] | **+23.1** [+19.1, +26.8] |
| Run 29 D-12 − ManchuOCR, exact | **+2.9** [+0.8, +5.0] | **+15.6** [+7.3, +23.4] | **+12.2** [+7.6, +16.5] |
| Run 29 D-12 − ManchuOCR, char | **+2.1** [+1.4, +2.8] | **+7.5** [+4.8, +10.4] | **+5.8** [+4.0, +7.4] |
| Run 28 − ManchuOCR, exact | **−6.8** [−9.0, −4.7] | **+13.3** [+5.0, +21.6] | +2.4 [−2.4, +7.0] |
| Run 28 − ManchuOCR, char | **−0.9** [−1.6, −0.1] | **+7.2** [+4.4, +10.0] | **+2.2** [+0.3, +4.0] |

On the benchmark, run 30 D-12 reads 838 photos that ManchuOCR misses, and
ManchuOCR reads 92 that run 30 misses. For run 29 it was 606 against 515,
so the two models' errors used to overlap only partly. Run 30 now covers
nearly everything ManchuOCR gets right.

Run 30 cuts CER by roughly 4–5× on photos and about 2–3× on handwriting.

### Paired differences (percentage points)

| Comparison | Photo benchmark | Paper set | Repository set |
|---|---|---|---|
| Run 30 D-12 − run 29 D-12, exact | **+20.8** [+19.3, +22.3] | **+14.2** [+9.2, +19.7] | **+15.3** [+12.1, +18.6] |
| Run 30 D-12 − run 29 D-12, char | **+5.2** [+4.8, +5.6] | **+3.6** [+2.2, +5.2] | **+4.5** [+3.5, +5.6] |
| Run 30 D-8 − run 29 D-8, exact | **+22.4** [+20.8, +24.0] | **+10.6** [+5.0, +15.6] | **+11.3** [+8.2, +14.5] |
| Run 30 D-8 − run 29 D-8, char | **+5.8** [+5.4, +6.3] | **+2.3** [+0.9, +3.8] | **+4.1** [+3.1, +5.3] |
| Run 30 D-12 − run 28, exact | **+30.5** [+28.6, +32.2] | **+16.5** [+10.6, +22.5] | **+25.1** [+21.6, +28.8] |
| Run 30 D-12 − run 28, char | **+8.1** [+7.6, +8.7] | **+4.0** [+2.4, +5.5] | **+8.1** [+6.8, +9.3] |
| Run 30 D-12 − D-8, exact | +0.9 [−0.1, +1.9] | +1.4 [−2.3, +5.0] | **+4.4** [+1.7, +7.2] |
| Run 30 D-12 − D-8, char | +0.2 [−0.1, +0.4] | +0.6 [−0.2, +1.6] | **+1.2** [+0.4, +2.0] |
| Run 30 D-12 ep 18 − ep 15, exact | **+1.9** [+1.0, +2.9] | **+4.6** [+0.5, +8.7] | −0.3 [−2.7, +2.1] |
| Run 30 D-8 ep 18 − ep 15, exact | **+2.9** [+1.8, +3.9] | +2.8 [−1.4, +6.4] | −0.5 [−2.9, +1.9] |

Bold marks intervals that exclude zero. On the benchmark, run 30 D-12 reads
703 photos that run 29 D-12 misses, and run 29 reads 48 that run 30 misses.

### Against the ManchuAI-OCR models (exact)

| Model | Training data | Paper set (218) | Repository set (753) |
|---|---|---|---|
| [CRNN v3](https://huggingface.co/mic7ch/crnn-manchu-ngsyd) (checkpoint 426284) | real handwriting + our photos (split by image) + 60k synthetic | 97.7% | 97.3% |
| LLaMA-3.2-11B | 60k synthetic, LoRA | 93.1%* | 87.4% |
| **Run 30 D-12/64 ep 18** | renders + 85% of our photos (split by word) | **90.4%** | **85.5%** |
| Run 30 D-8/64 ep 18 | the same | 89.0% | 81.1% |
| Run 29 D-12/64 ep 14 | renders only | 76.1% | 70.3% |
| Run 28 | renders only | 73.9% | 60.4% |
| CRNN-3M | 60k synthetic | 72.5%* | 56.8% |

\* Checkpoint chosen on this test set, so the figure is optimistic.
ManchuAI figures are from `run028/comparison_manchuai_ocr.md`.

Paired on the 715 repository rows whose labels still match the
predictions the repository publishes:

| Ours vs theirs | Ours | Theirs | Right only in ours | Right only in theirs | McNemar z |
|---|---|---|---|---|---|
| Run 30 D-12 vs LLaMA-3.2-11B | 85.3% | 88.0% | 71 | 90 | −1.50 (p ≈ 0.13) |
| Run 30 D-8 vs LLaMA-3.2-11B | 80.8% | 88.0% | 66 | 117 | −3.77 |
| Run 29 D-12 vs LLaMA-3.2-11B | 70.5% | 88.0% | 55 | 180 | −8.15 |
| Run 30 D-12 vs CRNN-3M | 85.3% | 56.1% | 248 | 39 | +12.34 |

**Run 30 D-12 can no longer be told apart from LLaMA-3.2-11B on the
repository set.** It's 2.7 points behind, which isn't significant, on one
laptop core instead of a 48 GB GPU. On the paper set it's 2.7 points
behind LLaMA's test-selected figure. CRNN v3 is still about 7–12 points
ahead, but it trained on real handwriting.

None of the ManchuAI models can be scored on our photo benchmark. CRNN v3
and the 2026 models trained on these photos split by image, so they've
seen every benchmark word. The paper's synthetic-only models were never run
on photos.

## Our models, all passes (exact / char)

### Photo benchmark (3,150)

| Pass | Run 30 D-12 ep 18 | Run 30 D-8 ep 18 | Run 30 D-12 ep 15 | Run 30 D-8 ep 15 | Run 29 D-12 ep 14 | Run 29 D-8 ep 14 | Run 28 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 92.9% / 98.5% | 91.9% / 98.3% | 90.9% / 98.1% | 89.1% / 97.6% | 72.1% / 93.3% | 69.6% / 92.5% | 62.4% / 90.4% |
| whitened | 88.1% / 97.4% | 86.6% / 97.2% | 84.1% / 96.8% | 83.5% / 96.3% | 69.8% / 92.8% | 68.8% / 92.4% | 61.0% / 89.9% |
| cropped | 89.2% / 97.7% | 85.9% / 96.8% | 82.6% / 96.2% | 82.0% / 96.0% | 67.5% / 91.7% | 61.5% / 89.6% | 56.6% / 87.9% |
| plain | 80.6% / 95.5% | 79.0% / 94.9% | 74.1% / 93.9% | 73.8% / 93.6% | 63.7% / 90.5% | 59.9% / 89.1% | 50.7% / 85.6% |

The training photos are prepared whitened + cropped, so that pass gains the
most. The others gain too, by 13–17 points exact for D-12.

### Paper set (218)

| Pass | Run 30 D-12 ep 18 | Run 30 D-8 ep 18 | Run 30 D-12 ep 15 | Run 30 D-8 ep 15 | Run 29 D-12 ep 14 | Run 29 D-8 ep 14 | Run 28 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 90.4% / 98.1% | 89.0% / 97.4% | 85.8% / 97.3% | 86.2% / 96.7% | 76.1% / 94.4% | 78.4% / 95.1% | 73.9% / 94.1% |
| cropped | 90.4% / 98.1% | 89.9% / 97.6% | 85.3% / 97.1% | 86.2% / 96.7% | 77.1% / 94.7% | 77.1% / 94.9% | 76.6% / 94.7% |
| whitened | 9.2% / 75.6% | 24.8% / 74.0% | 10.6% / 73.2% | 56.0% / 87.9% | 5.0% / 70.8% | 15.6% / 75.2% | 46.8% / 83.1% |
| plain | 8.7% / 75.2% | 26.6% / 75.4% | 11.0% / 73.8% | 56.9% / 87.9% | 5.5% / 71.1% | 19.3% / 76.5% | 46.8% / 82.8% |

### Repository set (753)

| Pass | Run 30 D-12 ep 18 | Run 30 D-8 ep 18 | Run 30 D-12 ep 15 | Run 30 D-8 ep 15 | Run 29 D-12 ep 14 | Run 29 D-8 ep 14 | Run 28 |
|---|---|---|---|---|---|---|---|
| whitened + cropped | 85.5% / 96.8% | 81.1% / 95.5% | 85.8% / 96.7% | 81.7% / 95.5% | 70.3% / 92.3% | 69.9% / 91.4% | 60.4% / 88.7% |
| cropped | 85.5% / 96.8% | 81.1% / 95.5% | 85.8% / 96.7% | 81.7% / 95.5% | 70.3% / 92.3% | 69.9% / 91.4% | 60.4% / 88.7% |
| whitened | 2.5% / 71.0% | 26.0% / 75.5% | 6.1% / 71.9% | 60.0% / 88.9% | 0.9% / 67.6% | 21.6% / 74.8% | 38.0% / 79.6% |
| plain | 2.5% / 71.0% | 26.0% / 75.5% | 6.1% / 71.9% | 60.0% / 88.9% | 0.9% / 67.6% | 21.6% / 74.8% | 38.0% / 79.6% |

## Uncropped handwriting: the trailing space is still there

Run 30 still ends almost every guess on uncropped, padded handwriting with
a space: 96.9% of D-12's plain guesses on the repository set (run 29: 97.9%).
With the space stripped, its plain pass would be 65.5% exact. D-8 is
erratic: 44.0% of its guesses at epoch 18, and much fewer at epoch 15 (which
scores 60.0% plain). Cropped input never shows it (≤ 0.1%), and the
benchmark photos don't either (0.0%). It doesn't matter for the server,
which always crops.

## By word length (photo benchmark, D-12, whitened + cropped, exact)

| Letters | Photos | Run 29 ep 14 | Run 30 ep 18 |
|---|---|---|---|
| 1–3 | 250 | 85.2% | 94.8% |
| 4–7 | 2,200 | 72.4% | 92.7% |
| 8+ | 700 | 66.4% | 92.6% |

Run 29 got worse as words got longer. Run 30 is flat across lengths.

## Conclusions

1. **Training on photos is the biggest step yet.** Run 30 D-12 reads 92.9%
   of benchmark photos exactly (98.5% char, CER 0.014), against run 29's
   72.1%. That's +20.8 points on words it never trained on. The monitor
   set agrees (94.2% at epoch 18). One caveat: the benchmark is held out by
   word, but it comes from the same books and photography as the training
   photos, so a new book or camera could score lower.
2. **It carries over to handwriting, which run 30 has never seen:** +14 to
   +15 points exact for D-12 on both ManchuAI sets, to 90.4% and 85.5%.
   Real print photos seem to teach it something renders don't.
3. **It's now level with LLaMA-3.2-11B on the repository set** (85.3% vs
   88.0% paired, z = −1.50) and 29 points ahead of CRNN-3M. Only CRNN v3,
   trained on real handwriting, is clearly ahead.
4. **D-12 against D-8:** level on photos (+0.9, interval includes zero),
   ahead by 4.4 points on the repository set.
5. **Still gaining on photos:** ep 15 → 18 added +1.9 (D-12) and +2.9 (D-8)
   on the benchmark. On the repository set, ep 15 and ep 18 are level.
6. **ManchuOCR is now far behind everywhere.** On the benchmark it gets
   69.2% / 91.2% with our preprocessing first, 23.7 points exact behind
   run 30 D-12. Run 29 was only 2.9 points ahead of it, and run 28 was 6.8
   behind. On handwriting, run 30 leads its best setup by 28–30 points.

Per-image results and the scripts are in the session scratchpad only.
ManchuOCR's repository has no license, so this report gives only its
aggregate results. Its code, weights and guesses are not in this
repository.
Neither handwritten dataset states a license, so this report gives aggregate
results only.
