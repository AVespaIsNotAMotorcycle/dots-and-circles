# Photographed words: run 25 vs run 26

Both runs are D-8/64, Adam 3e-3, trained on the same mixed-padding data. The
difference is that run 26's training (and test) images were augmented each
epoch to look like photographed prints (`ocr/training_data/augment.py`), and it
trained for 90 epochs instead of 45. Both were scored with
`ocr/evaluate_non_synthetic.py` on the same 5,000 photographed words (seed 0,
2,175 distinct labels), each under all four preprocessing passes. The images are
paired, so every comparison below is on exactly the same photos.

## Headline

| Pass | Exact, run 25 | Exact, run 26 | Char acc., run 25 | Char acc., run 26 | Mean edits, 25 → 26 |
|---|---|---|---|---|---|
| plain | 0.2% | **27.0%** | 11.6% | **72.4%** | 5.50 → 1.83 |
| cropped | 0.9% | **35.5%** | 22.6% | **79.0%** | 4.86 → 1.36 |
| whitened | 12.8% | **41.7%** | 60.4% | **82.0%** | 2.53 → 1.18 |
| whitened + cropped | 16.4% | **47.0%** | 65.9% | **84.9%** | 2.17 → 0.96 |

Exact-word accuracy roughly triples on the best pass (16.4% → 47.0%; paired
bootstrap 95% CI for the gain is +29.2 to +32.0 points). Mean edit distance
halves, so run 26 is now about one edit per word away from the label.

## Paired outcomes (whitened + cropped)

| | Count |
|---|---|
| Exact in both | 694 |
| Exact only in run 25 | 125 |
| Exact only in run 26 | 1,654 |
| Exact in neither | 2,527 |

Character accuracy went up on 3,275 images, down on 359 and stayed the same
on 1,366. The improvement is broad, not a few words flipping.

The 125 words run 26 lost are mostly single-character slips on short common
words, such as `gebu → hebu`, `guwe → guw`, `jeke → eke`, `isinaha → isikaha`,
`usiha → usiaha`. These are typical near-misses, not a systematic regression.

## Preprocessing matters much less now

- Run 25 was nearly useless without whitening: 489 of the 5,000 plain-pass
  guesses were empty strings, and guesses were 4 characters too short on
  average. Run 26 leaves 2 empty guesses and is only 0.8 characters short.
- On character accuracy, the gap between plain and whitened + cropped fell
  from 54.3 points to 12.5, so the raw photos are now mostly readable. On
  exact match the gap actually grew a little (16.2 → 19.9 points), because
  run 26 now gets enough characters right for exact matches to depend on the
  last few slips.
- Whitening + cropping is still the best pass for run 26, and it is the one
  the augmentation was designed to imitate (`load_photo`'s defaults), so it
  should stay the default.
- If you could pick the best pass per image (an oracle, not a real method),
  run 26 would reach 59.1% exact / 90.1% char (run 25: 22.7% / 72.6%). That
  is a ceiling for something like trying several preprocessings and keeping
  the most confident result.

## Where it still fails (whitened + cropped)

**Word length.** Long words are still the main weakness, though character
accuracy is now nearly flat across lengths. Exact match falls because the
chance of a slip accumulates over more characters.

| Label length | n | Exact, 25 → 26 | Char acc., 25 → 26 |
|---|---|---|---|
| 1–4 | 1,112 | 33.5% → 67.5% | 67.6% → 86.4% |
| 5–6 | 1,889 | 18.4% → 51.8% | 68.6% → 85.5% |
| 7–8 | 1,339 | 7.2% → 37.3% | 64.9% → 84.5% |
| 9–10 | 481 | 0.6% → 19.8% | 58.8% → 81.2% |
| 11–13 | 167 | 0.0% → 13.2% | 53.9% → 81.4% |
| 14+ | 12 | 0.0% → 16.7% | 51.8% → 85.7% |

**Image height** (after cropping; quartile edges are 64, 79 and 99 px) mostly
tracks word length, with the same pattern: 63.2% exact in the shortest quartile,
31.1% in the tallest.

**Error types.** Total edits fell from 10,840 to 4,819. Deletions dropped the
most (3,312 → 1,360), which fits the "guesses too short" fix above. The most
common errors now are:

| Error | Run 25 | Run 26 |
|---|---|---|
| `a` dropped | 685 | 344 |
| `o` read as `u` | 182 | 254 |
| `e` dropped | 397 | 241 |
| `a` read as `e` | 287 | 183 |
| extra `a` inserted | – | 164 |
| extra `k` inserted | – | 163 |
| `n` dropped | 431 | 155 |
| `g` read as `h` | 229 | 134 |

Two things are new or worse in run 26:

- **`o → u` confusion went up** (182 → 254). This is the only common error
  that got worse. In print, the o/u distinction rests on a small mark and the
  augmentation's blur/noise may make it unreliable to learn.
- **Spurious `k` and `a` insertions** (163 and 164), for example
  `acanaha → acakaha` and `tuwaha → tuwakha`. These look like a stroke or
  speck being read as an extra tooth, which could be an artifact of the
  augmentation's noise.

The a/e and g/h confusions are mostly about dots and contextual shape
differences and were already the main confusions in run 25.

## Training curve (run 26 log)

The 500 fixed photos scored each epoch (seed 1, preprocessed like
whitened + cropped):

| Epoch | 10 | 20 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|
| Photo exact | 7.0% | 20.8% | 28.6% | 39.4% | 42.4% | 47.8% | 45.2% |
| Photo char | 56.1% | 71.3% | 76.2% | 81.0% | 83.3% | 86.1% | 84.9% |
| Synthetic test | 65.2% | 83.6% | 88.6% | 92.8% | 94.3% | 95.1% | 95.4% |

- Even at epoch 45, the same length as run 25, run 26 already reached 39.4% on
  the photo set. So most of the gain comes from the augmentation, not the
  extra 45 epochs.
- Photo exact plateaus from about epoch 60 at 42–49% while synthetic test
  keeps creeping up. The epoch-to-epoch noise on 500 photos is about ±4
  points, so 66 (49.2%) and 77 (49.2%) aren't meaningfully better than 90.
  More epochs of the same recipe probably won't help photos much.
- Run 26's synthetic test set is augmented too, so its 95.4% isn't directly
  comparable to run 25's 92.7% on clean images. At equal epochs (45) they're
  the same (92.8% vs 92.7%) even though run 26's test is harder, so the
  augmentation doesn't look like it cost synthetic accuracy. The clean
  synthetic set hasn't been re-scored with run 26's weights to confirm.

**Check for leakage.** 122 of the 5,000 eval photos are also in the 500-photo
monitoring set. The weights were the final epoch's, not chosen by photo
score, so this doesn't bias the result: run 26 gets 45.1% exact on those 122
and 47.0% on the other 4,878. If you ever pick a checkpoint by `photo_exact`,
evaluate on a sample that excludes the seed-1 set.

## Takeaways

1. The print augmentation works: whitened + cropped goes from 16% to 47%
   exact and 66% to 85% char, and the raw unprocessed photos go from useless
   to 27% exact.
2. Keep whitening + cropping. It's still worth +20 points exact over plain.
3. The rest of the gap is now mostly per-character slips that compound in
   long words. 75% of the photographed words aren't in the training lexicon,
   and run 26 gets 43.0% exact (84.2% char) on those vs 59.1% (87.0%) on
   words it trained on. The model has to read unseen words (transliterations,
   unknown words), so the gain should come from training it to read letter by
   letter, for example on made-up words, not from constraining its output.
4. Worth a look in the augmentation: why `o/u` got worse, and whether the
   noise causes the spurious `k`/`a` insertions.
5. Photo accuracy has plateaued by about epoch 60–75, so the next gain is more
   likely to come from changes to the training data or the model than from
   training longer.
