# Photographed words: runs 25, 26 and 28

All three runs are D-8/64, Adam 3e-3.

| Run | Training data | Augmentation | Length | Training images |
|---|---|---|---|---|
| 25 | old 1,989-word lexicon, mixed padding | none | 45 epochs | ~0.69M |
| 26 | same as 25 | print augmentation | 90 epochs | ~1.39M |
| 28 | 21,532-word lexicon (Norman added), 12 fonts (Abkai Xanyan added), 85/15 split | print augmentation | 10 epochs | ~2.20M |

Image counts are training batches in each log × 64. One epoch of run 28 is
3,434 batches, against 241 for run 26, so run 26's 90 epochs equal about 6.3
of run 28's epochs. Run 28 has also seen ~1.6× as many images in total, so
part of its gain could just be more training. The epoch-5 snapshot below
controls for that.

All three were scored with `ocr/evaluate_non_synthetic.py` on the same 5,000
photographed words (seed 0, 2,175 distinct labels), each under all four
preprocessing passes, so every comparison is paired.

## Headline

| Pass | Exact 25 | Exact 26 | Exact 28 | Char 25 | Char 26 | Char 28 | Mean edits 25 → 26 → 28 |
|---|---|---|---|---|---|---|---|
| plain | 0.2% | 27.0% | **49.3%** | 11.6% | 72.4% | **85.3%** | 5.50 → 1.83 → 0.90 |
| cropped | 0.9% | 35.5% | **55.0%** | 22.6% | 79.0% | **87.7%** | 4.86 → 1.36 → 0.74 |
| whitened | 12.8% | 41.7% | **59.2%** | 60.4% | 82.0% | **89.6%** | 2.53 → 1.18 → 0.64 |
| whitened + cropped | 16.4% | 47.0% | **62.3%** | 65.9% | 84.9% | **90.5%** | 2.17 → 0.96 → 0.56 |

On the best pass, run 28 gains 15.3 points exact over run 26 (paired
bootstrap 95% CI +13.9 to +16.9) and 5.6 points char (+5.1 to +6.1). The
gain is largest on the raw photos: +22.3 points exact on the plain pass.

**Equal training.** Run 28's epoch-5 snapshot (~1.10M images, 79% of run
26's total) already reaches **53.8% exact / 87.8% char** on whitened +
cropped, against run 26's final 47.0% / 84.9%. That is +6.8 points exact
(CI +5.4 to +8.3) with less training. Most of the rest of the gain to 62.3%
comes from training longer. So run 28's improvement is partly the new data
and partly that it hadn't plateaued. Run 26 had.

## Paired outcomes, run 26 vs run 28 (whitened + cropped)

| | Count |
|---|---|
| Exact in both | 1,940 |
| Exact only in run 26 | 408 |
| Exact only in run 28 | 1,177 |
| Exact in neither | 1,475 |

Character accuracy went up on 1,785 images, down on 622 and stayed the same
on 2,593. The improvement is broad, but run 28 loses more words than run 26
lost to run 25 (408 against 125).

The lost words are shorter on average (5.8 letters, against 6.8 for the words
it gained). Most are single-character slips, such as `jaka → jakan`,
`isabufi → isabuwi`, `efulefi → efuleti`, `duleke → tuleke` and
`adabume → andabume`. A few look like the larger lexicon pulling short or
transliterated words towards real words. `weilembi → beilembi` is one.
`sang` and `xan` are each right in run 26 but wrong in run 28: `sang → teng`
and `sar`, `xan → gean`. Those are small counts but fit run 27's weakness on
short words.

## Preprocessing matters less again

- Plain-pass guesses are now on average only 0.05 characters too short,
  against 0.76 for run 26 and 4.08 for run 25. Empty guesses: 1, 2 and 489.
- The exact-match gap between plain and whitened + cropped fell from 19.9
  points (run 26) to 13.0. The char gap fell from 12.5 to 5.2.
- Whitening + cropping is still the best pass and should stay the default.
- If you could pick the best pass per image (an oracle, not a real method),
  you would get 77.7% exact / 95.2% char (run 26: 59.1% / 90.1%). That 15-point
  gap over the best single pass is still the ceiling for something like trying
  several preprocessings and keeping the most confident result.

## Where it still fails (whitened + cropped)

**Word length.** This is the biggest change. Run 28's character accuracy is
now flat, or even rising, with length. Long words gained the most, and
exact match on 11+ letter words went from 13% to about 50%.

| Label length | n | Exact 25 | Exact 26 | Exact 28 ep. 5 | Exact 28 | Char 26 | Char 28 |
|---|---|---|---|---|---|---|---|
| 1–4 | 1,112 | 33.5% | 67.5% | 65.6% | 72.9% | 86.4% | 88.6% |
| 5–6 | 1,889 | 18.4% | 51.8% | 56.5% | 64.4% | 85.5% | 90.5% |
| 7–8 | 1,339 | 7.2% | 37.3% | 47.8% | 57.9% | 84.5% | 91.5% |
| 9–10 | 481 | 0.6% | 19.8% | 39.7% | 47.0% | 81.2% | 90.8% |
| 11–13 | 167 | 0.0% | 13.2% | | 49.7% | 81.4% | 92.9% |
| 14+ | 12 | 0.0% | 16.7% | | 50.0% | 85.7% | 95.8% |

(Epoch 5 on 11+ letters combined: 34.6%.) At equal training, short words
(1–4 letters) did slightly *worse* than in run 26 (65.6% vs 67.5%). Short
words only pull ahead with run 28's extra training. This matches run 27's
finding that the new lexicon has few short words (1.6% of entries against
8.9% of photos) and lots of long words and phrases. The new data teaches long
words well and short words less well.

**Image height** (quartile edges 64, 79 and 99 px) follows the same pattern:

| Height quartile | Exact 25 | Exact 26 | Exact 28 |
|---|---|---|---|
| shortest | 28.1% | 63.8% | 72.2% |
| 2nd | 17.7% | 50.5% | 64.4% |
| 3rd | 13.1% | 42.3% | 59.5% |
| tallest | 7.0% | 31.9% | 53.6% |

**Error types.** Total edits: 10,840 → 4,819 → 2,815. Deletions fell the
most again (3,312 → 1,360 → 628). Insertions are now about as common as
deletions (604 against 628).

| Error | Run 25 | Run 26 | Run 28 |
|---|---|---|---|
| `o` read as `u` | 182 | 254 | 182 |
| `a` dropped | 685 | 344 | 177 |
| `a` read as `e` | 287 | 183 | 130 |
| extra `a` inserted | 143 | 164 | 121 |
| `n` dropped | 431 | 155 | 111 |
| extra `n` inserted | 48 | 70 | **89** |
| `g` read as `h` | 229 | 134 | 88 |
| extra `e` inserted | 190 | 44 | **75** |
| `e` read as `a` | 165 | 129 | 74 |
| `d` read as `t` | 78 | 56 | **67** |
| `e` dropped | 397 | 241 | 61 |
| extra `k` inserted | 71 | 163 | 27 |

- **Run 26's two new problems mostly went away.** Spurious `k` insertions fell
  from 163 to 27. `o → u` fell back to run 25's level (254 → 182). Both point
  more to run 26's narrow data than to the augmentation noise. But `o → u` is
  now the most common error, so the o/u mark is still hard.
- **A few errors got worse:** extra `n` (70 → 89), extra `e` (44 → 75) and
  `d → t` (56 → 67). Run 27 had also flagged d. These are small counts, but
  extra `n` and `e` fit a lexicon with many `-n` and long `-e-` words.

## Training curve

The 500 fixed photos scored after each epoch (seed 1, preprocessed like
whitened + cropped), lined up by training images rather than epochs:

| Training images | ~0.22M | ~0.44M | ~0.66M | ~0.88M | ~1.10M | ~1.32M | ~1.54M | ~1.76M | ~1.98M | ~2.20M |
|---|---|---|---|---|---|---|---|---|---|---|
| Run 28 epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| Run 28 photo exact | 13.8% | 32.4% | 38.4% | 54.2% | 53.0% | 51.0% | 57.6% | 56.2% | 61.6% | 62.6% |
| Run 28 photo char | 66.0% | 80.6% | 83.0% | 88.1% | 87.5% | 87.1% | 89.3% | 87.7% | 90.8% | 90.7% |
| Run 26 epoch (same images) | 14 | 28 | 43 | 57 | 71 | 85 | – | – | – | – |
| Run 26 photo exact | 13.6% | 29.8% | 36.0% | 42.0% | 42.0% | 44.4% | – | – | – | – |

- The two runs track each other for the first ~0.66M images. From there run
  28 pulls ahead by about 10 points while run 26 levels off. The new data
  doesn't make learning faster. It raises the ceiling.
- Run 28 is still rising at epoch 10: 51 → 58 → 56 → 62 → 63% over epochs
  6–10. The epoch-to-epoch noise on 500 photos is about ±4 points, but the
  trend is clear. **Training longer should help**, unlike run 26.
- Run 28's synthetic test reaches 97.3%. It uses a different lexicon and a
  different 85/15 split, so it can't be compared with run 26's 95.4%.

**Check for leakage.** 122 of the 5,000 eval photos are also in the 500-photo
monitoring set. The weights scored are the final epoch's, not chosen by photo
score, so this doesn't bias the result. Run 28 gets 67.2% exact on those 122
and 62.2% on the other 4,878. The difference is within noise (the standard
error on 122 is about ±4 points).

## Words in and out of the lexicon (whitened + cropped)

| Photographed word is | n | Exact 25 | Exact 26 | Exact 28 | Char 26 | Char 28 |
|---|---|---|---|---|---|---|
| in the old lexicon (all runs trained on it) | 1,243 | 25.1% | 59.1% | 68.5% | 87.0% | 90.7% |
| added from Norman (only run 28 trained on it) | 1,426 | 16.6% | 48.4% | 63.5% | 85.1% | 90.5% |
| in neither lexicon | 2,331 | 11.6% | 39.6% | **58.4%** | 83.6% | **90.3%** |

The biggest gain is on words **no run trained on**: +18.8 points exact.
Character accuracy is now the same, about 90.5%, whether or not the word is
in the lexicon. The gap between words the model has seen and words it hasn't
shrank from 19.5 points exact (run 26) to 10.1. So the larger, more varied
lexicon taught the model to read letter by letter, not just to recognise more
words. That's what the open-vocabulary goal needs.

## Takeaways

1. Run 28 is the new best: 62.3% exact / 90.5% char on whitened + cropped,
   49.3% exact on raw photos. On whitened + cropped that's +15 points over
   run 26 and +46 over run 25.
2. With equal training it is still +7 points ahead of run 26, and it hasn't
   plateaued. **Train it longer** (or resume from epoch 10). Run 26 had
   stopped improving by epoch ~60.
3. The gain comes mostly from long words and from words outside the lexicon.
   Word length is no longer what drives character errors.
4. Short words are now the relative weak spot. At equal training they were
   slightly worse than run 26, matching run 27's finding that the new lexicon
   has few 1–3 letter words. Adding more short words and particles to the
   training mix would target this directly.
5. The `k` insertions and the extra `o → u` errors from run 26 are gone.
   `o → u` is still the most common error. Extra `n`/`e` insertions and
   `d → t` went up a little and are worth watching.
6. Keep whitening + cropping as the default. The best-pass oracle (77.7%)
   shows there's still about 15 points available from combining several
   preprocessings.
