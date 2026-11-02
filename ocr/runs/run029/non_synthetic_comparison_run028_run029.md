# Photographed words: runs 28 and 29

| Run | Models | Split | Length |
|---|---|---|---|
| 28 | D-8/64 | 85/15 by image, so photo words could be in training | 10 epochs |
| 29 | C-8/64, D-8/64, D-12/64 | by word (`ocr/split.py`), all photo words held out | 10 epochs so far |

Every model uses Adam at a constant 3e-3, print augmentation and the same
21,532-word lexicon and 12 fonts. An epoch is about 4,040 batches in both
runs, training and test together. Run 29 was trained in chunks: epochs 1–4, 5–8 and 9–10. Adam's state
wasn't saved yet when epoch 4 finished, so epoch 5 started with a fresh Adam.
Epochs 9–10 resumed with the saved state.

Every checkpoint was scored with `ocr/evaluate_non_synthetic.py` on the same
5,000 photographed words (seed 0) under all four preprocessing passes, so
every comparison is paired. The script only loads ModelD, so C-8/64 was run
through a small wrapper that swaps in ModelC. Intervals are paired bootstrap
95% CIs (2,000 resamples). Only epochs 5, 8 and 10 of run 29 could be
scored: epoch 9's weights were overwritten by epoch 10.

Per-photo results for run 29's epoch-10 checkpoints are in this folder as
`non_synthetic_eval_<model>_epoch010_<pass>.csv`. Run 28's are in
`ocr/runs/run028/`.

## Headline: epoch 10

Exact words, with the change from run 28 in points:

| Pass | Run 28 | D-12/64 | D-8/64 | C-8/64 |
|---|---|---|---|---|
| plain | 49.3% | 49.1% (−0.2 [−1.7, +1.2]) | 45.3% (−4.0 [−5.5, −2.5]) | 39.4% (−9.9 [−11.4, −8.5]) |
| cropped | 55.0% | 52.8% (−2.2 [−3.7, −0.7]) | 49.9% (−5.1 [−6.6, −3.5]) | 41.3% (−13.7 [−15.2, −12.2]) |
| whitened | 59.2% | **60.3%** (+1.1 [−0.3, +2.5]) | 58.7% (−0.5 [−1.9, +0.9]) | 47.2% (−12.0 [−13.4, −10.6]) |
| whitened + cropped | **62.3%** | 60.7% (−1.6 [−3.0, −0.2]) | 61.0% (−1.3 [−2.6, +0.0]) | 50.4% (−12.0 [−13.4, −10.4]) |

Character accuracy:

| Pass | Run 28 | D-12/64 | D-8/64 | C-8/64 |
|---|---|---|---|---|
| plain | 85.3% | 84.6% | 83.8% | 81.5% |
| cropped | 87.7% | 86.0% | 85.6% | 81.5% |
| whitened | 89.6% | 89.5% | 89.3% | 84.8% |
| whitened + cropped | **90.5%** | 89.3% | 89.6% | 85.6% |

On whitened + cropped, D-12/64 reads 629 words that run 28 gets wrong and
misses 709 that run 28 gets right. For D-8/64 it's 594 and 660, and for
C-8/64 448 and 1,046.

At equal training, neither D model beats run 28. Both are level with it or
slightly behind: within about 1.5 points on the best pass and on whitened
alone. C-8/64 is 10–14 points behind on every pass.

## Equal training: epoch 5

Exact words:

| Pass | Run 28 | D-12/64 | D-8/64 | C-8/64 |
|---|---|---|---|---|
| plain | 36.9% | 48.2% | 45.6% | 36.8% |
| cropped | 43.0% | 55.5% | 49.5% | 38.3% |
| whitened | 49.8% | 56.3% | 53.8% | 46.1% |
| whitened + cropped | 53.8% | **61.2%** | 55.9% | 47.6% |

On whitened + cropped the changes from run 28 are +7.4 points (CI +6.0 to
+8.8) for D-12/64, +2.1 (+0.7 to +3.6) for D-8/64 and −6.2 (−7.7 to −4.7)
for C-8/64. At epoch 5 D-12/64 is ahead of run 28 by 6.5–12.5 points on
every pass. It learns faster, but it hasn't improved since.

## Over training

Exact words on whitened + cropped, 5,000 photos:

| Epoch | Run 28 | D-12/64 | D-8/64 | C-8/64 |
|---|---|---|---|---|
| 5 | 53.8% | 61.2% | 55.9% | 47.6% |
| 8 | — | 61.8% | 57.2% | 52.3% |
| 10 | 62.3% | 60.7% | 61.0% | 50.4% |

`photo_exact` from the training logs (the fixed 500-photo set, seed 1,
whitened + cropped):

| Epoch | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| Run 28 D-8/64 | 53.0% | 51.0% | 57.6% | 56.2% | 61.6% | 62.6% |
| Run 29 D-12/64 | 60.0% | 58.8% | 56.2% | 60.0% | 62.2% | 57.6% |
| Run 29 D-8/64 | 50.4% | 55.2% | 53.6% | 52.2% | 53.2% | 58.6% |
| Run 29 C-8/64 | 49.2% | 50.4% | 45.4% | 51.6% | 48.0% | 49.2% |

The synthetic loss was still falling steadily at epoch 8. For D-12/64, mean
test CTC loss went 0.161, 0.144, 0.138 and 0.128 over epochs 5–8. The photo
score is flat and swings 2–4 points from one epoch to the next. With Adam,
each weight moves about the learning rate on every step, so at a constant
3e-3 the weights keep moving instead of settling. The next step to try is
lowering the rate from the epoch-10 snapshots, for example decaying from
3e-3 to about 3e-4.

The D-8/64 rerun also shows how much a single run varies: it was ahead of
run 28 at epoch 5, 5 points behind on the 5,000 photos at epoch 8 (run 28's
epoch-8 weights weren't kept, so that comparison is with run 28's final
model), and level again at epoch 10. The CIs above only cover which photos
were sampled. Differences of a point or two between single checkpoints
could be run-to-run variation.

## Does the old split favour run 28?

Run 28 predates the word split, so it could train on images of words that
are also photographed. Run 29 can't. The 5,000 photos were split by whether
their word is in the lexicon (whitened + cropped, exact):

| | In lexicon (2,669) | Not in lexicon (2,331) |
|---|---|---|
| Run 28 | 65.8% | 58.4% |
| D-12/64 | 63.8% (−2.0 [−3.9, −0.2]) | 57.3% (−1.1 [−3.3, +1.1]) |
| D-8/64 | 63.5% (−2.2 [−4.0, −0.4]) | 58.1% (−0.3 [−2.3, +2.0]) |
| C-8/64 | 53.8% (−12.0) | 46.5% (−11.9) |

Run 28's lead is a little larger on lexicon words, but the two groups'
intervals overlap. Run 29 also does about 6.5 points better on lexicon words
than on the others, even though it never trained on any of these photo
words. So most of that gap is that lexicon words are easier to read, not
memorization. Having seen these words helped run 28 by about a point at
most, so this is a roughly fair comparison.

## Summary

- At 10 epochs, run 29's D models are level with run 28 or slightly behind:
  D-8/64 61.0% and D-12/64 60.7% against 62.3% on whitened + cropped.
  D-12/64 is ahead on whitened alone (60.3% against 59.2%), but within the
  interval.
- D-12/64 learns fastest (61.2% at epoch 5, +7.4 over run 28) but hasn't
  improved since epoch 5.
- C-8/64 is clearly worse at every checkpoint. Drop it.
- The swings between epochs at a constant learning rate suggest decaying it
  next, from the epoch-10 snapshots of D-12/64 and D-8/64.
