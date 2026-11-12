# Run 31 against run 30: single words and whole pages

2026-11-12. Run 31 is run 30's D-12/64 (epoch 18 weights and Adam state)
fine-tuned for epochs 19–22 with two new output classes (᠈ ᠉) and
61,495 composites per epoch: tight 2–3-word phrases, the lone genitive i
and punctuated words, all from training words (`training_data/compose.py`).
Only D-12 trained; `weights_D-8-*` in this folder is run 30's, copied.

## Checkpoints

| Name | Weights |
|---|---|
| Run 30 ep 18 | `run030/weights_D-12-64_Adam_3e-3.npz` (= `server/weights/run030.npz`) |
| Run 31 ep 20 | `run031/weights_D-12-64_Adam_3e-3_epoch020.npz` |
| **Run 31 ep 22** | `run031/weights_D-12-64_Adam_3e-3.npz` (latest) |

No checkpoint was chosen by a test score. The monitor photos (500, logged)
give ep 20 and ep 22 the same 94.6% exact (run 30 ep 18: 94.2%).

## Single words

The sets, the handling of the handwritten sets, and the metrics all match
`run030/comparison_epoch018_benchmark_manchuai.md`. The handwritten
labels' U+202F is dropped, which reproduces that report exactly. Run 30's
figures below all reproduce it. Pass: whitened + cropped. Paired
differences use 95% bootstrap intervals over images.

| Test set | Run 30 ep 18 | Run 31 ep 20 | **Run 31 ep 22** |
|---|---|---|---|
| Photo benchmark (3,150), exact / char | 92.9% / 98.5% | **94.6% / 98.9%** | 94.0% / 98.7% |
| Paper set (218 handwritten) | **90.4% / 98.1%** | 90.4% / 97.6% | 89.9% / 98.1% |
| Repository set (753 handwritten) | 85.5% / 96.8% | 86.1% / 96.9% | **87.4% / 97.0%** |
| Synthetic test, char (training log, last epoch) | 99.04% | 99.06% | 99.10% |

| Run 31 − run 30, exact (points) | ep 20 | ep 22 |
|---|---|---|
| Photo benchmark | **+1.7** [+0.9, +2.6], fixed 122 / broke 67 | **+1.2** [+0.3, +2.1], 119 / 82 |
| … clustered by word (315 words) | [+0.7, +2.8] | [+0.3, +2.2] |
| Paper set | +0.0 [−3.7, +3.7] | −0.5 [−4.6, +3.7] |
| Repository set | +0.5 [−1.7, +2.8] | +1.9 [−0.5, +4.2] (McNemar z 1.53) |

Other passes on the benchmark (exact): plain 80.6 → 84.9%, cropped
89.2 → 90.1%, whitened 88.1 → 89.5% (run 30 → run 31 ep 22).

Training on phrases didn't make run 31 split single words. It put a space
in 1 of 3,150 benchmark guesses and 1 of 753 repository guesses (run 30:
0 and 2). It never wrote punctuation on a single word.

## Whole pages

### Real pages (`ocr/real_pages`, `evaluate_real_pages`, config box classifier)

| Page | Run 30 chars / words | Run 31 ep 20 | Run 31 ep 22 |
|---|---|---|---|
| page001 | 98.1 / 91.1 | 97.9 / 89.9 | 98.1 / 91.1 |
| page002 | 94.0 / 78.2 | 96.0 / 82.8 | 95.3 / 81.6 |
| page003 | **99.1 / 97.4** | 97.4 / 86.8 | 96.1 / 78.9 |
| page004 (handwritten) | **48.8 / 4.3** | 41.7 / −0.5 | 43.4 / −1.4 |
| page005 | 90.5 / 78.9 | 90.7 / 80.3 | 90.7 / 80.3 |
| page006 | 98.6 / 92.1 | 99.5 / 97.4 | 99.0 / 94.7 |
| page007 (tight) | 86.3 / 42.1 | 85.3 / 52.6 | **90.2 / 68.4** |
| **Printed pages pooled** (1,845 chars, 332 words) | 94.9 / 83.1 | 95.2 / 84.3 | 95.2 / 84.0 |
| All pages pooled | **76.3 / 52.9** | 73.7 / 51.8 | 74.3 / 51.2 |
| Punctuation found | 0 / 102 | 11 / 102, 1 extra | 11 / 102, 1 extra |

Word breaks on the printed pages (from aligning each line):

| | Run 30 | Run 31 ep 22 |
|---|---|---|
| Missing breaks (of 293) | 18 | 8 |
| Spurious breaks | 1 | 6 |

Run 31 now splits words the print sets close together: page007's
*beye be*, *yabun be*, *han i*; page002's *mutebumbi sehe*, *jakvn hergen*.
It also breaks some single words: *b i*, *b e*, *seh e*, *deri i hese*,
*wasimbuhai ge*. Most of these spurious breaks put a lone *i* or *e* after a
consonant, the shapes the lone-genitive composites teach. Most of
page003's 3-point loss comes from these. Page004 is the handwritten letter that neither model
reads well; run 31 is 5 points worse there, so it dominates the
all-pages figure.

### Synthetic pages (`evaluate_pages --photo`, 100 pages from seed 100000)

| | Run 30 | Run 31 ep 20 | Run 31 ep 22 |
|---|---|---|---|
| Box precision / recall | 90.3 / 92.2% | 89.2 / 96.9% | 88.9 / 97.0% |
| True boxes merged / split | 365 / 79 | 42 / 120 | 36 / 118 |
| Manchu words read, exact / char | 83.2 / 94.0% (of 2,488) | 80.2 / 94.2% (of 2,770) | 78.3 / 93.6% (of 2,777) |
| Page text, char | 93.0% | 93.0% | 92.8% |
| Seconds a page | 1.19 | 0.99 | 1.00 |

The same 100 seeds, forced tight or not (`--tight yes|no`), run 30 → run 31 ep 22:

| | Normal pages | Tight pages |
|---|---|---|
| Recall | 97.9 → 98.0% | **77.2 → 91.7%** |
| Merged / split | 16 / 78 → 16 / 70 | 1,099 / 138 → 70 / 281 |
| Words read exactly | 87.8% of 2,725 → 87.6% of 2,733 | 59.9% of 1,813 (1,086) → 55.7% of 2,700 (1,504) |
| Page text, char | 94.5 → **95.0%** | 85.8 → 85.0% |

"Words read" covers only matched boxes. On tight pages run 31 matches
900 more boxes, many of them words that were merged before, and reads 418
more words exactly. Its rate per matched box falls only because the set
it's measured on grew. Page text, the end-to-end figure, is level:
merges fixed are offset by new splits (138 → 281).

## Summary

- **Single words:** run 31 is better on the photo benchmark: +1.2 points
  exact (ep 22) and +1.7 (ep 20), both significant even clustered by word.
  On handwriting the differences are within noise (repository +1.9, paper
  −0.5). Nothing on single words got worse.
- **Pages:** the tight-phrase training works as intended on box
  splitting. Missed word breaks are halved on real prints, page007 goes
  from 86.3 to 90.2%, and tight synthetic pages' recall goes from 77 to
  92%. But the model now also breaks words, mostly before *i*/*e*. Page
  text accuracy is therefore about level: printed real pages 94.9 →
  95.2%, synthetic 93.0 → 92.8%. Punctuation went from 0 to 11 of 102
  marks.
- **Epoch 20 vs 22:** ep 20 is slightly better on the photo benchmark,
  and ep 22 slightly better on handwriting and page007. Neither difference
  is clearly real.

The scoring scripts (handwriting, page dumps, word-break counts) are in
the session scratchpad only.
