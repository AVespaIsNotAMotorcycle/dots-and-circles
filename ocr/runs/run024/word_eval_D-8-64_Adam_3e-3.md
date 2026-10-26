# Word-level evaluation: D-8/64 Adam 3e-3, run 24 vs run 22

Weights: `weights_D-8-64_Adam_3e-3.npz` from each run (final epoch, 30 epochs each). Evaluated 2026-10-25 in eval mode (no dropout), greedy CTC decoding, on **every lexigraph × every font**: 1,989 lexigraphs × 11 fonts = 21,879 images.

**What changed between the runs.** Same model, optimizer, learning rate and epoch count. The only change is the data: run 24 trained after commit 9e1ba09, so every image has a seeded random number of blank rows above the word (0 up to 256 − height). Run 22 trained on top-aligned images, where the word always starts at row 0. The two models therefore did not do the same task, so each was evaluated on **both** datasets:

- **top-aligned**: `generate()` with the padding switched off (exactly run 22's data; this rerun reproduces all 21,879 of run 22's earlier guesses).
- **padded**: the current `generate()` (exactly run 24's data).

"Run 22" in the tables below means run 22 on top-aligned images, and "run 24" means run 24 on padded images, i.e. each model on the data it trained on. Where it matters, run 24 on top-aligned images is shown too.

Files:
- `word_eval_D-8-64_Adam_3e-3.csv`: run 24 per image on padded data (the same columns as run 22's file plus `ink_start_row`, the number of blank rows above the word).
- `compare_run022_run024.csv`: one row per image, with both models' guesses and exact flags on both datasets.
- `comparison_run022_run024.png`: training curves, shift test, exact match by length and by font.

**Caveat, as before: train and test are mixed.** The split is random, isn't saved, and differs between the runs, so ~70% of these images were trained on. Train and test character accuracy were about equal in both logs, so this shouldn't change the comparison.

## Key findings

1. **Run 24 is clearly worse: 63.3% exact vs 78.9%** (char accuracy 92.0% vs 95.8%). Its per-character error on single words more than doubled, from 2.7% to 6.4%. It is worse in every font (by 12–20 points), at every length, and for every kind of lexigraph. Image by image, only run 22 gets 4,843 right and only run 24 gets 1,440.
2. **The padding did its job: position no longer matters.** Run 22 falls apart when a word is moved down the image (shift test: 86% → 60% after 4 blank tokens → 5% after 40), and on padded images it gets 4.8%. Run 24 is flat at ~67% for any shift from 4 to 40 tokens. At a fixed word height, its accuracy is the same wherever the word starts. The 2nd/3rd-word collapse inside phrases is also gone: the gap between in-phrase and standalone is now similar for every position (−23 / −23 / −40 points vs run 22's −6 / −27 / −66).
3. **But run 24 is undertrained, not converged.** Its learning curve lags run 22's by about 12 epochs: at epoch 30 it reaches 0.891 test char accuracy, which run 22 passed at about epoch 15. It's still rising by ~0.4 points per epoch (run 22's was rising by ~0.15). In the first two epochs it learned almost nothing (0.1% / 1.0%), where run 22 was already at 23%. Most of the drop looks like "the same model needs more epochs on a harder, position-varying task", not a ceiling.
4. **The new weak spot is the top edge.** Words that start at row 0 are now the rare case: under the uniform padding, a short word starts at row 0 only ~0.5% of the time. Shifted by 0 tokens, run 24 gets 54.6%; shifted by 4 or more, it gets ~67%. On the whole top-aligned dataset it gets 46.3% vs 63.3% padded. Even 1–3 blank rows help: the same images go from 41.4% top-aligned to 54.1%. **If real inputs will be cropped to the ink (top-aligned), prepend a few blank rows before reading them, or train with padding 0 more often.**
5. **The errors are the same kinds as before, only more of them.** One-letter errors are still the majority (58.5%). The tooth-stroke confusions dominate even more: medial ᠠ dropped 1,128× (run 22: 371), final ᠠ read as ᠨ 612× (155). That produces a family of −ka/−ta words read as −kn/−tn in every font (ᠵᠠᡴᠠ, ᡩᡠᡴᠠ, ᠸᠠᡴᠠ, ᡨᠠᡴᠠ, ᠠᡳᡴᠠ: 0/11 each, run 22 got 4–10/11). New confusions have appeared: ᠣ→ᡡ (27 → 208), ᠵ→ᡳ (8 → 154), ᠪ→ᡦ (4 → 115), ᡤ→ᡥ (18 → 107), ᠸ→ᡶ (13 → 93), and ᡴ dropped (46 → 271).
6. **Variation selectors got worse too.** FVS1 recall fell from 52% to 32%, and exact match on lexigraphs with a selector fell from 35.5% to 22.3%. Visible selectors are hit hardest (56.7% → 35.0% exact), and memorised invisible ones are lost too (ᡶᡝᡨᡝ᠋ᠮᠪᡳ: 10/11 → 0/11). FVS2 is still effectively unreadable for both models (17% / 20% recall).
7. **Length still drives word accuracy, more steeply now.** With a per-character error of ~6% instead of ~3%, exact match decays faster with length: 82.5% at 4 characters, 58.9% at 7, 34.9% at 10 (run 22: 91.0 / 80.1 / 65.6). Per-character error is still roughly flat across lengths, so this is still compounding rather than long words being undertrained.
8. **Unchanged:** font ranking (LiuYe and WenJian hardest, YaBai/ZhengBai/ZhengHei easiest), the hopeless squashed >256 px images (1.9%), and spaces (word count right 99.9% for both).
9. 298 lexigraphs (15%) are read right in all 11 fonts (run 22: 697), and 151 are wrong in every font (run 22: 71). Run 24 reads 303 lexigraphs in more fonts than run 22, and 1,269 in fewer.

**Suggestions:** train run 24's setup for longer (or resume from its weights) before judging the padding. Make padding 0 (and small amounts) common, e.g. pad only half the images, or draw the padding from a distribution weighted toward small values. At inference, prepend a few blank rows. The ᠠ/ᠨ tooth confusions are now the single largest error source, so they're the thing to watch as training continues.

Some numbers recomputed here for run 22 differ slightly from run 22's own report, because the method changed: the phrase-position table now strips selectors before counting words, and the shift test uses a different random sample of 2,000 words.

## Tables

### Overall: each model on each dataset

| Model | Evaluated on | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---|---:|---:|---:|---:|
| Run 22 | top-aligned (its training data) | 78.85% | 82.28% | 95.77% | 0.317 |
| Run 22 | padded | 4.81% | 5.01% | 50.60% | 2.963 |
| Run 24 | padded (its training data) | 63.30% | 66.49% | 91.95% | 0.572 |
| Run 24 | top-aligned | 46.26% | 49.00% | 86.13% | 0.928 |

### Training logs (mean over each epoch's batches; train has dropout on)

| Epoch | Run 22 train acc | Run 22 test acc | Run 22 test loss | Run 24 train acc | Run 24 test acc | Run 24 test loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.034 | 0.037 | 3.766 | 0.008 | 0.001 | 5.059 |
| 2 | 0.171 | 0.233 | 2.421 | 0.005 | 0.010 | 3.259 |
| 3 | 0.350 | 0.421 | 1.825 | 0.052 | 0.067 | 2.981 |
| 4 | 0.477 | 0.544 | 1.414 | 0.183 | 0.245 | 2.437 |
| 5 | 0.570 | 0.629 | 1.157 | 0.317 | 0.372 | 2.000 |
| 7 | 0.692 | 0.733 | 0.830 | 0.491 | 0.534 | 1.442 |
| 10 | 0.791 | 0.815 | 0.585 | 0.639 | 0.673 | 1.002 |
| 15 | 0.868 | 0.886 | 0.380 | 0.764 | 0.779 | 0.675 |
| 20 | 0.903 | 0.912 | 0.301 | 0.824 | 0.837 | 0.511 |
| 25 | 0.918 | 0.931 | 0.245 | 0.867 | 0.869 | 0.426 |
| 27 | 0.930 | 0.934 | 0.231 | 0.869 | 0.876 | 0.414 |
| 28 | 0.932 | 0.937 | 0.233 | 0.877 | 0.879 | 0.397 |
| 29 | 0.936 | 0.943 | 0.206 | 0.888 | 0.886 | 0.374 |
| 30 | 0.934 | 0.939 | 0.215 | 0.888 | 0.891 | 0.360 |

### By font

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XM_BiaoHei | 1989 | 82.86% | 66.87% | -16.0 | 86.4% → 70.1% | 96.9% → 93.1% | 0.238 → 0.503 | 51.78% |
| XM_GuFeng | 1989 | 80.69% | 61.39% | -19.3 | 84.3% → 64.8% | 96.2% → 91.6% | 0.281 → 0.601 | 47.96% |
| XM_LiuYe | 1989 | 66.92% | 51.38% | -15.5 | 69.8% → 54.8% | 93.3% → 88.6% | 0.495 → 0.773 | 32.63% |
| XM_ShuKai | 1989 | 74.16% | 57.57% | -16.6 | 78.3% → 60.6% | 94.7% → 90.7% | 0.374 → 0.647 | 36.60% |
| XM_WenJian | 1989 | 72.30% | 51.99% | -20.3 | 75.8% → 54.3% | 93.4% → 87.7% | 0.514 → 0.881 | 38.96% |
| XM_WenQin | 1989 | 79.64% | 66.47% | -13.2 | 82.6% → 69.7% | 95.9% → 93.0% | 0.310 → 0.507 | 49.57% |
| XM_XingShu | 1989 | 77.53% | 65.51% | -12.0 | 80.9% → 68.8% | 95.5% → 92.6% | 0.339 → 0.525 | 47.11% |
| XM_YaBai | 1989 | 84.41% | 70.84% | -13.6 | 87.7% → 74.0% | 97.1% → 94.2% | 0.216 → 0.427 | 57.52% |
| XM_YingBi | 1989 | 79.54% | 65.16% | -14.4 | 82.6% → 68.2% | 95.9% → 92.5% | 0.301 → 0.519 | 42.74% |
| XM_ZhengBai | 1989 | 83.76% | 69.88% | -13.9 | 87.5% → 73.4% | 97.0% → 94.0% | 0.220 → 0.452 | 51.84% |
| XM_ZhengHei | 1989 | 85.57% | 69.23% | -16.3 | 89.3% → 72.6% | 97.5% → 93.6% | 0.196 → 0.455 | 52.19% |

### Variation selectors

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With variation selector(s) | 1496 | 35.49% | 22.26% | -13.2 | 78.4% → 63.6% | 87.3% → 81.9% | 1.008 → 1.432 | 8.42% |
| Without | 20383 | 82.03% | 66.31% | -15.7 | 82.6% → 66.7% | 96.4% → 92.7% | 0.266 → 0.509 | 49.04% |

### By label length (characters, incl. spaces & variation selectors)

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 341 | 88.27% | 83.58% | -4.7 | 88.9% → 83.9% | 88.3% → 83.6% | 0.117 → 0.179 | 64.81% |
| 2 | 374 | 91.18% | 86.63% | -4.5 | 94.1% → 88.5% | 95.3% → 92.6% | 0.094 → 0.147 | 74.06% |
| 3 | 990 | 90.20% | 81.21% | -9.0 | 90.2% → 82.4% | 96.3% → 92.4% | 0.111 → 0.228 | 64.65% |
| 4 | 3355 | 91.00% | 82.50% | -8.5 | 91.5% → 82.7% | 97.5% → 94.9% | 0.101 → 0.205 | 68.91% |
| 5 | 4466 | 86.45% | 73.65% | -12.8 | 88.7% → 76.4% | 96.8% → 93.5% | 0.159 → 0.326 | 55.73% |
| 6 | 3817 | 80.17% | 65.10% | -15.1 | 84.8% → 69.6% | 95.8% → 92.0% | 0.250 → 0.481 | 46.61% |
| 7 | 3146 | 80.13% | 58.84% | -21.3 | 86.5% → 64.1% | 96.5% → 92.2% | 0.246 → 0.547 | 40.15% |
| 8 | 1430 | 73.36% | 47.83% | -25.5 | 79.6% → 55.5% | 95.8% → 90.9% | 0.337 → 0.731 | 30.84% |
| 9 | 1309 | 69.29% | 47.06% | -22.2 | 74.2% → 51.0% | 95.2% → 90.5% | 0.434 → 0.853 | 26.28% |
| 10 | 770 | 65.58% | 34.94% | -30.6 | 73.1% → 39.4% | 95.2% → 89.5% | 0.477 → 1.053 | 18.44% |
| 11 | 671 | 58.12% | 40.24% | -17.9 | 58.7% → 40.5% | 93.7% → 90.0% | 0.697 → 1.100 | 18.78% |
| 12 | 429 | 45.45% | 24.01% | -21.4 | 46.2% → 24.7% | 92.7% → 87.3% | 0.879 → 1.527 | 9.56% |
| 13-15 | 550 | 30.36% | 16.73% | -13.6 | 33.8% → 17.8% | 89.6% → 85.1% | 1.442 → 2.049 | 7.45% |
| 16+ | 231 | 3.90% | 3.90% | +0.0 | 5.6% → 5.2% | 80.9% → 79.3% | 3.957 → 4.221 | 2.60% |

### Single vs multiple word lexigraphs

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single word | 18623 | 83.76% | 68.95% | -14.8 | 87.4% → 72.5% | 96.4% → 92.7% | 0.200 → 0.412 | 51.86% |
| Multiple words | 3256 | 50.77% | 30.96% | -19.8 | 53.1% → 32.0% | 92.3% → 87.6% | 0.986 → 1.487 | 14.28% |

### Image squashed to fit 256 px (original height > 256)

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fits | 21825 | 79.04% | 63.45% | -15.6 | 82.5% → 66.6% | 95.8% → 92.0% | 0.303 → 0.558 | 46.37% |
| Squashed | 54 | 3.70% | 1.85% | -1.9 | 3.7% → 1.9% | 75.1% → 75.6% | 5.981 → 5.981 | 1.85% |

### By number of words

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 words | 18623 | 83.76% | 68.95% | -14.8 | 87.4% → 72.5% | 96.4% → 92.7% | 0.200 → 0.412 | 51.86% |
| 2 words | 2981 | 54.38% | 33.18% | -21.2 | 56.8% → 34.2% | 93.2% → 88.3% | 0.766 → 1.279 | 15.10% |
| 3 words | 209 | 11.48% | 8.61% | -2.9 | 12.9% → 10.0% | 85.0% → 82.2% | 2.502 → 2.914 | 6.22% |
| 4 words | 44 | 18.18% | 2.27% | -15.9 | 18.2% → 2.3% | 79.8% → 78.6% | 3.727 → 3.727 | 4.55% |
| 5 words | 11 | 0.00% | 0.00% | +0.0 | 0.0% → 0.0% | 66.7% → 65.4% | 7.000 → 7.273 | 0.00% |
| 6 words | 11 | 0.00% | 0.00% | +0.0 | 0.0% → 0.0% | 57.7% → 54.5% | 14.818 → 15.909 | 0.00% |

### Single-word lexigraphs by length (isolating length from spaces)

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 341 | 88.27% | 83.58% | -4.7 | 88.9% → 83.9% | 88.3% → 83.6% | 0.117 → 0.179 | 64.81% |
| 2 | 374 | 91.18% | 86.63% | -4.5 | 94.1% → 88.5% | 95.3% → 92.6% | 0.094 → 0.147 | 74.06% |
| 3 | 990 | 90.20% | 81.21% | -9.0 | 90.2% → 82.4% | 96.3% → 92.4% | 0.111 → 0.228 | 64.65% |
| 4 | 3344 | 91.06% | 82.51% | -8.6 | 91.6% → 82.7% | 97.5% → 94.9% | 0.100 → 0.205 | 68.81% |
| 5 | 4466 | 86.45% | 73.65% | -12.8 | 88.7% → 76.4% | 96.8% → 93.5% | 0.159 → 0.326 | 55.73% |
| 6 | 3740 | 80.43% | 65.29% | -15.1 | 85.2% → 69.8% | 95.9% → 92.0% | 0.246 → 0.479 | 46.82% |
| 7 | 3003 | 80.25% | 58.91% | -21.3 | 86.8% → 64.5% | 96.5% → 92.2% | 0.246 → 0.544 | 40.79% |
| 8 | 1265 | 74.15% | 49.64% | -24.5 | 80.8% → 57.9% | 95.9% → 91.3% | 0.326 → 0.696 | 32.09% |
| 9 | 704 | 73.01% | 51.85% | -21.2 | 81.8% → 59.1% | 95.7% → 91.3% | 0.388 → 0.786 | 34.09% |
| 10 | 253 | 74.70% | 40.32% | -34.4 | 83.8% → 47.8% | 96.6% → 90.9% | 0.336 → 0.909 | 26.09% |
| 11 | 132 | 68.18% | 54.55% | -13.6 | 68.2% → 55.3% | 95.8% → 94.1% | 0.462 → 0.652 | 31.06% |
| 12 | 11 | 81.82% | 18.18% | -63.6 | 81.8% → 18.2% | 98.5% → 91.7% | 0.182 → 1.000 | 0.00% |

### By word height in tokens (before padding)

| Group | Images | Exact: run 22 | Exact: run 24 | Change | Exact ignoring VS: 22 → 24 | Char acc: 22 → 24 | Edit dist: 22 → 24 | Run 24 on top-aligned: exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-7 | 243 | 91.77% | 87.65% | -4.1 | 91.8% → 87.7% | 92.7% → 88.8% | 0.086 → 0.136 | 65.84% |
| 8-15 | 4355 | 87.00% | 77.63% | -9.4 | 90.1% → 81.0% | 96.2% → 93.1% | 0.165 → 0.294 | 62.41% |
| 16-23 | 8731 | 84.15% | 69.91% | -14.2 | 88.4% → 73.9% | 96.6% → 93.2% | 0.200 → 0.401 | 51.46% |
| 24-31 | 5137 | 77.79% | 56.26% | -21.5 | 81.2% → 59.4% | 95.9% → 91.2% | 0.307 → 0.650 | 38.78% |
| 32-39 | 2126 | 66.37% | 43.18% | -23.2 | 68.8% → 44.6% | 94.8% → 89.7% | 0.486 → 0.958 | 25.92% |
| 40-47 | 828 | 48.79% | 33.09% | -15.7 | 50.4% → 34.3% | 92.7% → 88.4% | 0.842 → 1.310 | 19.08% |
| 48-55 | 290 | 24.14% | 19.66% | -4.5 | 25.9% → 19.7% | 87.5% → 84.9% | 1.814 → 2.131 | 13.45% |
| 56-63 | 105 | 9.52% | 10.48% | +1.0 | 11.4% → 10.5% | 85.0% → 84.2% | 2.343 → 2.390 | 9.52% |
| 64 | 64 | 3.12% | 1.56% | -1.6 | 3.1% → 1.6% | 75.0% → 76.0% | 5.688 → 5.609 | 1.56% |

### Anatomy of the wrong guesses

Run 22: 4627 of 21879 images read wrong. Run 24: 8030 of 21879.

| Kind of mistake | Run 22 images | Run 22 share | Run 24 images | Run 24 share |
|---|---:|---:|---:|---:|
| one letter wrong | 2630 | 56.8% | 4700 | 58.5% |
| two letters wrong | 809 | 17.5% | 1681 | 20.9% |
| only variation-selector mistakes | 750 | 16.2% | 698 | 8.7% |
| 3+ edits | 431 | 9.3% | 948 | 11.8% |
| only space (and/or VS) mistakes | 7 | 0.2% | 3 | 0.0% |

Edit operations (difflib opcode blocks): run 22 delete 2668, replace 2040, insert 1343; run 24 delete 4965, replace 4008, insert 1961.

Most common edits for run 24, with run 22's count for the same edit (expected → guessed; ∅ = nothing):

| Expected | Guessed | Run 24 | Run 22 |
|---|---|---:|---:|
| ᠠ | ∅ | 1264 | 435 |
| FVS1 | ∅ | 771 | 609 |
| ᠠ | ᠨ | 663 | 163 |
| ᡝ | ∅ | 477 | 326 |
| ∅ | ᠠ | 386 | 223 |
| ᡳ | ∅ | 381 | 266 |
| ᠨ | ∅ | 309 | 127 |
| ∅ | ᡳ | 305 | 58 |
| ᡴ | ∅ | 271 | 46 |
| ∅ | ᡝ | 244 | 60 |
| ∅ | ᠨ | 225 | 286 |
| ᠣ | ᡡ | 208 | 27 |
| FVS2 | ∅ | 155 | 208 |
| ᠵ | ᡳ | 154 | 8 |
| ᠠ | ᡝ | 136 | 42 |
| ᠨ | ᠠ | 121 | 54 |
| ᠪ | ᡦ | 115 | 4 |
| ᡤ | ᡥ | 107 | 18 |
| ᡝ | ᠠ | 104 | 83 |
| ᠸ | ᡶ | 93 | 13 |
| ᠩ | ∅ | 92 | 19 |
| ᠵ | ∅ | 91 | 24 |
| ∅ | FVS1 | 89 | 127 |
| ∅ | ᡴ | 83 | 133 |
| ᠠ | ᡴ | 80 | 31 |

Edits that grew or shrank the most (run 24 − run 22):

| Expected | Guessed | Run 22 | Run 24 | Change |
|---|---|---:|---:|---:|
| ᠠ | ∅ | 435 | 1264 | +829 |
| ᠠ | ᠨ | 163 | 663 | +500 |
| ∅ | ᡳ | 58 | 305 | +247 |
| ᡴ | ∅ | 46 | 271 | +225 |
| ∅ | ᡝ | 60 | 244 | +184 |
| ᠨ | ∅ | 127 | 309 | +182 |
| ᠣ | ᡡ | 27 | 208 | +181 |
| ∅ | ᠠ | 223 | 386 | +163 |
| FVS1 | ∅ | 609 | 771 | +162 |
| ᡝ | ∅ | 326 | 477 | +151 |
| ᠵ | ᡳ | 8 | 154 | +146 |
| ᡳ | ∅ | 266 | 381 | +115 |
| ᠪ | ᡦ | 4 | 115 | +111 |
| ᠠ | ᡝ | 42 | 136 | +94 |

### How many fonts each lexigraph is misread in

| Fonts misread (of 11) | Run 22 lexigraphs | Run 24 lexigraphs |
|---:|---:|---:|
| 0 | 697 | 298 |
| 1 | 453 | 307 |
| 2 | 221 | 270 |
| 3 | 161 | 211 |
| 4 | 103 | 177 |
| 5 | 65 | 133 |
| 6 | 54 | 92 |
| 7 | 36 | 93 |
| 8 | 43 | 86 |
| 9 | 39 | 85 |
| 10 | 46 | 86 |
| 11 | 71 | 151 |

193 lexigraphs are read right in every font by both models.

Hardest 20 lexigraphs for run 24:

| Lexigraph | Length | Fonts right: 24 | Fonts right: 22 | Run 24 mean char acc | Example run 24 guess |
|---|---:|---:|---:|---:|---|
| ᡶᡳᠴᠠᡴᡡ | 6 | 0/11 | 0/11 | 33.3% | ᡡ |
| ᠣᡴ᠋ᡨ᠋ᠣ | 6 | 0/11 | 0/11 | 53.0% | ᠣᡴᠣᠣ |
| ᡩᠣᡵᡤᡳ ᠪᠠᡳᡨᠠ ᠪᡝ ᡠᡥᡝᡵᡳ ᡴᠠᡩᠠᠯᠠᡵᠠ ᠶᠠᠮᡠᠨ | 35 | 0/11 | 0/11 | 54.5% | ᡩᡡᡵᡤᡝ ᡝᡳᡨᠨ ᠪᡝ ᡝᡠᡝᡵᡳ ᡩᠯᠠᡵ ᠶᠮᡠᠨ |
| ᡩ᠋ᡠ᠋ᡳ᠌ᠨ | 7 | 0/11 | 1/11 | 57.1% | ᡩᡠᡳᠨ |
| ᠵᠠᡴᠠ | 4 | 0/11 | 9/11 | 59.1% | ᠵᠠᡴᠨ |
| ᡨᠠᡴᠠ | 4 | 0/11 | 4/11 | 59.1% | ᡨᠠᡴᠨ |
| ᡨᠣᠨ᠋ᡩ᠋ᠣ | 7 | 0/11 | 0/11 | 59.7% | ᡨᠣᡩᠣ |
| ᠠᠮᠠᡴᠠ | 5 | 0/11 | 3/11 | 60.0% | ᠠᠮᡴ |
| ᡰᡳᠨᡯᡠᠩ | 6 | 0/11 | 0/11 | 60.6% | ᡳᠠᡯᡠᠩ |
| ᡶᠠᡨ᠋ᠠᠨ | 6 | 0/11 | 5/11 | 60.6% | ᡶᠠᠣᠠᠨ |
| ᠠᡳᡴᠠ | 4 | 0/11 | 6/11 | 61.4% | ᠠᡳᡴᠨ |
| ᠸᠠᡴᠠ | 4 | 0/11 | 9/11 | 61.4% | ᠸᠠᡴᠨ |
| ᠰᡠᠸᠠᠨ᠋ᡩ᠋ᠠ | 9 | 0/11 | 0/11 | 63.6% | ᠰᡠᠸᠠᠨᡩᠠ |
| ᡩᡠᡴᠠ | 4 | 0/11 | 10/11 | 63.6% | ᡩᡠᡴᠨ |
| ᡩᡠᡴᠠᡳ ᡝᠨᡩᡠᡵᡳ | 12 | 0/11 | 0/11 | 64.4% | ᡩᡠᡳ ᡩᡠᡵᡳ |
| ᡥᠣᠩ ᡨᠠᡳᠵᡳ | 9 | 0/11 | 2/11 | 64.6% | ᡥᡡ ᡨᠠᡳᡳ |
| ᠭ᠋ᠤᠰᠢᠨ | 6 | 0/11 | 0/11 | 65.2% | ᠭᠤᠰᠢᠨ |
| ᠵᡠᡵᠠᡴᠠ | 6 | 0/11 | 2/11 | 65.2% | ᠵᡠᡵᡴᠨ |
| ᡶ᠋ᡠᡨ᠋ᠠ | 6 | 0/11 | 1/11 | 65.2% | ᡶ᠋ᡠᡨᠨ |
| ᠪᡝ ᠯᡳᡳ ᡥᡳ ᡨ᠋ᡳᠶᠠᠨ ᡩ᠋ᡝ᠋ | 21 | 0/11 | 0/11 | 65.4% | ᠪᡝ ᡝᡳ ᡥᡳ ᡨᡳᠶᡳᡝ ᡩᡝ |

Lexigraphs with the biggest drop (fonts read exactly, run 22 → run 24):

| Lexigraph | Length | Run 22 | Run 24 |
|---|---:|---:|---:|
| ᠨᡳᡴᠠᠨ ᡤᡳᠰᡠᠨ | 11 | 11/11 | 0/11 |
| ᠮᠠᡶᠠ ᡤᡠᡵᡠᠨ | 10 | 11/11 | 0/11 |
| ᡤᡡᠰᡳᠨ ᠵᡠᠸᡝ | 10 | 11/11 | 0/11 |
| ᠪᡠᠯᠠ | 4 | 11/11 | 1/11 |
| ᠪᡳᠶᠠ ᠵᡝᠮᠪᡳ | 10 | 11/11 | 1/11 |
| ᠰᡠᠨᠵᠠᠩᡤᡝᡵᡳ | 10 | 10/11 | 0/11 |
| ᡠᠶᠠᠨ ᠯᠠᠯᠠ | 9 | 10/11 | 0/11 |
| ᡤᡳᡵᠠᠩᡤᡳ | 7 | 11/11 | 1/11 |
| ᡩᡠᡴᠠ | 4 | 10/11 | 0/11 |
| ᡶᡝᡨᡝ᠋ᠮᠪᡳ | 8 | 10/11 | 0/11 |
| ᠨᡠᡵᡤᠠᠴᡳ | 7 | 10/11 | 1/11 |
| ᠪᡠᠯᠠ ᠰᠣᡤᡳ | 9 | 9/11 | 0/11 |

Run 24 reads 303 lexigraphs in more fonts than run 22 and 1269 in fewer.

Image by image: both right 12409, only run 22 right 4843, only run 24 right 1440, both wrong 3187.

### Single vs multiple word lexigraphs at matched lengths (exact match)

| Label length | Single: run 22 | Single: run 24 | Multiple: run 22 | Multiple: run 24 |
|---:|---:|---:|---:|---:|
| 4 | 91.1% (3344) | 82.5% | 72.7% (11) | 81.8% |
| 6 | 80.4% (3740) | 65.3% | 67.5% (77) | 55.8% |
| 7 | 80.3% (3003) | 58.9% | 77.6% (143) | 57.3% |
| 8 | 74.2% (1265) | 49.6% | 67.3% (165) | 33.9% |
| 9 | 73.0% (704) | 51.8% | 65.0% (605) | 41.5% |
| 10 | 74.7% (253) | 40.3% | 61.1% (517) | 32.3% |
| 11 | 68.2% (132) | 54.5% | 55.7% (539) | 36.7% |
| 12 | 81.8% (11) | 18.2% | 44.5% (418) | 24.2% |
| 13+ | — | — | 22.5% (781) | 12.9% |

## Position in the image

Run 24, padded data: exact match by where the word starts (rows) and how tall it is, so length and position aren't confounded (cells with < 30 images left out).

| Word height (tokens) | starts tok 0-7 | starts tok 8-15 | starts tok 16-23 | starts tok 24-31 | starts tok 32-39 | starts tok 40-47 | starts tok 48-55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-15 | 77.2% | 75.8% | 80.8% | 79.5% | 78.3% | 78.1% | 76.6% |
| 16-23 | 67.4% | 69.9% | 71.1% | 69.0% | 71.3% | 71.6% | — |
| 24-31 | 54.7% | 56.5% | 58.5% | 53.9% | 58.8% | — | — |
| 32-47 | 35.8% | 41.7% | 43.7% | 40.2% | — | — | — |

Same comparison for run 22 on the padded data (it never saw padding):

| Word height (tokens) | starts tok 0-7 | starts tok 8-15 | starts tok 16-23 | starts tok 24-31 | starts tok 32-39 | starts tok 40-47 | starts tok 48-55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-15 | 33.4% | 1.0% | 0.1% | 0.1% | 0.1% | 0.0% | 0.0% |
| 16-23 | 27.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — |
| 24-31 | 20.2% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| 32-47 | 14.3% | 0.0% | 0.0% | 0.0% | — | — | — |

Run 24, padded data, by exact number of blank rows above the word, with the same images top-aligned for comparison:

| Blank rows above | Images | Run 24 padded | Run 24 same images top-aligned | Mean word height (tokens) |
|---|---:|---:|---:|---:|
| 0-0 | 197 | 37.1% | 37.1% | 36.9 |
| 1-3 | 423 | 54.1% | 41.4% | 28.2 |
| 4-7 | 626 | 55.6% | 39.8% | 27.4 |
| 8-15 | 1165 | 55.8% | 42.8% | 25.8 |
| 16-31 | 2235 | 57.8% | 42.6% | 25.4 |
| 32-63 | 4372 | 60.7% | 43.3% | 24.6 |
| 64-+ | 12861 | 66.9% | 48.9% | 20.9 |

**Shift test.** 2,000 random short single words (≤ 24 tokens), top-aligned and then moved down by prepending blank tokens. Leading spaces in the guess are ignored when scoring.

| Shift (tokens) | 0 | 4 | 8 | 16 | 24 | 32 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run 22 exact | 86.1% | 59.9% | 44.2% | 30.0% | 19.7% | 10.4% | 4.6% |
| Run 22 guess starts with a space | 0.0% | 87.2% | 96.2% | 99.3% | 99.4% | 99.9% | 100.0% |
| Run 24 exact | 54.6% | 66.8% | 67.0% | 68.8% | 67.3% | 67.4% | 67.3% |
| Run 24 guess starts with a space | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

The same word, in the same font, read as part of a phrase vs on its own (VS stripped, phrases where the number of words was right):

| Word's position in phrase | Run 22 in phrase | Run 22 standalone | Run 24 in phrase | Run 24 standalone |
|---|---:|---:|---:|---:|
| 1st | 85.7% | 91.8% | 57.1% | 79.9% |
| 2nd | 62.1% | 89.1% | 54.2% | 77.3% |
| 3rd+ | 31.0% | 97.4% | 44.4% | 84.4% |

Word count right: run 22 99.9%, run 24 99.9%.

### Length: per-character error of single words (VS stripped)

| Length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 22 per-char error | 3.4% | 3.8% | 2.5% | 2.8% | 3.0% | 2.2% | 3.3% | 2.3% | 1.9% | 4.4% | 1.5% |
| Run 22 exact (no VS) | 93.8% | 90.0% | 90.9% | 88.2% | 85.8% | 86.9% | 79.1% | 83.0% | 84.0% | 66.1% | 81.8% |
| Run 24 per-char error | 6.2% | 7.3% | 5.1% | 6.0% | 6.7% | 6.5% | 7.8% | 5.9% | 7.7% | 6.1% | 8.3% |
| Run 24 exact (no VS) | 88.6% | 82.0% | 82.5% | 75.9% | 69.7% | 64.1% | 54.3% | 61.1% | 48.1% | 52.9% | 18.2% |

Overall per-character error for single words: run 22 2.7%, run 24 6.4%.

### Letters: share of each letter's occurrences misread

| Letter | Occurrences in lexicon | Run 22 | Run 24 | Change |
|---|---:|---:|---:|---:|
| FVS3 | 2 | 77.3% | 86.4% | +9.1 |
| FVS2 | 27 | 83.2% | 80.5% | -2.7 |
| ᠇ | 1 | 9.1% | 72.7% | +63.6 |
| FVS1 | 141 | 47.7% | 67.7% | +20.0 |
| ᠭ | 1 | 100.0% | 54.5% | -45.5 |
| ᡮ | 2 | 9.1% | 50.0% | +40.9 |
| ᡰ | 3 | 15.2% | 42.4% | +27.3 |
| ᠳ | 1 | 0.0% | 36.4% | +36.4 |
| ᡬ | 7 | 26.0% | 31.2% | +5.2 |
| ᡦ | 21 | 56.7% | 28.1% | -28.6 |
| ᠊ | 21 | 16.9% | 24.2% | +7.4 |
| ᡟ | 2 | 18.2% | 18.2% | +0.0 |
| ᡱ | 1 | 0.0% | 18.2% | +18.2 |
| ᡴ | 303 | 5.0% | 17.9% | +13.0 |
| ᠠ | 1538 | 5.6% | 16.7% | +11.1 |
| ᠤ | 4 | 0.0% | 15.9% | +15.9 |
| ᡯ | 12 | 18.9% | 15.2% | -3.8 |
| ᠵ | 264 | 1.8% | 14.7% | +12.9 |
| ᠸ | 132 | 3.7% | 13.7% | +10.1 |
| ᠺ | 1 | 18.2% | 9.1% | -9.1 |
| ᡷ | 4 | 0.0% | 9.1% | +9.1 |
| ᠩ | 228 | 1.8% | 8.1% | +6.3 |
| ᡝ | 981 | 5.8% | 8.1% | +2.3 |
| ᡳ | 1376 | 3.4% | 4.7% | +1.3 |
| ᠨ | 987 | 2.9% | 5.8% | +3.0 |
| ᠪ | 506 | 1.4% | 5.3% | +3.8 |

## Variation selectors

Visibility comes from run 22's `variation_selector_visibility.csv`. Padding doesn't change how glyphs are drawn, so it still holds.

| | Images | Run 22 exact | Run 22 exact ignoring VS | Run 24 exact | Run 24 exact ignoring VS |
|---|---:|---:|---:|---:|---:|
| Selector invisible | 882 | 20.7% | 81.5% | 13.4% | 64.9% |
| Selector visible | 614 | 56.7% | 73.9% | 35.0% | 61.9% |

Selector recall (share of a selector's occurrences that the model got right, from the letter table above):

- FVS1: run 22 52.3%, run 24 32.3% (1551 occurrences across all images)
- FVS2: run 22 16.8%, run 24 19.5% (297 occurrences across all images)
- FVS3: run 22 22.7%, run 24 13.6% (22 occurrences across all images)

Selectors the model wrote anywhere: run 22 1024, run 24 716 (labels contain 1870).

The two lexigraphs whose FVS2 is visible, plus the run 22 examples:

| Lexigraph | Run 22 fonts exact | Run 24 fonts exact |
|---|---:|---:|
| ᠨᠣᠮᡠᠨ ᡥᠠᠨ[2] | 0/11 | 0/11 |
| ᠮᡝᡳ[2]ᡨᡝ[1]ᠮᠪᡳ | 0/11 | 0/11 |
| ᡶ[1]ᠣᡵᠣᠨ | 11/11 | 8/11 |
| ᡥᠣᡨ[1]ᠣᠨ | 5/11 | 0/11 |
| ᠣᡴ[1]ᡨ[1]ᠣ | 0/11 | 0/11 |
| ᡨ[1]ᡳᠶᠠᠨ ᠵᡳᠨ | 2/11 | 0/11 |
| ᠪᡝᡳ[2]ᠰᡝ | 0/11 | 2/11 |
| ᡨᡠ[1]ᡵᠠᡴᡳ | 0/11 | 0/11 |
| ᠨᠠᡩ[1]ᠠᠨ | 1/11 | 2/11 |
| ᡶᡝᡨᡝ[1]ᠮᠪᡳ | 10/11 | 0/11 |
