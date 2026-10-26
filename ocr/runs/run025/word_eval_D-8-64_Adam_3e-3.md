# Word-level evaluation: D-8/64 Adam 3e-3, runs 22, 24 and 25

Weights: `weights_D-8-64_Adam_3e-3.npz` from each run (final epoch). Evaluated 2026-10-25 in eval mode (no dropout), greedy CTC decoding, on **every lexigraph × every font**: 1,989 lexigraphs × 11 fonts = 21,879 images.

**What differs between the runs.** Same model, optimizer and learning rate. The runs differ in their data and their length:

| Run | Training data | Epochs |
|---|---|---:|
| 22 | **top-aligned**: the word always starts at row 0 | 30 |
| 24 | **padded**: every image gets a seeded random number of blank rows above the word (0 to 256 − height) | 30 |
| 25 | **mixed**: run 24's padding, but about a third of images (33.5%) get none; the rest keep exactly their run 24 padding | 45 |

Every model was evaluated on all three datasets. The reruns of run 22 on top-aligned data and run 24 on padded data are byte-for-byte identical to their earlier evaluations. Run 25 trained 15 epochs longer, so the run 24 → 25 comparison mixes the data change with the extra training (see finding 6).

Files:
- `word_eval_D-8-64_Adam_3e-3.csv`: run 25 per image on its own (mixed) data. It has the same columns as run 24's file; `ink_start_row` is the number of blank rows above the word.
- `compare_run022_run024_run025.csv`: one row per image, with every model's guess and exact flag on every dataset.
- `comparison_run022_run024_run025.png`: training curves, shift test, exact match by length and by font.

**Caveat, as before: train and test are mixed.** The split is random, isn't saved, and differs between runs, so ~70% of these images were trained on. Train and test character accuracy are about equal in all three logs.

## Key findings

1. **Run 25 is the best model so far: 79.2% exact on its own data, 95.9% character accuracy.** It's also the first that does well everywhere: 76.9% on top-aligned images and 76.7% on fully padded ones. Run 22 manages 78.9% top-aligned but only 4.8% padded; run 24 manages 46.3% and 63.3%.
2. **On run 22's own task (top-aligned) run 25 is 2 points behind** (76.9% vs 78.9%; char accuracy 95.2% vs 95.8%). On their own datasets the two are about even: 2,256 images are read right only by run 22 and 2,342 only by run 25.
3. **Position no longer matters, including the top edge.** Shift test: run 25 gets 82.3% with no shift and 79–81% at every shift up to 40 tokens. Run 22 falls from 86% to 5%; run 24 has a hole at the top (54.6% → 67%). Words that start at row 0 even on padded data (long words with no room) go from 37.1% for run 24 to 57.9%. On its own data, run 25 reads unpadded images slightly *better* than padded ones (80.5% vs 78.9%).
4. **Long words, phrases and tall images improved past run 22.** Multi-word lexigraphs: 56.9% vs 50.8%. 13–15 characters: 39.3% vs 30.4%; 16+: 13.4% vs 3.9%. Words 48–55 tokens tall: 42.8% vs 24.1%. Squashed >256 px images: 13.0% vs 3.7%. In phrases, the 2nd word is read right 75.0% of the time (run 22: 62.1%), and 3rd+ words 54.3% (31.0%). These are exactly the late token positions run 22 rarely trained on.
5. **Short and mid-length single words are slightly worse than run 22.** 1–2 characters: 84–87% vs 88–91%. Single words of 9–10 characters: 69.3% / 65.2% vs 73.0% / 74.7%. Overall, single words are about equal (83.2% vs 83.8%), and per-character error is 2.8% vs 2.7%.
6. **The extra epochs did most of the catching up, and run 25 is still learning.** Its curve tracks run 24's almost exactly: 0.892 test character accuracy at epoch 30, against run 24's 0.891. It then gained 3.5 points in epochs 31–45, finishing at 0.927 and still rising (0.921, 0.918, 0.920, 0.922, 0.927 over its last 5 epochs). It never reached run 22's final 0.939, though those test sets aren't the same images. So the unpadded third mainly fixed the top edge; it didn't speed up learning.
7. **Variation selectors improved the most.** Run 25 reads FVS1 correctly 57.6% of the time (run 22: 52.3%) and FVS2 42.4% (run 22: 16.8%). Lexigraphs with a selector are 44.5% exact (run 22: 35.5%); visible selectors 61.7%, invisible ones 32.4%. Since there is nothing in the image to read for invisible selectors, that last number is memorisation. It also writes selectors that aren't there more often (FVS1 inserted 181× vs 127×; FVS2 88× vs 16×).
8. **The errors are the same kinds as run 22's, with a few shifts.** Run 24's tooth-stroke blow-up is gone: ᠠ dropped 351× (run 24: 1,264, run 22: 435), ᠠ→ᠨ 165× (663 / 163). New or grown: spurious ᠠ (453× vs 223×), spurious ᡝ (199× vs 60×), ᠯ→ᡥ (62× vs 5×), and spurious ᠊ (56× vs 3×).
9. **Fonts** keep the same order. LiuYe improved most over run 22 (+4.4 points) and GuFeng and ShuKai gained 2–3 points; WenJian (−2.9) and YingBi (−2.2) slipped. 643 lexigraphs are right in all 11 fonts (run 22: 697), and only 45 are wrong in all 11 (run 22: 71).

**Suggestions:** keep the mixed padding; it's the first data setup without a positional blind spot. Train longer or resume from run 25's weights, since accuracy was still rising at epoch 45. The gap to run 22 on short top-aligned words is small and might close with more epochs. To separate the effect of the data from the extra training, a 30-epoch run on the mixed data could be compared with run 24 directly.

## Tables

### Overall: each model on each dataset

Each model's own training data is in bold.

| Model | Evaluated on | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---|---:|---:|---:|---:|
| Run 22 | **top-aligned** | **78.85%** | 82.28% | 95.77% | 0.317 |
| Run 22 | padded (every image) | 4.81% | 5.01% | 50.60% | 2.963 |
| Run 22 | mixed (a third unpadded) | 29.04% | 30.32% | 65.40% | 2.095 |
| Run 24 | top-aligned | 46.26% | 49.00% | 86.13% | 0.928 |
| Run 24 | **padded (every image)** | **63.30%** | 66.49% | 91.95% | 0.572 |
| Run 24 | mixed (a third unpadded) | 57.99% | 61.02% | 90.13% | 0.684 |
| Run 25 | top-aligned | 76.93% | 80.74% | 95.20% | 0.330 |
| Run 25 | padded (every image) | 76.70% | 80.30% | 95.17% | 0.332 |
| Run 25 | **mixed (a third unpadded)** | **79.24%** | 82.74% | 95.85% | 0.293 |

### Training logs (mean over each epoch's batches; train has dropout on)

| Epoch | Run 22 train | Run 22 test | Run 24 train | Run 24 test | Run 25 train | Run 25 test | Run 25 test loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.034 | 0.037 | 0.008 | 0.001 | 0.018 | 0.012 | 4.612 |
| 2 | 0.171 | 0.233 | 0.005 | 0.010 | 0.058 | 0.082 | 2.995 |
| 3 | 0.350 | 0.421 | 0.052 | 0.067 | 0.170 | 0.219 | 2.494 |
| 5 | 0.570 | 0.629 | 0.317 | 0.372 | 0.398 | 0.433 | 1.777 |
| 10 | 0.791 | 0.815 | 0.639 | 0.673 | 0.667 | 0.693 | 0.922 |
| 15 | 0.868 | 0.886 | 0.764 | 0.779 | 0.778 | 0.793 | 0.646 |
| 20 | 0.903 | 0.912 | 0.824 | 0.837 | 0.836 | 0.840 | 0.506 |
| 25 | 0.918 | 0.931 | 0.867 | 0.869 | 0.867 | 0.879 | 0.404 |
| 30 | 0.934 | 0.939 | 0.888 | 0.891 | 0.890 | 0.892 | 0.360 |
| 35 | — | — | — | — | 0.908 | 0.908 | 0.311 |
| 40 | — | — | — | — | 0.916 | 0.912 | 0.309 |
| 43 | — | — | — | — | 0.924 | 0.920 | 0.284 |
| 44 | — | — | — | — | 0.926 | 0.922 | 0.271 |
| 45 | — | — | — | — | 0.925 | 0.927 | 0.265 |

Run 25's test accuracy over its last 5 epochs: 0.921, 0.918, 0.920, 0.922, 0.927.
Run 25 never reached run 22's final-epoch test accuracy (0.939); its best epoch was 45 (0.927).
Run 24's final-epoch test accuracy (0.891) was first reached by run 25 at epoch 29.

In the grouped tables, "Run 22/24/25" = exact match of each model on its own training data; the top-aligned columns show runs 24 and 25 on run 22's data.

### By font

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XM_BiaoHei | 1989 | 82.9% | 66.9% | 81.0% | 51.8% | 79.7% | 85.2% | 96.6% |
| XM_GuFeng | 1989 | 80.7% | 61.4% | 83.0% | 48.0% | 80.4% | 86.1% | 96.8% |
| XM_LiuYe | 1989 | 66.9% | 51.4% | 71.3% | 32.6% | 67.9% | 74.5% | 94.3% |
| XM_ShuKai | 1989 | 74.2% | 57.6% | 77.0% | 36.6% | 73.2% | 80.1% | 95.1% |
| XM_WenJian | 1989 | 72.3% | 52.0% | 69.4% | 39.0% | 65.7% | 72.3% | 92.5% |
| XM_WenQin | 1989 | 79.6% | 66.5% | 80.4% | 49.6% | 79.7% | 83.7% | 96.5% |
| XM_XingShu | 1989 | 77.5% | 65.5% | 79.3% | 47.1% | 78.3% | 83.3% | 96.0% |
| XM_YaBai | 1989 | 84.4% | 70.8% | 83.2% | 57.5% | 81.3% | 87.4% | 96.7% |
| XM_YingBi | 1989 | 79.5% | 65.2% | 77.3% | 42.7% | 76.1% | 80.8% | 95.2% |
| XM_ZhengBai | 1989 | 83.8% | 69.9% | 85.0% | 51.8% | 81.9% | 88.7% | 97.3% |
| XM_ZhengHei | 1989 | 85.6% | 69.2% | 84.6% | 52.2% | 82.3% | 88.1% | 97.3% |

### Variation selectors

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With variation selector(s) | 1496 | 35.5% | 22.3% | 44.5% | 8.4% | 39.9% | 80.1% | 89.3% |
| Without | 20383 | 82.0% | 66.3% | 81.8% | 49.0% | 79.6% | 82.9% | 96.3% |

### By label length (characters, incl. spaces & variation selectors)

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 341 | 88.3% | 83.6% | 84.2% | 64.8% | 81.2% | 84.2% | 84.2% |
| 2 | 374 | 91.2% | 86.6% | 87.2% | 74.1% | 82.9% | 89.8% | 93.0% |
| 3 | 990 | 90.2% | 81.2% | 91.9% | 64.6% | 88.6% | 92.5% | 96.6% |
| 4 | 3355 | 91.0% | 82.5% | 91.6% | 68.9% | 90.3% | 92.5% | 97.5% |
| 5 | 4466 | 86.5% | 73.6% | 85.3% | 55.7% | 82.7% | 88.4% | 96.6% |
| 6 | 3817 | 80.2% | 65.1% | 80.8% | 46.6% | 79.0% | 85.5% | 96.2% |
| 7 | 3146 | 80.1% | 58.8% | 79.4% | 40.1% | 76.2% | 84.1% | 96.3% |
| 8 | 1430 | 73.4% | 47.8% | 74.1% | 30.8% | 72.4% | 81.4% | 95.9% |
| 9 | 1309 | 69.3% | 47.1% | 70.7% | 26.3% | 68.2% | 76.1% | 95.7% |
| 10 | 770 | 65.6% | 34.9% | 64.7% | 18.4% | 63.4% | 70.6% | 95.5% |
| 11 | 671 | 58.1% | 40.2% | 60.4% | 18.8% | 59.8% | 61.5% | 94.5% |
| 12 | 429 | 45.5% | 24.0% | 49.2% | 9.6% | 46.9% | 50.3% | 93.0% |
| 13-15 | 550 | 30.4% | 16.7% | 39.3% | 7.5% | 34.4% | 42.5% | 91.1% |
| 16+ | 231 | 3.9% | 3.9% | 13.4% | 2.6% | 10.4% | 16.9% | 87.2% |

### Single vs multiple word lexigraphs

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single word | 18623 | 83.8% | 69.0% | 83.2% | 51.9% | 80.9% | 86.9% | 96.2% |
| Multiple words | 3256 | 50.8% | 31.0% | 56.9% | 14.3% | 54.1% | 59.0% | 93.8% |

### Image squashed to fit 256 px (original height > 256)

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fits | 21825 | 79.0% | 63.5% | 79.4% | 46.4% | 77.1% | 82.9% | 95.9% |
| Squashed | 54 | 3.7% | 1.9% | 13.0% | 1.9% | 13.0% | 13.0% | 83.2% |

### By number of words

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 words | 18623 | 83.8% | 69.0% | 83.2% | 51.9% | 80.9% | 86.9% | 96.2% |
| 2 words | 2981 | 54.4% | 33.2% | 60.5% | 15.1% | 57.9% | 62.7% | 94.4% |
| 3 words | 209 | 11.5% | 8.6% | 21.1% | 6.2% | 15.8% | 23.0% | 90.0% |
| 4 words | 44 | 18.2% | 2.3% | 11.4% | 4.5% | 9.1% | 11.4% | 84.4% |
| 5 words | 11 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 74.0% |
| 6 words | 11 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 57.7% |

### Single-word lexigraphs by length

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 341 | 88.3% | 83.6% | 84.2% | 64.8% | 81.2% | 84.2% | 84.2% |
| 2 | 374 | 91.2% | 86.6% | 87.2% | 74.1% | 82.9% | 89.8% | 93.0% |
| 3 | 990 | 90.2% | 81.2% | 91.9% | 64.6% | 88.6% | 92.5% | 96.6% |
| 4 | 3344 | 91.1% | 82.5% | 91.6% | 68.8% | 90.3% | 92.5% | 97.5% |
| 5 | 4466 | 86.5% | 73.6% | 85.3% | 55.7% | 82.7% | 88.4% | 96.6% |
| 6 | 3740 | 80.4% | 65.3% | 80.7% | 46.8% | 79.0% | 85.5% | 96.2% |
| 7 | 3003 | 80.3% | 58.9% | 79.4% | 40.8% | 76.5% | 84.2% | 96.3% |
| 8 | 1265 | 74.2% | 49.6% | 73.6% | 32.1% | 71.8% | 81.5% | 95.9% |
| 9 | 704 | 73.0% | 51.8% | 69.3% | 34.1% | 68.3% | 79.1% | 95.6% |
| 10 | 253 | 74.7% | 40.3% | 65.2% | 26.1% | 60.5% | 73.9% | 95.5% |
| 11 | 132 | 68.2% | 54.5% | 72.0% | 31.1% | 68.2% | 72.0% | 96.6% |
| 12 | 11 | 81.8% | 18.2% | 54.5% | 0.0% | 54.5% | 54.5% | 95.5% |

### By word height in tokens (before padding)

| Group | Images | Run 22 | Run 24 | Run 25 | Run 24 top-aligned | Run 25 top-aligned | Run 25 exact ignoring VS | Run 25 char acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-7 | 243 | 91.8% | 87.7% | 87.7% | 65.8% | 84.0% | 87.7% | 89.3% |
| 8-15 | 4355 | 87.0% | 77.6% | 87.7% | 62.4% | 85.0% | 90.9% | 96.0% |
| 16-23 | 8731 | 84.1% | 69.9% | 84.2% | 51.5% | 81.4% | 88.4% | 96.7% |
| 24-31 | 5137 | 77.8% | 56.3% | 76.5% | 38.8% | 75.3% | 80.0% | 95.8% |
| 32-39 | 2126 | 66.4% | 43.2% | 66.5% | 25.9% | 64.1% | 69.1% | 94.9% |
| 40-47 | 828 | 48.8% | 33.1% | 54.7% | 19.1% | 52.2% | 56.6% | 93.9% |
| 48-55 | 290 | 24.1% | 19.7% | 42.8% | 13.4% | 43.1% | 43.8% | 92.4% |
| 56-63 | 105 | 9.5% | 10.5% | 24.8% | 9.5% | 25.7% | 27.6% | 91.3% |
| 64 | 64 | 3.1% | 1.6% | 10.9% | 1.6% | 10.9% | 10.9% | 83.3% |

### Anatomy of the wrong guesses (each model on its own data)

| Kind of mistake | Run 22 images | Run 22 share | Run 24 images | Run 24 share | Run 25 images | Run 25 share |
|---|---:|---:|---:|---:|---:|---:|
| one letter wrong | 2630 | 56.8% | 4700 | 58.5% | 2727 | 60.1% |
| two letters wrong | 809 | 17.5% | 1681 | 20.9% | 694 | 15.3% |
| only variation-selector mistakes | 750 | 16.2% | 698 | 8.7% | 765 | 16.8% |
| 3+ edits | 431 | 9.3% | 948 | 11.8% | 349 | 7.7% |
| only space (and/or VS) mistakes | 7 | 0.2% | 3 | 0.0% | 6 | 0.1% |
| **total wrong** | 4627 |  | 8030 |  | 4541 |  |

Edit operations (difflib opcode blocks): run 22 delete 2668, replace 2040, insert 1343; run 24 delete 4965, replace 4008, insert 1961; run 25 delete 2171, replace 1867, insert 1719.

Most common edits for run 25, with the other runs' counts (expected → guessed; ∅ = nothing):

| Expected | Guessed | Run 22 | Run 24 | Run 25 |
|---|---|---:|---:|---:|
| FVS1 | ∅ | 609 | 771 | 575 |
| ∅ | ᠠ | 223 | 386 | 453 |
| ᠠ | ∅ | 435 | 1264 | 351 |
| ∅ | ᡝ | 60 | 244 | 199 |
| ᠨ | ∅ | 127 | 309 | 191 |
| ᡳ | ∅ | 266 | 381 | 190 |
| ∅ | FVS1 | 127 | 89 | 181 |
| ∅ | ᡳ | 58 | 305 | 168 |
| ᠠ | ᠨ | 163 | 663 | 165 |
| ᡝ | ∅ | 326 | 477 | 142 |
| ∅ | ᡴ | 133 | 83 | 134 |
| FVS2 | ∅ | 208 | 155 | 129 |
| ᡴ | ∅ | 46 | 271 | 122 |
| ᠨ | ᠠ | 54 | 121 | 108 |
| ∅ | ᠨ | 286 | 225 | 97 |
| ᠠ | ᡝ | 42 | 136 | 88 |
| ∅ | FVS2 | 16 | 32 | 88 |
| ᡝ | ᠠ | 83 | 104 | 69 |
| ᡦ | ᠪ | 105 | 29 | 65 |
| ᠯ | ᡥ | 5 | 35 | 62 |
| ᡤ | ᡥ | 18 | 107 | 61 |
| ∅ | ᠊ | 3 | 10 | 56 |
| ᡵ | ∅ | 42 | 62 | 55 |
| ᡵ | ᡳ | 8 | 17 | 46 |
| ∅ | ᠵ | 33 | 25 | 45 |

### How many fonts each lexigraph is misread in

| Fonts misread (of 11) | Run 22 | Run 24 | Run 25 |
|---:|---:|---:|---:|
| 0 | 697 | 298 | 643 |
| 1 | 453 | 307 | 436 |
| 2 | 221 | 270 | 274 |
| 3 | 161 | 211 | 174 |
| 4 | 103 | 177 | 113 |
| 5 | 65 | 133 | 87 |
| 6 | 54 | 92 | 66 |
| 7 | 36 | 93 | 43 |
| 8 | 43 | 86 | 45 |
| 9 | 39 | 85 | 34 |
| 10 | 46 | 86 | 29 |
| 11 | 71 | 151 | 45 |

Hardest 20 lexigraphs for run 25:

| Lexigraph | Length | Run 22 fonts right | Run 24 | Run 25 | Run 25 mean char acc | Example run 25 guess |
|---|---:|---:|---:|---:|---:|---|
| ᡩᠣᡵᡤᡳ ᠪᠠᡳᡨᠠ ᠪᡝ ᡠᡥᡝᡵᡳ ᡴᠠᡩᠠᠯᠠᡵᠠ ᠶᠠᠮᡠᠨ | 35 | 0/11 | 0/11 | 0/11 | 57.7% | ᡩᠣᡳᡥᡳ ᠪᡝᡳᡨᠠ ᠪᡝ ᡝᡠᡥᡝᡳᡤ ᡤᠠᠵᡩᠯᡳ ᠶᠮᡠᡳᠠ |
| ᡩᠠᡳ᠌ᠴᡳᠩ | 7 | 1/11 | 0/11 | 0/11 | 64.9% | ᡩᠠᡳᠴᡳᠩ |
| ᠭ᠋ᠤᠰᠢᠨ | 6 | 0/11 | 0/11 | 0/11 | 66.7% | ᠤᠰᠢᠨ |
| ᡩᡠᡴᠠᡳ ᡝᠨᡩᡠᡵᡳ | 12 | 0/11 | 0/11 | 0/11 | 71.2% | ᡩᡠᡳ ᠠᡩᡠᡵᡳ |
| ᠶᠠᠪᡠᠪᡠᡶᡳ ᡳᠯᡤᡝ᠋ | 14 | 0/11 | 0/11 | 0/11 | 71.4% | ᠶᠠᠪᡠᠪᡠᠸᡳ ᡳᠯᡤᡝ |
| ᡨᠣᠨ᠋ᡩ᠋ᠣ | 7 | 0/11 | 0/11 | 0/11 | 71.4% | ᡨᠣᠨᡩᠣ |
| ᡶᡳᠴᠠᡴᡡ | 6 | 0/11 | 0/11 | 0/11 | 72.7% | ᡶᡳᡴᡡ |
| ᠪᡝ ᠯᡳᡳ ᡥᡳ ᡨ᠋ᡳᠶᠠᠨ ᡩ᠋ᡝ᠋ | 21 | 0/11 | 0/11 | 0/11 | 74.0% | ᠪ ᠯᡝᡳ ᡥᡳᡝ ᡩ᠋ᡳᠶᠠ᠋ ᡩ᠋ |
| ᡩᠠᡳ᠌ᠴᡳᠩ ᡤᡠᡵᡠᠨ | 13 | 2/11 | 0/11 | 0/11 | 74.1% | ᡩᠠᡶᠴᡳᠩ ᡤᡠᠵᡠᠨ |
| ᠠᡳ᠌᠇ᡠ᠌ᡥᠠᠨ | 9 | 0/11 | 0/11 | 0/11 | 74.7% | ᠠᡳ᠇ᡠᡥᠠᠨ |
| ᡝᠨᡩᡝᠮᠪᡳ | 7 | 0/11 | 0/11 | 0/11 | 76.6% | ᠠᡩᡝᠮᠪᡳ |
| ᠨᠠᡵᡥᡡᠨ ᡝᡩᡠᠨ ᡨᡠᠸᠠᡵᠠ ᠨᡳᠮᡝᡴᡠ | 25 | 0/11 | 0/11 | 0/11 | 76.7% | ᠨᠠᡵᡥᡡᠨ ᡩᡠᠨ ᡩᡠᡶᠠᡵᠨ ᠨᡳᠮᡝᡴᡠ |
| ᡝᡨᡠ᠋ᡴᡠ ᠠᡩᡠ᠋ | 11 | 3/11 | 0/11 | 0/11 | 78.5% | ᡝᡨᡠᡴᡠ ᠠᡩᡠ |
| ᠵᡠᡵᠠᡴᠠ | 6 | 2/11 | 0/11 | 0/11 | 78.8% | ᠵᡠᡵᠠᡴᠠᠨ |
| ᠠᠵᡳᡤᡝ ᠮᡳᠶᠣᠣᠴᠠᠨ | 14 | 0/11 | 0/11 | 0/11 | 79.2% | ᠠᠵᡳᡤᡝ ᠮᡳᠶᠣᠣᠴᠨ |
| ᠵᠠᡴ᠋ᡩ᠋ᠠᠨ | 8 | 0/11 | 0/11 | 0/11 | 79.5% | ᠵᠠᡴ᠋ᡩᡝᠨ |
| ᡨᡝ᠋ᡵᡝ | 5 | 2/11 | 0/11 | 0/11 | 80.0% | ᡨᡝᡵᡝ |
| ᡨᡠ᠋ᡤᡳ | 5 | 5/11 | 1/11 | 0/11 | 80.0% | ᡨᡠᡤᡳ |
| ᡴᠠᠯᠠᠩ ᡴᡳᠯᡳᠩ | 11 | 1/11 | 2/11 | 0/11 | 80.2% | ᠠᠯᠩ ᡴᡳᠯᡳᠩ |
| ᡤᡠᡵᡠᠨ ᡳ᠋ ᡝᡶᡠ | 12 | 0/11 | 0/11 | 0/11 | 80.3% | ᡤᡠᡵᡠᠨ ᡝᡳ ᡝᡶᡠ |

Run 25 vs run 22: run 25 reads 691 lexigraphs in more fonts and 664 in fewer. Image by image: both right 14996, only run 22 2256, only run 25 2342, both wrong 2285.

Run 25 vs run 24: run 25 reads 1316 lexigraphs in more fonts and 241 in fewer. Image by image: both right 12555, only run 24 1294, only run 25 4783, both wrong 3247.

Run 24's worst regressions from run 22, and how run 25 does on them:

| Lexigraph | Run 22 | Run 24 | Run 25 |
|---|---:|---:|---:|
| ᠨᡳᡴᠠᠨ ᡤᡳᠰᡠᠨ | 11/11 | 0/11 | 3/11 |
| ᠮᠠᡶᠠ ᡤᡠᡵᡠᠨ | 11/11 | 0/11 | 8/11 |
| ᡤᡡᠰᡳᠨ ᠵᡠᠸᡝ | 11/11 | 0/11 | 9/11 |
| ᠪᡠᠯᠠ | 11/11 | 1/11 | 10/11 |
| ᠪᡳᠶᠠ ᠵᡝᠮᠪᡳ | 11/11 | 1/11 | 10/11 |
| ᠰᡠᠨᠵᠠᠩᡤᡝᡵᡳ | 10/11 | 0/11 | 6/11 |
| ᡠᠶᠠᠨ ᠯᠠᠯᠠ | 10/11 | 0/11 | 9/11 |
| ᡤᡳᡵᠠᠩᡤᡳ | 11/11 | 1/11 | 10/11 |
| ᡩᡠᡴᠠ | 10/11 | 0/11 | 7/11 |
| ᡶᡝᡨᡝ᠋ᠮᠪᡳ | 10/11 | 0/11 | 1/11 |
| ᠨᡠᡵᡤᠠᠴᡳ | 10/11 | 1/11 | 5/11 |
| ᠪᡠᠯᠠ ᠰᠣᡤᡳ | 9/11 | 0/11 | 9/11 |

### Single vs multiple word lexigraphs at matched lengths (exact match, own data)

| Label length | Single: 22 | Single: 24 | Single: 25 | Multiple: 22 | Multiple: 24 | Multiple: 25 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 91.1% (3344) | 82.5% | 91.6% | 72.7% (11) | 81.8% | 100.0% |
| 6 | 80.4% (3740) | 65.3% | 80.7% | 67.5% (77) | 55.8% | 85.7% |
| 7 | 80.3% (3003) | 58.9% | 79.4% | 77.6% (143) | 57.3% | 79.7% |
| 8 | 74.2% (1265) | 49.6% | 73.6% | 67.3% (165) | 33.9% | 78.2% |
| 9 | 73.0% (704) | 51.8% | 69.3% | 65.0% (605) | 41.5% | 72.2% |
| 10 | 74.7% (253) | 40.3% | 65.2% | 61.1% (517) | 32.3% | 64.4% |
| 11 | 68.2% (132) | 54.5% | 72.0% | 55.7% (539) | 36.7% | 57.5% |
| 12 | 81.8% (11) | 18.2% | 54.5% | 44.5% (418) | 24.2% | 49.0% |
| 13+ | — | — | — | 22.5% (781) | 12.9% | 31.6% |

## Position in the image


Run 24 on padded data: exact match by where the word starts and how tall it is (cells with < 30 images left out).

| Word height (tokens) | starts tok 0-7 | starts tok 8-15 | starts tok 16-23 | starts tok 24-31 | starts tok 32-39 | starts tok 40-47 | starts tok 48-55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-15 | 77.2% | 75.8% | 80.8% | 79.5% | 78.3% | 78.1% | 76.6% |
| 16-23 | 67.4% | 69.9% | 71.1% | 69.0% | 71.3% | 71.6% | — |
| 24-31 | 54.7% | 56.5% | 58.5% | 53.9% | 58.8% | — | — |
| 32-47 | 35.8% | 41.7% | 43.7% | 40.2% | — | — | — |

Run 25 on padded data (every image padded, so positions are spread evenly): exact match by where the word starts and how tall it is (cells with < 30 images left out).

| Word height (tokens) | starts tok 0-7 | starts tok 8-15 | starts tok 16-23 | starts tok 24-31 | starts tok 32-39 | starts tok 40-47 | starts tok 48-55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-15 | 83.8% | 86.0% | 86.5% | 85.6% | 84.4% | 85.4% | 79.9% |
| 16-23 | 83.5% | 80.4% | 82.0% | 81.7% | 80.8% | 82.1% | — |
| 24-31 | 75.7% | 74.4% | 73.5% | 71.6% | 72.2% | — | — |
| 32-47 | 59.7% | 62.7% | 60.5% | 63.5% | — | — | — |

By exact number of blank rows above the word, on the padded data, with the same images top-aligned:

| Blank rows above | Images | Mean word height (tokens) | Run 24 padded | Run 24 top-aligned | Run 25 padded | Run 25 top-aligned |
|---|---:|---:|---:|---:|---:|---:|
| 0-0 | 197 | 36.9 | 37.1% | 37.1% | 57.9% | 57.9% |
| 1-3 | 423 | 28.2 | 54.1% | 41.4% | 74.5% | 68.3% |
| 4-7 | 626 | 27.4 | 55.6% | 39.8% | 72.8% | 69.6% |
| 8-15 | 1165 | 25.8 | 55.8% | 42.8% | 73.5% | 73.3% |
| 16-31 | 2235 | 25.4 | 57.8% | 42.6% | 74.9% | 74.1% |
| 32-63 | 4372 | 24.6 | 60.7% | 43.3% | 75.1% | 74.9% |
| 64-+ | 12861 | 20.9 | 66.9% | 48.9% | 78.4% | 79.4% |

Run 25 on its own data: unpadded images 80.5% exact (7272 images), padded 78.9% (14553). Mean word height: 22.8 tokens unpadded vs 22.7 padded. Same unpadded images, word height 0-23 tokens only: 86.1% unpadded vs 85.1% padded.

**Shift test.** 2,000 random short single words (≤ 24 tokens), top-aligned and then moved down by prepending blank tokens. Leading spaces in the guess are ignored when scoring.

| Shift (tokens) | 0 | 4 | 8 | 16 | 24 | 32 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run 22 exact | 86.1% | 59.9% | 44.2% | 30.0% | 19.7% | 10.4% | 4.6% |
| Run 24 exact | 54.6% | 66.8% | 67.0% | 68.8% | 67.3% | 67.4% | 67.3% |
| Run 25 exact | 82.3% | 80.9% | 80.5% | 79.7% | 80.2% | 79.6% | 79.2% |
| Run 22 guess starts with a space | 0.0% | 87.2% | 96.2% | 99.3% | 99.4% | 99.9% | 100.0% |

The same word, in the same font, read as part of a phrase vs on its own (VS stripped; phrases where the number of words was right; own data):

| Word's position | Run 22 in phrase | Run 22 standalone | Run 24 in phrase | Run 24 standalone | Run 25 in phrase | Run 25 standalone |
|---|---:|---:|---:|---:|---:|---:|
| 1st | 85.7% | 91.8% | 57.1% | 79.9% | 77.5% | 89.8% |
| 2nd | 62.1% | 89.1% | 54.2% | 77.3% | 75.0% | 90.0% |
| 3rd+ | 31.0% | 97.4% | 44.4% | 84.4% | 54.3% | 94.4% |

### Length: per-character error of single words (VS stripped, own data)

| Length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 22 per-char error | 3.4% | 3.8% | 2.5% | 2.8% | 3.0% | 2.2% | 3.3% | 2.3% | 1.9% | 4.4% | 1.5% |
| Run 24 per-char error | 6.2% | 7.3% | 5.1% | 6.0% | 6.7% | 6.5% | 7.8% | 5.9% | 7.7% | 6.1% | 8.3% |
| Run 25 per-char error | 5.6% | 3.1% | 2.3% | 2.9% | 2.9% | 2.6% | 3.2% | 3.0% | 2.8% | 3.4% | 4.5% |
| Run 22 exact (no VS) | 93.8% | 90.0% | 90.9% | 88.2% | 85.8% | 86.9% | 79.1% | 83.0% | 84.0% | 66.1% | 81.8% |
| Run 24 exact (no VS) | 88.6% | 82.0% | 82.5% | 75.9% | 69.7% | 64.1% | 54.3% | 61.1% | 48.1% | 52.9% | 18.2% |
| Run 25 exact (no VS) | 89.9% | 92.6% | 92.1% | 87.9% | 85.7% | 84.7% | 79.4% | 77.5% | 77.1% | 70.2% | 54.5% |

Overall per-character error for single words: run 22 2.7%, run 24 6.4%, run 25 2.8%.

### Letters: share of each letter's occurrences misread (own data)

| Letter | Occurrences in lexicon | Run 22 | Run 24 | Run 25 |
|---|---:|---:|---:|---:|
| FVS3 | 2 | 77.3% | 86.4% | 90.9% |
| ᠭ | 1 | 100.0% | 54.5% | 81.8% |
| FVS2 | 27 | 83.2% | 80.5% | 57.6% |
| FVS1 | 141 | 47.7% | 67.7% | 42.4% |
| ᠇ | 1 | 9.1% | 72.7% | 36.4% |
| ᡦ | 21 | 56.7% | 28.1% | 34.2% |
| ᠺ | 1 | 18.2% | 9.1% | 27.3% |
| ᡮ | 2 | 9.1% | 50.0% | 27.3% |
| ᡟ | 2 | 18.2% | 18.2% | 22.7% |
| ᠤ | 4 | 0.0% | 15.9% | 18.2% |
| ᡬ | 7 | 26.0% | 31.2% | 18.2% |
| ᡱ | 1 | 0.0% | 18.2% | 18.2% |
| ᡰ | 3 | 15.2% | 42.4% | 15.2% |
| ᡷ | 4 | 0.0% | 9.1% | 13.6% |
| ᡯ | 12 | 18.9% | 15.2% | 9.8% |
| ᠳ | 1 | 0.0% | 36.4% | 9.1% |
| ᡴ | 303 | 5.0% | 17.9% | 7.7% |
| ᠊ | 21 | 16.9% | 24.2% | 6.5% |
| ᠸ | 132 | 3.7% | 13.7% | 5.0% |
| ᠠ | 1538 | 5.6% | 16.7% | 4.9% |
| ᠨ | 987 | 2.9% | 5.8% | 3.8% |
| ᡶ | 203 | 1.3% | 5.5% | 3.7% |
| ᡝ | 981 | 5.8% | 8.1% | 3.1% |
| ᡳ | 1376 | 3.4% | 4.7% | 2.6% |
| ᠪ | 506 | 1.4% | 5.3% | 0.4% |

## Variation selectors

Visibility comes from run 22's `variation_selector_visibility.csv`; padding doesn't change how glyphs are drawn.

| | Images | Run 22 exact | Run 22 ignoring VS | Run 24 exact | Run 24 ignoring VS | Run 25 exact | Run 25 ignoring VS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selector invisible | 882 | 20.7% | 81.5% | 13.4% | 64.9% | 32.4% | 79.6% |
| Selector visible | 614 | 56.7% | 73.9% | 35.0% | 61.9% | 61.7% | 80.8% |

- FVS1 recall: run 22 52.3%, run 24 32.3%, run 25 57.6% (1551 occurrences across all images)
- FVS2 recall: run 22 16.8%, run 24 19.5%, run 25 42.4% (297 occurrences across all images)
- FVS3 recall: run 22 22.7%, run 24 13.6%, run 25 9.1% (22 occurrences across all images)
