# Word-level evaluation: D-8/64 Adam 3e-3 (run 22)

Weights: `weights_D-8-64_Adam_3e-3.npz` (final epoch). Evaluated 2026-10-25 in eval mode (no dropout), greedy CTC decoding, on **every lexigraph × every font**: 1,989 lexigraphs × 11 fonts = 21,879 images. Per-image results (guess, edit distance, image height...) are in `word_eval_D-8-64_Adam_3e-3.csv`.

**Caveat — train and test are mixed.** `generate_seeds` splits randomly and the split wasn't saved, so ~70% of these images were trained on. The log suggests this barely matters: over the last epoch, train and test character accuracy were about equal (0.934 vs 0.939; train measured with dropout on), so the model isn't memorising, and these numbers should be close to held-out numbers.

Columns:
- **Exact match** — the guess equals the label exactly.
- **Exact, ignoring variation selectors** — exact after deleting FVS1–4 (U+180B–D, U+180F) from both. This is generous: where a selector changes the glyph, missing it is a real error.
- **Mean char accuracy** — the training metric, `1 − edit distance / label length`, floored at 0.

## Key findings

1. **Word accuracy is 78.9%** while character accuracy is 95.8%. Most misreads are near-misses: 57% of wrong guesses are off by one letter, 17% by two, 16% are wrong *only* in variation selectors.
2. **Variation selectors are the biggest single error source.** Lexigraphs containing one get 35.5% exact vs 82.0% without. Ignoring the selectors, they jump to 78.4% — close to the rest. Dropping FVS1 (609 times) is the most common edit of all, FVS2 (208) is 7th. The model mostly can't predict them, which is expected given they're often invisible.
3. **Length is the strongest driver.** Exact match is ~90% up to 4 characters and falls steadily: 80% at 6–7, 66% at 10, 45% at 12, 30% at 13–15, 4% at 16+. With ~3–4% of characters wrong independently, a word's chance of being all right decays roughly exponentially with length.
4. **Multi-word lexigraphs: 50.8% vs 83.8% exact**, but much of that is length (multi-word labels are long). At matched lengths they're still worse by ~5–15 points (e.g. 9 chars: 73.0% single vs 65.0% multi; 11 chars: 68.2% vs 55.7%). Spaces themselves are fine: the word count is right 99.4% of the time and only 7 errors were space-only.
5. **Images taller than 256 px are hopeless** (54 images, 3.7% exact): they're squashed to fit, so letters are distorted. They make up a quarter of the 16+ bucket.
6. **Fonts range 66.9% (LiuYe) to 85.6% (ZhengHei).** LiuYe and WenJian are clearly hardest, then ShuKai; the "plain" fonts (ZhengHei, YaBai, ZhengBai, BiaoHei) are best.
7. **Characteristic confusions:** dropping vowels ᠠ/ᡝ/ᡳ (435/326/266), inserting a spurious ᠨ (286) or reading ᠠ as ᠨ (163) — the tooth-shaped strokes; ᡦ→ᠪ (105, they differ only by a small mark); ᡨ→ᡩ (70) and ᡥ→ᡤ (65), which are the dotted/undotted pairs.
8. 697 lexigraphs (35%) are read right in all 11 fonts; 71 (3.6%) are wrong in every font.

## Tables

### Overall


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| All images | 21879 | 78.85% | 82.28% | 95.77% | 0.317 |

### By font


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| XM_BiaoHei | 1989 | 82.86% | 86.43% | 96.85% | 0.238 |
| XM_GuFeng | 1989 | 80.69% | 84.26% | 96.23% | 0.281 |
| XM_LiuYe | 1989 | 66.92% | 69.78% | 93.26% | 0.495 |
| XM_ShuKai | 1989 | 74.16% | 78.28% | 94.71% | 0.374 |
| XM_WenJian | 1989 | 72.30% | 75.82% | 93.40% | 0.514 |
| XM_WenQin | 1989 | 79.64% | 82.60% | 95.95% | 0.310 |
| XM_XingShu | 1989 | 77.53% | 80.89% | 95.47% | 0.339 |
| XM_YaBai | 1989 | 84.41% | 87.68% | 97.14% | 0.216 |
| XM_YingBi | 1989 | 79.54% | 82.55% | 95.94% | 0.301 |
| XM_ZhengBai | 1989 | 83.76% | 87.48% | 97.01% | 0.220 |
| XM_ZhengHei | 1989 | 85.57% | 89.29% | 97.52% | 0.196 |

### Variation selectors


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| With variation selector(s) | 1496 | 35.49% | 78.41% | 87.34% | 1.008 |
| Without | 20383 | 82.03% | 82.56% | 96.39% | 0.266 |

### By label length (characters, incl. spaces & variation selectors)


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| 1 | 341 | 88.27% | 88.86% | 88.27% | 0.117 |
| 2 | 374 | 91.18% | 94.12% | 95.32% | 0.094 |
| 3 | 990 | 90.20% | 90.20% | 96.30% | 0.111 |
| 4 | 3355 | 91.00% | 91.54% | 97.48% | 0.101 |
| 5 | 4466 | 86.45% | 88.71% | 96.83% | 0.159 |
| 6 | 3817 | 80.17% | 84.83% | 95.84% | 0.250 |
| 7 | 3146 | 80.13% | 86.46% | 96.48% | 0.246 |
| 8 | 1430 | 73.36% | 79.58% | 95.79% | 0.337 |
| 9 | 1309 | 69.29% | 74.18% | 95.18% | 0.434 |
| 10 | 770 | 65.58% | 73.12% | 95.23% | 0.477 |
| 11 | 671 | 58.12% | 58.72% | 93.66% | 0.697 |
| 12 | 429 | 45.45% | 46.15% | 92.68% | 0.879 |
| 13-15 | 550 | 30.36% | 33.82% | 89.57% | 1.442 |
| 16+ | 231 | 3.90% | 5.63% | 80.92% | 3.957 |

### Single vs multiple word lexigraphs


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| Single word | 18623 | 83.76% | 87.39% | 96.38% | 0.200 |
| Multiple words | 3256 | 50.77% | 53.07% | 92.29% | 0.986 |

### Image squashed to fit 256 px (original height > 256)


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| Fits | 21825 | 79.04% | 82.47% | 95.82% | 0.303 |
| Squashed | 54 | 3.70% | 3.70% | 75.05% | 5.981 |

### Multi-word lexigraphs by number of words


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| 1 words | 18623 | 83.76% | 87.39% | 96.38% | 0.200 |
| 2 words | 2981 | 54.38% | 56.79% | 93.20% | 0.766 |
| 3 words | 209 | 11.48% | 12.92% | 85.02% | 2.502 |
| 4 words | 44 | 18.18% | 18.18% | 79.81% | 3.727 |

### Single-word lexigraphs by length (isolating length from spaces)


| Group | Images | Exact match | Exact, ignoring variation selectors | Mean char accuracy | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| 1 | 341 | 88.27% | 88.86% | 88.27% | 0.117 |
| 2 | 374 | 91.18% | 94.12% | 95.32% | 0.094 |
| 3 | 990 | 90.20% | 90.20% | 96.30% | 0.111 |
| 4 | 3344 | 91.06% | 91.60% | 97.50% | 0.100 |
| 5 | 4466 | 86.45% | 88.71% | 96.83% | 0.159 |
| 6 | 3740 | 80.43% | 85.16% | 95.90% | 0.246 |
| 7 | 3003 | 80.25% | 86.81% | 96.49% | 0.246 |
| 8 | 1265 | 74.15% | 80.79% | 95.93% | 0.326 |
| 9 | 704 | 73.01% | 81.82% | 95.69% | 0.388 |
| 10 | 253 | 74.70% | 83.79% | 96.64% | 0.336 |
| 11 | 132 | 68.18% | 68.18% | 95.80% | 0.462 |
| 12 | 11 | 81.82% | 81.82% | 98.48% | 0.182 |

### Anatomy of the wrong guesses

4627 of 21879 images were read wrong.

| Kind of mistake | Images | Share of wrong |
|---|---:|---:|
| one letter wrong | 2630 | 56.8% |
| two letters wrong | 809 | 17.5% |
| only variation-selector mistakes | 750 | 16.2% |
| 3+ edits | 431 | 9.3% |
| only space (and/or VS) mistakes | 7 | 0.2% |

Edit operations (from difflib alignment): delete 2668, replace 2040, insert 1343

Most common edits (expected → guessed; ∅ = nothing):

| Expected | Guessed | Count |
|---|---|---:|
| FVS1 | ∅ | 609 |
| ᠠ | ∅ | 435 |
| ᡝ | ∅ | 326 |
| ∅ | ᠨ | 286 |
| ᡳ | ∅ | 266 |
| ∅ | ᠠ | 223 |
| FVS2 | ∅ | 208 |
| ᠠ | ᠨ | 163 |
| ∅ | ᡴ | 133 |
| ᠨ | ∅ | 127 |
| ∅ | FVS1 | 127 |
| ᡦ | ᠪ | 105 |
| ᡝ | ᠠ | 83 |
| ᡨ | ᡩ | 70 |
| ᡥ | ᡤ | 65 |
| ᠯ | ∅ | 61 |
| ∅ | ᡝ | 60 |
| ∅ | ᡳ | 58 |
| ᠨ | ᠠ | 54 |
| ∅ | ᠩ | 52 |
| ᠶ | ᠸ | 47 |
| ᡴ | ∅ | 46 |
| ᠴ | ∅ | 43 |
| ᠶ | ∅ | 43 |
| ᡵ | ∅ | 42 |

### How many fonts each lexigraph is misread in

| Fonts misread (of 11) | Lexigraphs |
|---:|---:|
| 0 | 697 |
| 1 | 453 |
| 2 | 221 |
| 3 | 161 |
| 4 | 103 |
| 5 | 65 |
| 6 | 54 |
| 7 | 36 |
| 8 | 43 |
| 9 | 39 |
| 10 | 46 |
| 11 | 71 |

Hardest 20 lexigraphs:

| Lexigraph | Length | Fonts right | Mean char acc | Example guess |
|---|---:|---:|---:|---|
| ᡶᡳᠴᠠᡴᡡ | 6 | 0/11 | 56.1% | ᡶᠴᡳᡡ |
| ᠭ᠋ᠤᠰᠢᠨ | 6 | 0/11 | 57.6% | ᠳᠤᠰᠢᠨ |
| ᡩᠣᡵᡤᡳ ᠪᠠᡳᡨᠠ ᠪᡝ ᡠᡥᡝᡵᡳ ᡴᠠᡩᠠᠯᠠᡵᠠ ᠶᠠᠮᡠᠨ | 35 | 0/11 | 57.7% | ᡩᡵᡤᡳ ᠪᡨᠠ ᠪ ᡠᡤᡵᡳ ᡳᡵᠯᡵᠨ ᠠᠮᡠᠨ |
| ᡦᠣᠴᠣᠩ | 5 | 0/11 | 61.8% | ᠪᠪᠣᠣᠩ |
| ᡩ᠋᠋ᡝ᠋ | 5 | 0/11 | 63.6% | ᡩ᠋᠋ᡝ |
| ᠨᠠᡵᡥᡡᠨ ᡝᡩᡠᠨ ᡨᡠᠸᠠᡵᠠ ᠨᡳᠮᡝᡴᡠ | 25 | 0/11 | 66.5% | ᠨᠠᡵᡥᡡᠨ ᡩᡠᠨ ᡨᡠᠸᠨ ᠨᠮᡝᡠ |
| ᠪᡝ ᠯᡳᡳ ᡥᡳ ᡨ᠋ᡳᠶᠠᠨ ᡩ᠋ᡝ᠋ | 21 | 0/11 | 66.7% | ᠪᡝ ᠯᡳᡳ ᡥᡳ ᡨᡳᠶᡝ ᡩᡝ |
| ᡡᠯᡝᡨ | 4 | 0/11 | 70.5% | ᡡᠯᡝᡨᠨ |
| ᠨᠠᡵᡥᡡᠨ ᡝᡩᡠᠨ ᡨᡠᠸᠠᠮᠪᡳ | 19 | 0/11 | 70.8% | ᠨᠠᡵᡥᡡᠨ ᡩᡠᠨ ᡨᡠᡠᠮᠪ |
| ᠰᠠᡵᡦᠠ | 5 | 0/11 | 70.9% | ᠰᠠᡵᠪᠠ |
| ᠰᡠᡩᡠᡵᡳ ᠪᡳᡨ᠌ᡥᡝ | 13 | 0/11 | 71.3% | ᠰᡠᡩᡠᡵᡳ ᠪᡳᠣᡤ |
| ᠣᡴ᠋ᡨ᠋ᠣ | 6 | 0/11 | 72.7% | ᠣᡴᡨᡴᠣ |
| ᡠᡴ᠋ᠰᡳ᠍ᠨ | 7 | 0/11 | 72.7% | ᡠᡴᠰᡳᠨ |
| ᡳᠰ ᠪᠠ ᠨᡳ ᠶᠠ | 11 | 0/11 | 74.4% | ᡳᠰ ᠪᠠ ᠨᡳ ᡳᠨ |
| ᡥᡡᠸᠠᠩ ᡥᠣ ᠪᡳᡵᠠ | 13 | 0/11 | 74.8% | ᡥᡡᠸᠠᠩ ᡥᠣ ᠣᡵᠨ |
| ᠵᠠᡴ᠋ᡩ᠋ᠠᠨ | 8 | 0/11 | 75.0% | ᠵᠠᡴᡩᠠᠨ |
| ᠪᡝ ᠯᡳᠶᠠᠨ ᡨ᠋ᠠᠴᡳᡥᡳᠶᠠᠨ | 19 | 0/11 | 75.1% | ᠪᡝ ᠯᡳᠶᠠᠨ ᡨᠠᠴᡥᡳᠶᠨ |
| ᡨᡝᡳᡴᡝ ᡨᡝᡳᡴᡝ | 11 | 0/11 | 75.2% | ᡨᡝᡳᡳᡴᡝ ᡨᡳᡴ |
| ᠠᠵᡳᡤᡝ ᠮᡳᠶᠣᠣᠴᠠᠨ | 14 | 0/11 | 75.3% | ᠠᠵᡳᡤᡝ ᠮᡳᠶᠣᠣᠴᠨ |
| ᡝᠨᡩᡝᠮᠪᡳ | 7 | 0/11 | 75.3% | ᠨᡩᡝᠮᠪᡳ |

### Single vs multiple word lexigraphs at matched lengths (exact match)

| Label length | Single word | Multiple words |
|---:|---:|---:|
| 4 | 91.1% (3344) | 72.7% (11) |
| 6 | 80.4% (3740) | 67.5% (77) |
| 7 | 80.3% (3003) | 77.6% (143) |
| 8 | 74.2% (1265) | 67.3% (165) |
| 9 | 73.0% (704) | 65.0% (605) |
| 10 | 74.7% (253) | 61.1% (517) |
| 11 | 68.2% (132) | 55.7% (539) |
| 12 | 81.8% (11) | 44.5% (418) |
| 13+ | — | 22.5% (781) |

Within multi-word lexigraphs (where the guess had the right number of words), 69.5% of the individual words were read exactly.

### Notes on the tables
- Length-1 lexigraphs have char accuracy equal to exact match, since one wrong character floors the score at 0.
- Several of the "hardest" lexigraphs are hard because of selectors (e.g. ᡩ᠋᠋ᡝ᠋, ᠣᡴ᠋ᡨ᠋ᠣ, ᠵᠠᡴ᠋ᡩ᠋ᠠᠨ are read correctly apart from them), or because the 35-character one is squashed. ᠭ᠋ᠤᠰᠢᠨ is in Mongolian letters (ᠭ/ᠤ/ᠢ) rather than Manchu ones, and is read with ᠳ.

## Follow-up: is the model just bad at what's underrepresented?

Short answer: partly, but under-representation doesn't explain most of the misses. The biggest real problem is **where in the image a word sits**.

### Composition of the training data

Every lexigraph is rendered once per font, so fonts are exactly balanced (1/11 each). Font differences reflect how hard each font is to read, not how much data there is.

| Characteristic | Share of the lexicon |
|---|---|
| Lexigraphs with a variation selector | 6.8% (136 of 1,989); selectors are 1.3% of characters |
| Multiple words | 14.9% (296) |
| Length 1–3 / 4–7 / 8–12 / 13–15 / 16+ | 7.8% / 67.5% / 21.1% / 2.5% / 1.1% |

### Length: the drop is compounding, not a lack of data

With variation selectors stripped from both label and guess, the per-character error rate of **single words** is flat across lengths, even the rare ones:

| Length (single words) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Per-char error | 3.4% | 3.8% | 2.6% | 2.8% | 3.0% | 2.2% | 3.3% | 2.3% | 2.0% | 4.4% | 1.5% |
| Exact (no VS) | 93.8% | 90.0% | 90.9% | 88.2% | 85.8% | 86.9% | 79.1% | 83.0% | 84.0% | 66.1% | 81.8% |
| Predicted by a constant 2.4%/char | 95.3% | 93.1% | 90.9% | 88.7% | 86.6% | 84.6% | 82.6% | 80.7% | 78.8% | 76.9% | 75.1% |

10-letter words are 3.5% of the lexicon but have the lowest error rate. Long single words lose exact matches only because each letter is another chance to make a mistake.

### Letters: rare ones suffer, but so do some common ones

| Letter | Occurrences in lexicon | Share of its occurrences misread |
|---|---:|---:|
| ᡦ | 21 | 56.7% (mostly → ᠪ) |
| ᡬ / ᡯ / ᠊ | 7 / 12 / 21 | 26.0% / 18.9% / 16.9% |
| FVS1 / FVS2 | 141 / 27 | 47.7% / 83.2% |
| ᠸ / ᡧ (similarly rare, for comparison) | 132 / 98 | 3.7% / 1.1% |
| ᠶ / ᡨ / ᠴ / ᡝ / ᠠ (common) | 263 / 253 / 232 / 981 / 1538 | 7.6% / 6.1% / 5.7% / 5.8% / 5.6% |

The very rare letters (ᡦ, ᡬ, ᡯ) look short of data. The common but error-prone letters are visually ambiguous: tooth strokes, and dotted vs undotted pairs. Variation selectors are about as common as ᠸ but missed 13× as often, because they're often invisible (see below).

### Multi-word lexigraphs: position in the image

The same word, in the same font, read as part of a phrase vs on its own (VS stripped, phrases where the number of words was right):

| Word's position in phrase | Exact in phrase | Same word standalone |
|---|---:|---:|
| 1st | 89.9% | 91.8% |
| 2nd | 63.1% | 89.5% |
| 3rd+ | 36.1% | 97.4% |

The spaces aren't the problem: the number of words is right 99.4% of the time. What changes is how far down the image the word starts, and those positions are rare in the training data:

| Images reaching token | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|
| Share | 79.0% | 39.1% | 15.6% | 5.9% | 2.1% | 0.8% | 0.3% |

(Median image: 21 tokens.) Single words show it too: per-character error is 2.3% for images of 16–23 tokens, 4.0% at 32–39, and 5.7% at 40+.

**Shift test.** 2,000 random short single words (≤ 24 tokens) were shifted down the image by prepending blank tokens. The model reads the leading blank as a space, so the leading spaces are ignored when scoring:

| Shift (tokens) | 0 | 4 | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|
| Exact | 86.8% | 59.7% | 45.1% | 31.1% | 21.7% | 10.0% | 3.8% |

No training image starts with blank space, so part of this drop is the blank space being unfamiliar. Combined with the phrase results, though, it strongly suggests the model has learned letters at particular absolute positions, and later positions are undertrained. This also fits the near-total failure on images over 256 px, which are squashed.

**Ideas:** randomly pad each training image with blank space above and below (this trains every position, and teaches the model that a gap at the edge isn't a space); try relative/learned positions; add more or concatenated multi-word training images; oversample rare letters.

## Follow-up: are the variation selectors visible?

Every one of the 136 selector lexigraphs was rendered in every font with and without its selectors, and the two images compared. Results per image are in `variation_selector_visibility.csv`; examples are in `variation_selector_examples.png`.

- **59.0% of the images (882 of 1,496) are pixel-identical without the selector.** The selector is invisible, so it can't be read from the image.
- By lexigraph: **76 of 136 are invisible in every font, 53 are visible in every font, 7 depend on the font.** Font hardly matters overall: 39–43% visible in each.
- The visible differences are real changes in letter shape. 567 of 614 change the word's height (usually by 2–5 px, up to 60), and the rest change at least 6 pixels strongly. Some are subtle, though: 114 images change height by only 1–2 px.

| | Images | Exact | Exact ignoring VS |
|---|---:|---:|---:|
| Selector invisible | 882 | 20.7% | 81.5% |
| Selector visible | 614 | 56.7% | 73.9% |

When the selector is invisible, the model still gets it right 20.7% of the time. It can only be doing that by memorising those lexigraphs, since there's nothing in the image to read: e.g. ᡶᡝᡨᡝ᠋ᠮᠪᡳ is exact in 10/11 fonts with an invisible FVS1. When it's visible, it's right only 57% of the time.

Which letter + selector pairs are visible. (Caveat: this table counts a lexigraph as visible if removing *all* of its selectors changes the image, so in lexigraphs with several selectors a visible FVS1 can make a neighbouring selector look visible too. See the FVS2 check below.)

| Pair | Visible | Lexigraphs |
|---|---|---:|
| ᡶ + FVS1 | always | 36 |
| ᡩ + FVS1 | always 12, never 12 | 24 |
| ᡠ + FVS1 | never 17, always 4, sometimes 5 | 26 |
| ᡳ + FVS2 | never 17, always 2, sometimes 1 | 20 |
| ᡝ + FVS1 | never 9, always 7, sometimes 1 | 17 |
| ᡨ + FVS1 | always 8, never 7, sometimes 1 | 16 |
| ᡴ + FVS1 | never 7, always 2 | 9 |
| others | | 17 |

Whether a selector shows depends on the letter *and* where it falls in the word: a selector only changes a letter's shape where the font has an alternate form for that position.

### Is FVS2 ever visible?

Checked by removing **only** the FVS2 from each of the 26 lexigraphs that contain one (other selectors left in), in all 11 fonts: **16 of 286 images (5.6%) change**, from just 2 lexigraphs.

| Lexigraph | Visible in | What changes | Model exact |
|---|---|---|---:|
| ᠨᠣᠮᡠᠨ ᡥᠠᠨ[2] | all 11 fonts | final ᠨ; the image is ~2 px shorter without it in most fonts | 0/11 |
| ᠮᡝᡳ[2]ᡨᡝ[1]ᠮᠪᡳ | 5 fonts (LiuYe, ShuKai, WenQin, XingShu, YingBi) | ᡳ before ᡨ; the image is 1–6 px longer with it | 0/11 |

In the other 24 FVS2 lexigraphs, FVS2 never shows in any font, including all 17 cases of ᡳ+FVS2 in the middle of a word. Three lexigraphs (ᡩ[1]ᡝ[1]ᡨ[2]ᡥᡝ, ᡩ[1]ᡠ[1]ᡳ[2]ᠨ, ᡩ[1]ᡠ[1]ᡳ[2]ᠨ ᠪᡳᠯᡨᡝᠨ) are visible in every font only because of their FVS1, so the all-selectors test above overstated FVS2 (49 visible images rather than 16). The image-level figure (59% of selector images unchanged) isn't affected, since it asks whether the lexigraph changes at all.

So in this data FVS2 is effectively invisible, and the model drops it 83% of the time. It's a good candidate for removing from the labels, unless the model should learn these two cases.

Examples (`variation_selector_examples.png` shows each in LiuYe and ZhengHei, with and without the selector):

| # | Lexigraph (selector shown as [n]) | Selector | Visible? | Fonts read exactly | Typical guess |
|---|---|---|---|---:|---|
| 1 | ᡶ[1]ᠣᡵᠣᠨ | ᡶ+FVS1 | always | 11/11 | ᡶ᠋ᠣᡵᠣᠨ |
| 2 | ᡥᠣᡨ[1]ᠣᠨ | ᡨ+FVS1 | always | 5/11 | ᡥᠣᡨᡴᠣᠨ |
| 3 | ᠣᡴ[1]ᡨ[1]ᠣ | ᡴ+FVS1, ᡨ+FVS1 | always | 0/11 | ᠣᡴᡨᡴᠣ |
| 4 | ᡨ[1]ᡳᠶᠠᠨ ᠵᡳᠨ | ᡨ+FVS1 | 5 of 11 fonts | 2/11 | ᡨ᠋ᡳᠶᠠᠨ ᠵᡳᠨ |
| 5 | ᠪᡝᡳ[2]ᠰᡝ | ᡳ+FVS2 | never | 0/11 | ᠪᡝᡳᠰᡝ |
| 6 | ᡨᡠ[1]ᡵᠠᡴᡳ | ᡠ+FVS1 | never | 0/11 | ᡨᡠᡵᠠᡴᡳ |
| 7 | ᠨᠠᡩ[1]ᠠᠨ | ᡩ+FVS1 | never | 1/11 | ᠨᠠᡩᠠᠨ |
| 8 | ᡶᡝᡨᡝ[1]ᠮᠪᡳ | ᡝ+FVS1 | never | 10/11 | ᡶᡝᡨᡝ᠋ᠮᠪᡳ |

In examples 2 and 3 the visible selector form of ᡨ is misread as an extra ᡴ: the model sees the different shape but doesn't know what to call it. One lexicon entry, ᡩ᠋᠋ᡝ᠋ (ᡩ + FVS1 + FVS1 + ᡝ + FVS1), has a doubled selector, which is probably a data error.
